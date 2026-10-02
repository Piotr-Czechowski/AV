"""Adapter between CarlaEnv and the A3C worker.

CarlaEnv owns the low-level CARLA objects. This wrapper keeps those details out
of the worker loop and exposes only:

    reset() -> obs
    step(action) -> (obs, reward, done, info)
    episode_stats() -> dict for the episode logger
    reconnect()

``obs`` is a dict of NumPy values. ``rl_configuration.observation_spec``
describes it and every observation is checked against that spec:

    image     float32 [C, H, W] in [0, 1]
    speed     float32 [1], km/h divided by SPEED_SCALE
    maneuver  int64 scalar (0-d array): 0 = left, 1 = straight, 2 = right

The wrapper also handles action repeat, the episode length limit, reward
shaping, and optional frame saving.
"""

import os
import time
import numpy as np

from carla_env import CarlaEnv, GOAL_RADIUS_M
from rl_configuration import check_observation


SPEED_SCALE = 100.0  # the ``speed`` observation is km/h divided by this


class CarlaA3CWrapper:
    """Gym-like adapter: normalize observations, repeat actions, shape rewards."""

    def __init__(self, port, config, run_id, run_output_dir):
        """Connect to CARLA.

        Every option is a required field of ``config`` (the namespace built
        by ``train_a3c.py``). A missing field is an AttributeError, not a
        silent second default.
        """
        self.port = port
        self._scenario = config.scenario
        self._camera = config.camera
        self._obs_spec = config.obs_spec
        self._mp_density = config.mp_density
        self._map_name = config.map_name
        self.max_connect_retries = config.max_connect_retries
        self.connect_retry_wait = config.connect_retry_wait
        self.reconnect_wait = config.carla_timeout_wait
        self.n_actions = config.n_actions
        self._run_id = run_id
        self._run_output_dir = run_output_dir
        self._action_repeat = max(1, int(config.action_repeat))
        self._episode_max_decisions = int(config.episode_max_decisions)
        self._world_reload_interval = int(config.world_reload_interval)
        self._reward_mode = config.reward_mode
        self._reward_progress_coef = float(config.reward_progress_coef)
        self._reward_target_speed_coef = float(
            config.reward_target_speed_coef)
        self._reward_route_penalty_coef = float(
            config.reward_route_penalty_coef)
        self._reward_time_penalty = float(config.reward_time_penalty)
        self._reward_goal_bonus = float(config.reward_goal_bonus)
        self._reward_collision_penalty = float(
            config.reward_collision_penalty)
        self._reward_offroute_penalty = float(config.reward_offroute_penalty)
        self._reward_lane_invasion_penalty = float(
            config.reward_lane_invasion_penalty)
        self._reward_target_speed_kmh = float(config.reward_target_speed_kmh)
        self._reward_offroute_threshold = float(
            config.reward_offroute_threshold)
        self._reward_clip = float(config.reward_clip)
        self._verbose_env_logs = bool(config.verbose_env_logs)

        self._save_episodes = set(config.save_episodes) \
            if config.save_episodes else set()
        self._save_episode_interval = config.save_episode_interval
        self.global_episode = 0

        self.episode = 0
        self.step_count = 0
        self._episode_max_speed = 0.0
        self._episode_min_route_dist = float('inf')
        self._episode_goal_dist = float('inf')
        self._episode_reached_goal = False
        self._episode_reward_components = {}
        self._action_counts = np.zeros(self.n_actions, dtype=np.int64)
        self._prev_goal_dist = None
        self._prev_collision_count = 0

        self._maneuver_idx = 0
        self._current_maneuver = 1

        # Per-episode cache so frame saving creates the directory once and
        # does not rebuild the path in every step().
        self._save_dir_cached = None

        self.env = None
        self._connect_with_retries()

    def _connect_with_retries(self):
        """Create ``CarlaEnv``, retrying on failure up to ``max_connect_retries``."""
        # The camera resolution comes from the observation spec.
        _, height, width = self._obs_spec['image']['shape']
        for attempt in range(1, self.max_connect_retries + 1):
            try:
                self.env = CarlaEnv(
                    scenario=self._scenario, spawn_point=False,
                    terminal_point=False, mp_density=self._mp_density,
                    port=self.port,
                    camera=self._camera, resX=width, resY=height,
                    map_name=self._map_name,
                    manual_control=False,
                    verbose=self._verbose_env_logs,
                )
                return
            except Exception as e:
                if attempt == self.max_connect_retries:
                    raise
                print('[CARLA port:{}] attempt {}/{} failed: {}. '
                      'retrying in {}s...'.format(
                          self.port, attempt, self.max_connect_retries, e,
                          self.connect_retry_wait), flush=True)
                time.sleep(self.connect_retry_wait)

    def reconnect(self):
        """Destroy actors, drop the stale client, wait, then connect again."""
        try:
            if self.env is not None:
                try:
                    self.env.destroy_agents()
                except Exception:
                    pass
                if hasattr(self.env, 'world'):
                    self.env.world = None
                if hasattr(self.env, 'client'):
                    self.env.client = None
        except Exception:
            pass
        self.env = None
        print('[CARLA port:{}] waiting {}s for server restart...'.format(
            self.port, self.reconnect_wait), flush=True)
        time.sleep(self.reconnect_wait)
        self._connect_with_retries()

    def _state_to_chw_float(self, state):
        """Normalize a camera observation to ``[C, H, W]`` float32 in ``[0, 1]``."""
        channels, height, width = self._obs_spec['image']['shape']
        if isinstance(state, np.ndarray):
            arr = state
        elif hasattr(state, 'detach'):
            arr = state.detach().cpu().numpy()
        elif hasattr(state, 'cpu'):
            arr = state.cpu().numpy()
        else:
            arr = np.asarray(state)

        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[-1] == channels \
                and arr.shape[0] != channels:
            arr = np.transpose(arr, (2, 0, 1))
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape != (channels, height, width):
            raise ValueError('expected observation [{},{},{}], got {}'.format(
                channels, height, width, arr.shape))
        if not np.isfinite(arr).all():
            raise ValueError('observation contains NaN or inf')
        if arr.size and arr.max() > 1.0:
            arr = arr / 255.0
        if arr.min() < -1e-4 or arr.max() > 1.0 + 1e-4:
            raise ValueError('observation outside [0,1]: min={} max={}'
                             .format(float(arr.min()), float(arr.max())))
        return np.ascontiguousarray(arr)

    @staticmethod
    def _speed_to_float(speed):
        """Unwrap a tensor or array speed to a Python float."""
        if hasattr(speed, 'detach'):
            speed = speed.detach()
        if hasattr(speed, 'cpu'):
            speed = speed.cpu()
        if hasattr(speed, 'item'):
            speed = speed.item()
        return float(speed)

    def _observation(self, state, speed_kmh):
        """Build the observation dict and check it against the spec.

        A new input is a new key here, in ``observation_spec``, and in the
        model's ``forward``. The worker loop passes the dict through.
        """
        obs = {
            'image': self._state_to_chw_float(state),
            'speed': np.array([speed_kmh / SPEED_SCALE], dtype=np.float32),
            'maneuver': np.array(self._current_maneuver, dtype=np.int64),
        }
        check_observation(obs, self._obs_spec)
        return obs

    def _current_goal_distance(self):
        """Return Euclidean distance to the goal, or None if the query fails."""
        try:
            distance, _ = self.env.calculate_distance()
            return float(distance)
        except Exception:
            return None

    def _accumulate_reward_components(self, components):
        """Add named reward terms into the per-episode totals for the JSONL logger."""
        for key, value in components.items():
            self._episode_reward_components[key] = \
                self._episode_reward_components.get(key, 0.0) + float(value)

    def _shape_reward(self, legacy_reward, done, route_distance,
                      speed_kmh, distance_from_goal, collisions,
                      lane_invasions):
        """Return ``(reward, components)``. Legacy mode forwards CarlaEnv's reward."""
        if self._reward_mode == 'legacy':
            components = {'legacy': float(legacy_reward)}
            return float(legacy_reward), components

        route_distance = float(route_distance) \
            if route_distance is not None else self._reward_offroute_threshold
        distance_from_goal = float(distance_from_goal) \
            if distance_from_goal is not None else self._prev_goal_dist
        speed_kmh = float(speed_kmh)
        collisions = int(collisions or 0)
        lane_invasions = int(lane_invasions or 0)

        if self._prev_goal_dist is None or distance_from_goal is None:
            progress = 0.0
        else:
            progress = self._prev_goal_dist - distance_from_goal
        progress = float(np.clip(progress, -5.0, 5.0))

        target = max(1.0, self._reward_target_speed_kmh)
        speed_score = 1.0 - min(2.0, abs(speed_kmh - target) / target)
        offroute = route_distance >= self._reward_offroute_threshold
        new_collision = collisions > self._prev_collision_count
        reached_goal = bool(done and distance_from_goal is not None and
                            distance_from_goal < GOAL_RADIUS_M)

        components = {
            'progress': self._reward_progress_coef * progress,
            'target_speed': self._reward_target_speed_coef * speed_score,
            'route_penalty': -self._reward_route_penalty_coef *
            min(route_distance, self._reward_offroute_threshold),
            'time_penalty': -self._reward_time_penalty,
            'goal_bonus': self._reward_goal_bonus if reached_goal else 0.0,
            'collision_penalty': -self._reward_collision_penalty
            if new_collision else 0.0,
            'offroute_penalty': -self._reward_offroute_penalty
            if offroute else 0.0,
            'lane_invasion_penalty': -self._reward_lane_invasion_penalty *
            lane_invasions,
        }
        reward = float(sum(components.values()))
        if self._reward_clip > 0:
            reward = float(np.clip(reward, -self._reward_clip,
                                   self._reward_clip))
        components['total'] = reward
        return reward, components

    def _should_save_this_episode(self):
        """Return True if this episode should dump camera frames."""
        if self.global_episode in self._save_episodes:
            return True
        if self._save_episode_interval > 0 \
                and self.global_episode > 0 \
                and self.global_episode % self._save_episode_interval == 0:
            return True
        return False

    def reset(self):
        """Reset episode stats and the env. Returns the first observation."""
        self.episode += 1
        self.step_count = 0
        self._episode_max_speed = 0.0
        self._episode_min_route_dist = float('inf')
        self._episode_goal_dist = float('inf')
        self._episode_reached_goal = False
        self._episode_reward_components = {}
        self._action_counts = np.zeros(self.n_actions, dtype=np.int64)
        self._prev_goal_dist = None
        self._prev_collision_count = 0

        save_images = self._should_save_this_episode()
        self._save_images = save_images
        # New episode: new save dir; create it on the first saved frame.
        self._save_dir_cached = None

        if hasattr(self.env, 'state_observer'):
            if self._run_output_dir:
                self.env.state_observer.output_dir = os.path.join(
                    self._run_output_dir, 'images')
            self.env.state_observer.reset()

        full_reload = self._world_reload_interval > 0 and \
            self.episode % self._world_reload_interval == 0
        state, speed = self.env.reset(
            save_image=save_images, episode=self.global_episode,
            reload_world=full_reload)

        self._prev_goal_dist = self._current_goal_distance()
        if self._prev_goal_dist is not None:
            self._episode_goal_dist = self._prev_goal_dist
        if hasattr(self.env, 'collision_history_list'):
            self._prev_collision_count = len(self.env.collision_history_list)

        if hasattr(self.env, 'car_decisions') and self.env.car_decisions:
            self._car_decisions = list(self.env.car_decisions)
        else:
            self._car_decisions = [1]
        self._maneuver_idx = 0
        self._current_maneuver = int(self._car_decisions[0])

        return self._observation(state, self._speed_to_float(speed))

    def _frames_dir(self):
        """Return ``<run_output_dir>/episodes/<episode>-<port>/``."""
        if self._run_output_dir:
            return os.path.join(self._run_output_dir, 'episodes',
                                '{}-{}'.format(self.global_episode, self.port))
        base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, 'episodes',
                            self._run_id or 'unnamed_run',
                            '{}-{}'.format(self.global_episode, self.port))

    def _ensure_save_dir(self):
        """Create the frame directory once per episode."""
        if self._save_dir_cached is None:
            self._save_dir_cached = self._frames_dir()
            os.makedirs(self._save_dir_cached, exist_ok=True)
        return self._save_dir_cached

    def _save_frame(self):
        """Write the current camera frame as JPEG. IO errors are swallowed."""
        if not self._save_images:
            return
        if not hasattr(self.env, 'state_observer'):
            return
        carla_img = getattr(self.env.state_observer, 'image', None)
        if carla_img is None:
            return
        try:
            ep_dir = self._ensure_save_dir()
            carla_img.save_to_disk(
                os.path.join(ep_dir, '{}.jpeg'.format(self.step_count)))
        except Exception:
            pass

    def _update_maneuver(self):
        """Advance the planned turn index when the vehicle leaves a junction."""
        try:
            if hasattr(self.env, 'planner') and hasattr(self.env, 'vehicle'):
                _, left_junction = self.env.planner.on_junction(
                    self.env.vehicle.get_location())
                if left_junction:
                    self._maneuver_idx += 1
                    if self._maneuver_idx < len(self._car_decisions):
                        self._current_maneuver = int(
                            self._car_decisions[self._maneuver_idx])
                    else:
                        self._current_maneuver = 1
        except Exception:
            pass

    def step(self, action):
        """Apply one action (with repeat) and return ``(obs, reward, done, info)``."""
        self.step_count += 1
        if 0 <= action < self.n_actions:
            self._action_counts[int(action)] += 1

        if hasattr(self.env, 'image_queue'):
            while not self.env.image_queue.empty():
                self.env.image_queue.get()
        for _ in range(self._action_repeat):
            self.env.step_apply_action(int(action))
            self.env.world.tick()

        (next_state, reward, done, route_distance,
         next_speed, distance_from_goal) = self.env.step(
            save_image=self._save_images,
            episode=self.global_episode,
            step=self.step_count,
        )

        speed_kmh = self._speed_to_float(next_speed)

        self._update_maneuver()
        self._save_frame()
        obs = self._observation(next_state, speed_kmh)

        collisions = len(self.env.collision_history_list) \
            if hasattr(self.env, 'collision_history_list') else 0
        lane_invasions = getattr(self.env, 'last_invasion_counter', 0)
        # The one episode length limit. CarlaEnv has none of its own. It is
        # applied before the reward, so the shaped goal bonus and the
        # reached_goal statistic see the same done flag.
        if self._episode_max_decisions > 0 and \
                self.step_count >= self._episode_max_decisions:
            done = True
        reward_f, reward_components = self._shape_reward(
            reward, done, route_distance, speed_kmh,
            distance_from_goal, collisions, lane_invasions)
        self._accumulate_reward_components(reward_components)
        if speed_kmh > self._episode_max_speed:
            self._episode_max_speed = speed_kmh
        if route_distance is not None and \
                route_distance < self._episode_min_route_dist:
            self._episode_min_route_dist = float(route_distance)
        self._episode_goal_dist = float(distance_from_goal) \
            if distance_from_goal is not None else self._episode_goal_dist
        if done and distance_from_goal is not None \
                and float(distance_from_goal) < GOAL_RADIUS_M:
            self._episode_reached_goal = True
        self._prev_goal_dist = float(distance_from_goal) \
            if distance_from_goal is not None else self._prev_goal_dist
        self._prev_collision_count = collisions

        info = {
            'route_distance': route_distance,
            'speed_kmh': speed_kmh,
            'distance_from_goal': distance_from_goal,
            'maneuver': self._current_maneuver,
            'collisions': collisions,
            'lane_invasions': lane_invasions,
            'reward_components': reward_components,
            'legacy_reward': float(reward),
        }
        return obs, reward_f, bool(done), info

    def episode_stats(self):
        """Statistics of the current episode for the episode logger.

        The keys are the keyword names of ``TrainingLogger.log_episode``.
        """
        return {
            'max_speed_kmh': self._episode_max_speed,
            'min_route_dist': self._episode_min_route_dist,
            'goal_dist': self._episode_goal_dist,
            'reached_goal': self._episode_reached_goal,
            'action_counts': self._action_counts.tolist(),
            'collisions': len(self.env.collision_history_list),
            'port': self.port,
            'reward_components': dict(self._episode_reward_components),
        }
