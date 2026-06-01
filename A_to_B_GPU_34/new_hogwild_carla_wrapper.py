"""Adapter between CarlaEnv and the A3C worker.

CarlaEnv owns the low-level CARLA objects. This wrapper keeps those details out
of the worker loop and exposes only:

    reset() -> (state, speed, maneuver)
    step(action) -> (next_state, next_speed, next_maneuver, reward, done, info)
    reconnect()
    is_server_alive()

It also handles observation normalization, action repeat, reward shaping,
optional frame saving, and per-episode statistics for the JSONL logger.
"""

import os
import time
import numpy as np

from carla_env import CarlaEnv


class CarlaA3CWrapper:
    def __init__(self, port, scenario, camera='semantic', resX=250, resY=250,
                 action_space='discrete', mp_density=25,
                 max_connect_retries=5, connect_retry_wait=30,
                 reconnect_wait=60, save_episodes=None,
                 save_episode_interval=0, run_id='', n_actions=10,
                 run_output_dir=None, action_repeat=2,
                 episode_max_decisions=100,
                 world_reload_interval=0, reward_mode='shaped',
                 reward_progress_coef=1.0, reward_target_speed_coef=1.0,
                 reward_route_penalty_coef=0.1, reward_time_penalty=0.01,
                 reward_goal_bonus=50.0, reward_collision_penalty=50.0,
                 reward_offroute_penalty=25.0,
                 reward_lane_invasion_penalty=5.0,
                 reward_target_speed_kmh=20.0,
                 reward_offroute_threshold=10.0, reward_clip=50.0,
                 verbose_env_logs=False):
        """Initialise all reward coefficients, episode counters, and the underlying CarlaEnv connection."""
        self.port = port
        self._scenario = scenario
        self._camera = camera
        self._resX = resX
        self._resY = resY
        self._action_space = action_space
        self._mp_density = mp_density
        self.max_connect_retries = max_connect_retries
        self.connect_retry_wait = connect_retry_wait
        self.reconnect_wait = reconnect_wait
        self.n_actions = n_actions
        self._run_id = run_id
        self._run_output_dir = run_output_dir
        self._action_repeat = max(1, int(action_repeat))
        self._episode_max_decisions = int(episode_max_decisions)
        self._world_reload_interval = int(world_reload_interval)
        self._reward_mode = reward_mode
        self._reward_progress_coef = float(reward_progress_coef)
        self._reward_target_speed_coef = float(reward_target_speed_coef)
        self._reward_route_penalty_coef = float(reward_route_penalty_coef)
        self._reward_time_penalty = float(reward_time_penalty)
        self._reward_goal_bonus = float(reward_goal_bonus)
        self._reward_collision_penalty = float(reward_collision_penalty)
        self._reward_offroute_penalty = float(reward_offroute_penalty)
        self._reward_lane_invasion_penalty = float(
            reward_lane_invasion_penalty)
        self._reward_target_speed_kmh = float(reward_target_speed_kmh)
        self._reward_offroute_threshold = float(reward_offroute_threshold)
        self._reward_clip = float(reward_clip)
        self._verbose_env_logs = bool(verbose_env_logs)

        self._save_episodes = set(save_episodes) if save_episodes else set()
        self._save_episode_interval = save_episode_interval
        self.global_episode = 0

        self.episode = 0
        self.step_count = 0
        self._episode_total_reward = 0.0
        self._episode_max_speed = 0.0
        self._episode_min_route_dist = float('inf')
        self._episode_goal_dist = float('inf')
        self._episode_reached_goal = False
        self._episode_reward_components = {}
        self._action_counts = np.zeros(n_actions, dtype=np.int64)
        self._prev_goal_dist = None
        self._prev_collision_count = 0

        self._maneuver_idx = 0
        self._current_maneuver = 1

        # Per-episode cache so frame saving creates the directory once and
        # does not rebuild the path in every step().
        self._save_dir_cached = None
        self._save_failures = 0

        self.env = None
        self._connect_with_retries()

    def _connect_with_retries(self):
        """Attempt to create a CarlaEnv up to max_connect_retries times, sleeping between failures."""
        for attempt in range(1, self.max_connect_retries + 1):
            try:
                self.env = CarlaEnv(
                    scenario=self._scenario, spawn_point=False,
                    terminal_point=False, mp_density=self._mp_density,
                    port=self.port, action_space=self._action_space,
                    camera=self._camera, resX=self._resX, resY=self._resY,
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
        """Tear down the stale CarlaEnv, wait for the server to restart, then reconnect with retries."""
        try:
            if self.env is not None:
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

    def is_server_alive(self):
        """Return True iff the CARLA world object exists and responds to a snapshot request without error."""
        try:
            if self.env is None or not hasattr(self.env, 'world') \
                    or self.env.world is None:
                return False
            self.env.world.get_snapshot()
            return True
        except Exception:
            return False

    def _state_to_chw_float(self, state):
        """Convert any raw observation (HWC uint8, CHW float, or tensor) to a validated [3,H,W] float32 array in [0,1]."""
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
        if arr.ndim == 3 and arr.shape[-1] == 3 and arr.shape[0] != 3:
            arr = np.transpose(arr, (2, 0, 1))
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[0] != 3:
            raise ValueError('expected observation [3,H,W], got {}'.format(
                arr.shape))
        if arr.shape[1] != self._resY or arr.shape[2] != self._resX:
            raise ValueError('expected observation [3,{},{}], got {}'.format(
                self._resY, self._resX, arr.shape))
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
        """Unwrap a speed value from any tensor or array type to a plain Python float."""
        if hasattr(speed, 'detach'):
            speed = speed.detach()
        if hasattr(speed, 'cpu'):
            speed = speed.cpu()
        if hasattr(speed, 'item'):
            speed = speed.item()
        return float(speed)

    def _current_goal_distance(self):
        """Query CarlaEnv for the Euclidean distance to the goal; returns None on any error so callers can handle gracefully."""
        try:
            distance, _ = self.env.calculate_distance()
            return float(distance)
        except Exception:
            return None

    def _accumulate_reward_components(self, components):
        """Add each named reward component to the running per-episode totals used by the JSONL logger."""
        for key, value in components.items():
            self._episode_reward_components[key] = \
                self._episode_reward_components.get(key, 0.0) + float(value)

    def _shape_reward(self, legacy_reward, done, route_distance,
                      speed_kmh, distance_from_goal, collisions,
                      lane_invasions):
        """Compute the shaped scalar reward and its named components for logging; legacy mode passes CarlaEnv's reward through unchanged."""
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
                            distance_from_goal < 3.0)

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
        """Return True if this episode's frames should be saved, based on the explicit set or periodic interval."""
        if self.global_episode in self._save_episodes:
            return True
        if self._save_episode_interval > 0 \
                and self.global_episode > 0 \
                and self.global_episode % self._save_episode_interval == 0:
            return True
        return False

    def reset(self):
        """Reset all per-episode statistics and CarlaEnv state; return (state_chw, speed_normalised, maneuver_id)."""
        self.episode += 1
        self.step_count = 0
        self._episode_total_reward = 0.0
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
            self.env.state_observer.reset()

        full_reload = self._world_reload_interval > 0 and \
            self.episode % self._world_reload_interval == 0
        state, speed = self.env.reset(
            save_image=save_images, episode=self.global_episode,
            reload_world=full_reload)

        state_np = self._state_to_chw_float(state)
        speed_f = self._speed_to_float(speed) / 100.0
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

        return state_np, speed_f, self._current_maneuver

    def _frames_dir(self):
        """Return the canonical on-disk path for this episode's saved frames, rooted at <script_dir>/episodes/<run_id>/<episode>-<port>/."""
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
        """Write the current camera frame to disk as JPEG, silently counting failures so IO errors never interrupt training."""
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
            self._save_failures += 1

    def _update_maneuver(self):
        """Advance the maneuver index when the vehicle exits a junction, cycling through the planned decision sequence."""
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
        """Apply one learner action and return the normalized next transition."""
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

        next_np = self._state_to_chw_float(next_state)
        next_speed_kmh = self._speed_to_float(next_speed)
        next_speed_f = next_speed_kmh / 100.0

        self._update_maneuver()
        self._save_frame()

        collisions = len(self.env.collision_history_list) \
            if hasattr(self.env, 'collision_history_list') else 0
        lane_invasions = getattr(self.env, 'last_invasion_counter', 0)
        reward_f, reward_components = self._shape_reward(
            reward, done, route_distance, next_speed_kmh,
            distance_from_goal, collisions, lane_invasions)
        if self._episode_max_decisions > 0 and \
                self.step_count >= self._episode_max_decisions:
            done = True
        self._accumulate_reward_components(reward_components)
        self._episode_total_reward += reward_f
        speed_kmh = next_speed_f * 100.0
        if speed_kmh > self._episode_max_speed:
            self._episode_max_speed = speed_kmh
        if route_distance is not None and \
                route_distance < self._episode_min_route_dist:
            self._episode_min_route_dist = float(route_distance)
        self._episode_goal_dist = float(distance_from_goal) \
            if distance_from_goal is not None else self._episode_goal_dist
        if done and distance_from_goal is not None \
                and float(distance_from_goal) < 3.0:
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
        return next_np, next_speed_f, self._current_maneuver, \
               reward_f, bool(done), info
