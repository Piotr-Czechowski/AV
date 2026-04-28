"""CARLA environment wrapper with retry + health-check, adapted for PyTorch.

Wraps the existing `CarlaEnv` so the training loop never touches
`env.world`, `env.client`, `env.planner`, or `env.image_queue` directly.

Exposes: reset() -> (state, speed, maneuver)
         step(action) -> (next_state, next_speed, next_maneuver,
                          reward, done, info)
         reconnect()
         is_server_alive()

The wrapper tracks per-episode stats (action counts, max speed, goal
distance, collisions, reached_goal) for the JSONL logger and W&B.
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
                 outdir=None):
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
        self._outdir = outdir

        self._save_episodes = set(save_episodes) if save_episodes else set()
        self._save_episode_interval = save_episode_interval
        self.global_episode = 0

        self.episode = 0
        self.step_count = 0
        self._ep_reward = 0.0
        self._ep_max_speed = 0.0
        self._ep_min_route_dist = float('inf')
        self._ep_goal_dist = float('inf')
        self._ep_reached_goal = False
        self._action_counts = np.zeros(n_actions, dtype=np.int64)

        self._maneuver_idx = 0
        self._current_maneuver = 1

        # Per-episode cache so we makedirs() once and skip the path
        # rebuild in the hot path of step().
        self._save_dir_cached = None
        self._save_failures = 0

        self.env = None
        self._connect_with_retries()

    def _connect_with_retries(self):
        for attempt in range(1, self.max_connect_retries + 1):
            try:
                self.env = CarlaEnv(
                    scenario=self._scenario, spawn_point=False,
                    terminal_point=False, mp_density=self._mp_density,
                    port=self.port, action_space=self._action_space,
                    camera=self._camera, resX=self._resX, resY=self._resY,
                    manual_control=False,
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
        try:
            if self.env is None or not hasattr(self.env, 'world') \
                    or self.env.world is None:
                return False
            self.env.world.get_snapshot()
            return True
        except Exception:
            return False

    def _should_save_this_episode(self):
        if self.global_episode in self._save_episodes:
            return True
        if self._save_episode_interval > 0 \
                and self.global_episode > 0 \
                and self.global_episode % self._save_episode_interval == 0:
            return True
        return False

    def reset(self):
        self.episode += 1
        self.step_count = 0
        self._ep_reward = 0.0
        self._ep_max_speed = 0.0
        self._ep_min_route_dist = float('inf')
        self._ep_goal_dist = float('inf')
        self._ep_reached_goal = False
        self._action_counts = np.zeros(self.n_actions, dtype=np.int64)

        save_images = self._should_save_this_episode()
        self._save_images = save_images
        # New episode → new save dir; force a re-makedirs on first save.
        self._save_dir_cached = None

        if hasattr(self.env, 'state_observer'):
            self.env.state_observer.reset()

        state, speed = self.env.reset(
            save_image=save_images, episode=self.global_episode)

        if isinstance(state, np.ndarray):
            state_np = state.astype(np.float32) / 255.0
        else:
            state_np = (state.float() / 255.0).cpu().numpy() \
                if hasattr(state, 'cpu') else np.asarray(state, dtype=np.float32) / 255.0
        # carla_env emits a batched 4D tensor [1, C, H, W]; the trainer
        # re-adds the batch dim, so drop the leading singleton here.
        if state_np.ndim == 4 and state_np.shape[0] == 1:
            state_np = state_np[0]
        speed_f = float(speed) / 100.0

        if hasattr(self.env, 'car_decisions') and self.env.car_decisions:
            self._car_decisions = list(self.env.car_decisions)
        else:
            self._car_decisions = [1]
        self._maneuver_idx = 0
        self._current_maneuver = int(self._car_decisions[0])

        return state_np, speed_f, self._current_maneuver

    def _frames_dir(self):
        """Where this run+episode+server's frames go.

        Layout (matches async-rl/episodes exactly):
            <project_dir>/episodes/<run_id>/<global_episode>-<port>/

        Sits above any per-run outdir, so frames from different runs
        share a single root and can be diffed/compared by run name.
        """
        base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, 'episodes',
                            self._run_id or 'unnamed_run',
                            '{}-{}'.format(self.global_episode, self.port))

    def _ensure_save_dir(self):
        """makedirs once per episode; cache to avoid repeated stat()."""
        if self._save_dir_cached is None:
            self._save_dir_cached = self._frames_dir()
            os.makedirs(self._save_dir_cached, exist_ok=True)
        return self._save_dir_cached

    def _save_frame(self):
        """Dump the latest CARLA camera image to disk as JPEG.

        Mirrors async-rl's behaviour: pulls the raw `carla.Image` cached
        on `state_observer.image` and calls `.save_to_disk(path)` on it.
        IO failures are counted (so we can announce silent saving issues
        in logs) but never raised — disk hiccups must not kill training.
        """
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
        self.step_count += 1
        if 0 <= action < self.n_actions:
            self._action_counts[int(action)] += 1

        # CarlaEnv.step() consumes two camera frames per call
        # (image_queue.get ×2), so we must produce two frames between
        # consecutive step() calls — one world.tick per frame. The
        # pre-tick drain clears stale frames from the previous step so
        # we read exactly the two frames produced by the ticks below.
        self.env.step_apply_action(int(action))
        if hasattr(self.env, 'image_queue'):
            while not self.env.image_queue.empty():
                self.env.image_queue.get()
        self.env.world.tick()
        self.env.step_apply_action(int(action))
        self.env.world.tick()

        (next_state, reward, done, _,
         next_speed, distance_from_target) = self.env.step(
            save_image=self._save_images,
            episode=self.global_episode,
            step=self.step_count,
        )

        if isinstance(next_state, np.ndarray):
            next_np = next_state.astype(np.float32) / 255.0
        else:
            next_np = (next_state.float() / 255.0).cpu().numpy() \
                if hasattr(next_state, 'cpu') \
                else np.asarray(next_state, dtype=np.float32) / 255.0
        if next_np.ndim == 4 and next_np.shape[0] == 1:
            next_np = next_np[0]

        try:
            next_speed_f = float(next_speed) / 100.0
        except Exception:
            next_speed_f = float(next_speed.cpu()) / 100.0

        self._update_maneuver()
        self._save_frame()

        reward_f = float(reward)
        self._ep_reward += reward_f
        speed_kmh = next_speed_f * 100.0
        if speed_kmh > self._ep_max_speed:
            self._ep_max_speed = speed_kmh
        if distance_from_target is not None and \
                distance_from_target < self._ep_min_route_dist:
            self._ep_min_route_dist = float(distance_from_target)
        self._ep_goal_dist = float(distance_from_target) \
            if distance_from_target is not None else self._ep_goal_dist
        if done and distance_from_target is not None \
                and float(distance_from_target) < 3.0:
            self._ep_reached_goal = True

        info = {
            'route_distance': distance_from_target,
            'speed_kmh': speed_kmh,
            'distance_from_goal': distance_from_target,
            'maneuver': self._current_maneuver,
            'collisions': len(self.env.collision_history_list)
                          if hasattr(self.env, 'collision_history_list') else 0,
        }
        return next_np, next_speed_f, self._current_maneuver, \
               reward_f, bool(done), info
