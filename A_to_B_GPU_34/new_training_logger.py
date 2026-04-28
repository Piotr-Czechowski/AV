"""JSONL logger for A3C training runs. Ported from async-rl.

Each worker writes its own log files, line-buffered so data survives crashes.

Layout:
    <outdir>/logs/
        metadata.json
        events.jsonl
        system.jsonl
        worker_0/
            episodes.jsonl
            updates.jsonl
            steps.jsonl
            timing.jsonl
            system.jsonl
        worker_1/ ...
"""

import json
import os
from datetime import datetime


def _json_default(obj):
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            if v != v:
                return None
            return v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except ImportError:
        pass
    if hasattr(obj, 'get'):
        return _json_default(obj.get())
    return str(obj)


class TrainingLogger:
    def __init__(self, outdir, worker_id, log_steps=False,
                 log_update_arrays=False):
        self.outdir = outdir
        self.worker_id = worker_id
        self.log_steps_enabled = log_steps
        self.log_update_arrays = log_update_arrays

        self.log_dir = os.path.join(outdir, 'logs',
                                    'worker_{}'.format(worker_id))
        os.makedirs(self.log_dir, exist_ok=True)

        self._episodes_f = self._open('episodes.jsonl')
        self._updates_f = self._open('updates.jsonl')
        self._timing_f = self._open('timing.jsonl')
        self._steps_f = self._open('steps.jsonl') if log_steps else None

        events_dir = os.path.join(outdir, 'logs')
        os.makedirs(events_dir, exist_ok=True)
        self._events_f = open(
            os.path.join(events_dir, 'events.jsonl'), 'a', buffering=1)

    def _open(self, name):
        return open(os.path.join(self.log_dir, name), 'a', buffering=1)

    @staticmethod
    def _ts():
        return datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]

    def _write(self, f, data):
        data['ts'] = self._ts()
        data['worker'] = self.worker_id
        f.write(json.dumps(data, default=_json_default) + '\n')

    def log_step(self, global_t, local_t, global_episode, step_in_ep,
                 action, value, entropy, reward, done,
                 speed_kmh=None, route_dist=None, goal_dist=None,
                 maneuver=None, **extra):
        if self._steps_f is None:
            return
        rec = {
            'global_t': global_t,
            'local_t': local_t,
            'global_episode': global_episode,
            'step_in_ep': step_in_ep,
            'action': int(action),
            'value': value,
            'entropy': entropy,
            'reward': reward,
            'done': done,
        }
        if speed_kmh is not None:
            rec['speed_kmh'] = speed_kmh
        if route_dist is not None:
            rec['route_dist'] = route_dist
        if goal_dist is not None:
            rec['goal_dist'] = goal_dist
        if maneuver is not None:
            rec['maneuver'] = maneuver
        rec.update(extra)
        self._write(self._steps_f, rec)

    def log_episode(self, global_episode, global_t, total_reward, steps,
                    duration_s=None, max_speed_kmh=None,
                    min_route_dist=None, goal_dist=None,
                    collisions=None, reached_goal=False,
                    action_counts=None, port=None, **extra):
        rec = {
            'global_episode': global_episode,
            'global_t': global_t,
            'total_reward': total_reward,
            'steps': steps,
            'mean_reward': total_reward / steps if steps > 0 else 0.0,
            'reached_goal': reached_goal,
        }
        if duration_s is not None:
            rec['duration_s'] = round(duration_s, 3)
        if max_speed_kmh is not None:
            rec['max_speed_kmh'] = max_speed_kmh
        if min_route_dist is not None:
            rec['min_route_dist'] = min_route_dist
        if goal_dist is not None:
            rec['goal_dist'] = goal_dist
        if collisions is not None:
            rec['collisions'] = collisions
        if action_counts is not None:
            rec['action_counts'] = action_counts
        if port is not None:
            rec['port'] = port
        rec.update(extra)
        self._write(self._episodes_f, rec)

    def log_update(self, update_count, global_t, traj_len, is_terminal,
                   pi_loss, v_loss, total_loss, grad_norm, lr,
                   advantages=None, values=None, rewards=None,
                   entropies=None, **extra):
        import numpy as np

        rec = {
            'update': update_count,
            'global_t': global_t,
            'traj_len': traj_len,
            'is_terminal': is_terminal,
            'pi_loss': pi_loss,
            'v_loss': v_loss,
            'total_loss': total_loss,
            'grad_norm': grad_norm,
            'lr': lr,
        }

        if advantages is not None:
            adv = np.asarray(advantages)
            rec['adv_mean'] = float(adv.mean()) if len(adv) else 0.0
            rec['adv_std'] = float(adv.std()) if len(adv) > 1 else 0.0
        if values is not None:
            val = np.asarray(values)
            rec['val_mean'] = float(val.mean()) if len(val) else 0.0
            rec['val_std'] = float(val.std()) if len(val) > 1 else 0.0
        if entropies is not None:
            ent = np.asarray(entropies)
            rec['ent_mean'] = float(ent.mean()) if len(ent) else 0.0
        if rewards is not None:
            rew = np.asarray(rewards)
            rec['rew_mean'] = float(rew.mean()) if len(rew) else 0.0
            rec['rew_sum'] = float(rew.sum()) if len(rew) else 0.0

        if self.log_update_arrays:
            if advantages is not None:
                rec['advantages'] = advantages
            if values is not None:
                rec['values'] = values
            if rewards is not None:
                rec['rewards'] = rewards
            if entropies is not None:
                rec['entropies'] = entropies

        rec.update(extra)
        self._write(self._updates_f, rec)

    def log_timing(self, timing_stats, window_updates=None):
        rec = {'window_updates': window_updates, 'ops': {}}
        for name, (avg_ms, count, total_s) in timing_stats.items():
            rec['ops'][name] = {
                'avg_ms': round(avg_ms, 4),
                'count': count,
                'total_s': round(total_s, 4),
            }
        self._write(self._timing_f, rec)

    def log_event(self, event_type, **kwargs):
        kwargs['event'] = event_type
        self._write(self._events_f, kwargs)

    def log_save(self, path, global_t, **kwargs):
        self.log_event('model_save', path=path, global_t=global_t, **kwargs)

    def log_load(self, path, global_t, **kwargs):
        self.log_event('model_load', path=path, global_t=global_t, **kwargs)

    def log_checkpoint(self, path, global_t, **kwargs):
        self.log_event('checkpoint_save', path=path, global_t=global_t,
                       **kwargs)

    def log_crash_recovery(self, global_t, error=None, **kwargs):
        self.log_event('crash_recovery', global_t=global_t,
                       error=str(error) if error else None, **kwargs)

    def log_nan(self, global_t, nan_count, nan_layers=None, **kwargs):
        self.log_event('nan_gradient', global_t=global_t,
                       nan_count=nan_count, nan_layers=nan_layers, **kwargs)

    def close(self):
        for f in [self._episodes_f, self._updates_f, self._timing_f,
                  self._steps_f, self._events_f]:
            if f is not None:
                try:
                    f.flush()
                    f.close()
                except Exception:
                    pass

    @staticmethod
    def write_metadata(outdir, args_dict, model_name, n_params,
                       n_workers, **extra):
        logs_dir = os.path.join(outdir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        meta = {
            'start_time': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            'args': args_dict,
            'model': model_name,
            'n_params': n_params,
            'n_workers': n_workers,
        }
        meta.update(extra)
        with open(os.path.join(logs_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2, default=_json_default)

    @staticmethod
    def read_max_episode(outdir, worker_id=None):
        import glob as globmod
        max_ep = 0
        logs_dir = os.path.join(outdir, 'logs')
        if worker_id is not None:
            paths = [os.path.join(logs_dir,
                                  'worker_{}'.format(worker_id),
                                  'episodes.jsonl')]
        else:
            paths = globmod.glob(
                os.path.join(logs_dir, 'worker_*', 'episodes.jsonl'))
        for ep_file in paths:
            if not os.path.exists(ep_file):
                continue
            try:
                with open(ep_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            ep = rec.get('global_episode',
                                         rec.get('episode', 0))
                            if ep > max_ep:
                                max_ep = ep
                        except json.JSONDecodeError:
                            pass
            except OSError:
                pass
        return max_ep
