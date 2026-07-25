"""Local JSONL logging and compact W&B telemetry for A3C.

Workers keep writing their own detailed files.  Episode and update records
may additionally be copied to one queue, whose consumer owns W&B and the
run-global ``events.jsonl`` file.
"""

import json
import math
import os
import queue as queue_module
import time
from datetime import datetime


RECORD_KINDS = frozenset(
    ('step', 'episode', 'update', 'timing', 'system', 'event'))
TELEMETRY_CONTROL_KEY = '_telemetry_control'


def _increment_counter(counter, amount=1):
    if counter is None:
        return
    try:
        lock = counter.get_lock()
    except (AttributeError, TypeError):
        lock = None
    if lock is None:
        counter.value += amount
    else:
        with lock:
            counter.value += amount


def _counter_value(counter):
    try:
        return int(counter.value)
    except (AttributeError, TypeError, ValueError):
        return 0


def normalize_for_json(value, non_finite_counter=None):
    """Recursively convert runtime values into strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        _increment_counter(non_finite_counter)
        return None

    try:
        import numpy as np
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return normalize_for_json(
                float(value), non_finite_counter)
        if isinstance(value, np.ndarray):
            return normalize_for_json(
                value.tolist(), non_finite_counter)
    except ImportError:
        pass

    try:
        import torch
        if isinstance(value, torch.Tensor):
            return normalize_for_json(
                value.detach().cpu().tolist(), non_finite_counter)
    except ImportError:
        pass

    if isinstance(value, dict):
        return {
            str(key): normalize_for_json(item, non_finite_counter)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [normalize_for_json(item, non_finite_counter)
                for item in value]
    if hasattr(value, 'value'):
        try:
            return normalize_for_json(value.value, non_finite_counter)
        except (AttributeError, TypeError, ValueError):
            pass
    if hasattr(value, 'get'):
        try:
            return normalize_for_json(value.get(), non_finite_counter)
        except (AttributeError, TypeError, ValueError):
            pass
    return str(value)


def _timestamp():
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]


def build_record(kind, session_id='legacy', worker_id=None, data=None,
                 timestamp=None, non_finite_counter=None):
    """Add the small routing envelope used by local logs and telemetry."""
    if kind not in RECORD_KINDS:
        raise ValueError('unsupported record kind: {}'.format(kind))
    record = dict(data or {})
    record['kind'] = kind
    record['session_id'] = str(session_id)
    record['ts'] = timestamp or record.get('ts') or _timestamp()
    record['worker'] = worker_id
    return normalize_for_json(record, non_finite_counter)


def enqueue_telemetry(telemetry_queue, record, required=False,
                      dropped_counter=None, timeout_s=1.0):
    """Publish without allowing optional telemetry to block training."""
    if telemetry_queue is None:
        return False
    try:
        if required:
            telemetry_queue.put(record, timeout=timeout_s)
        else:
            telemetry_queue.put_nowait(record)
        return True
    except queue_module.Full:
        _increment_counter(dropped_counter)
        if required:
            print('[TELEMETRY] required record dropped: queue full',
                  flush=True)
        return False


def _put_metric(payload, name, value):
    if isinstance(value, bool):
        payload[name] = int(value)
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        payload[name] = value


def project_episode_to_wandb(record):
    """Project one local episode record onto a small metric whitelist."""
    payload = {}
    _put_metric(payload, 'global_step', record.get('global_t'))
    mapping = {
        'global_episode': 'episode/id',
        'total_reward': 'episode/reward',
        'mean_reward': 'episode/reward_per_step',
        'steps': 'episode/length',
        'duration_s': 'episode/duration_s',
        'reached_goal': 'episode/success',
        'max_speed_kmh': 'episode/max_speed_kmh',
        'min_route_dist': 'episode/min_route_distance',
        'goal_dist': 'episode/goal_distance',
        'collisions': 'episode/collisions',
        'global_mean_reward': 'global/reward_mean',
        'best_reward': 'global/best_reward',
    }
    for source, target in mapping.items():
        _put_metric(payload, target, record.get(source))

    worker_id = record.get('worker')
    if worker_id is not None and int(worker_id) >= 0:
        prefix = 'worker_{}/'.format(worker_id)
        for source, suffix in (
                ('total_reward', 'reward'),
                ('steps', 'episode_length'),
                ('local_mean_reward', 'reward_mean_100')):
            _put_metric(payload, prefix + suffix, record.get(source))

    action_counts = record.get('action_counts')
    if isinstance(action_counts, list):
        total = sum(value for value in action_counts
                    if isinstance(value, (int, float)))
        if total > 0:
            for index, count in enumerate(action_counts):
                _put_metric(
                    payload,
                    'episode/action_{}_fraction'.format(index),
                    count / total)

    reward_components = record.get('reward_components')
    if isinstance(reward_components, dict):
        for name, value in reward_components.items():
            if name != 'total':
                _put_metric(
                    payload, 'episode/reward_{}'.format(name), value)
    return payload


def wandb_update_policy(config):
    """Return flags for config values that genuinely vary during a run."""
    config = config or {}
    steps = int(config.get('steps') or 0)
    lr = float(config.get('lr') or 0.0)
    beta_start = float(config.get(
        'beta_start', config.get('entropy_coef', 0.0)) or 0.0)
    beta_end = float(config.get('beta_end', beta_start) or 0.0)
    anneal_fraction = float(config.get('beta_anneal_frac') or 0.0)
    return {
        'include_learning_rate': steps > 0 and lr != 0.0,
        'include_entropy_coef': (
            steps > 0 and anneal_fraction > 0.0 and
            beta_start != beta_end),
    }


class TelemetryAggregator:
    """Aggregate update and node-system records into compact W&B points."""

    _UPDATE_FIELDS = {
        'pi_loss': 'train/pi_loss',
        'v_loss': 'train/v_loss',
        'total_loss': 'train/total_loss',
        'gradient_norm': 'train/gradient_norm',
        'gradient_norm_pre_clip': 'train/gradient_norm_pre_clip',
        'ent_mean': 'train/entropy',
        'advantages_mean': 'train/advantages_mean',
        'advantages_std': 'train/advantages_std',
        'val_mean': 'train/value_mean',
        'val_std': 'train/value_std',
        'rew_mean': 'train/reward_mean',
        'rew_sum': 'train/reward_sum',
    }

    def __init__(self, update_interval=100, system_interval_s=60.0,
                 include_learning_rate=True, include_entropy_coef=True):
        self.update_interval = max(1, int(update_interval))
        self.system_interval_s = max(0.0, float(system_interval_s))
        self.include_learning_rate = bool(include_learning_rate)
        self.include_entropy_coef = bool(include_entropy_coef)
        self._updates = []
        self._systems = []
        self._system_window_started = None

    @staticmethod
    def _values(records, field):
        values = []
        for record in records:
            value = record.get(field)
            if isinstance(value, bool):
                values.append(float(value))
            elif isinstance(value, (int, float)) and math.isfinite(value):
                values.append(float(value))
        return values

    @staticmethod
    def _nested_values(records, section, field):
        values = []
        for record in records:
            nested = record.get(section)
            if not isinstance(nested, dict):
                continue
            value = nested.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool) \
                    and math.isfinite(value):
                values.append(float(value))
        return values

    def _flush_updates(self):
        if not self._updates:
            return None
        records, self._updates = self._updates, []
        steps = self._values(records, 'global_t')
        payload = {'global_step': int(max(steps))} if steps else {}
        fields = dict(self._UPDATE_FIELDS)
        if self.include_learning_rate:
            fields['lr'] = 'train/lr'
        if self.include_entropy_coef:
            fields['entropy_coef'] = 'train/entropy_coef'
        for source, target in fields.items():
            values = self._values(records, source)
            if values:
                payload[target] = sum(values) / len(values)

        clipped = self._values(records, 'grad_clipped')
        if clipped:
            payload['train/grad_clip_fraction'] = \
                sum(clipped) / len(clipped)
        component_fields = sorted({
            key for record in records for key in record
            if key.startswith('reward_') and key.endswith('_sum')
            and key != 'reward_total_sum'
        })
        for field in component_fields:
            values = self._values(records, field)
            if values:
                payload['train/{}'.format(field)] = \
                    sum(values) / len(values)
        return payload or None

    def _flush_system(self):
        if not self._systems:
            return None
        records, self._systems = self._systems, []
        self._system_window_started = None
        steps = self._values(records, 'global_t')
        payload = {'global_step': int(max(steps))} if steps else {}

        fields = (
            ('cpu_percent_mean', 'system/cpu_mean_percent', 'mean'),
            ('cpu_percent_max', 'system/cpu_max_percent', 'max'),
            ('mem_percent', 'system/memory_percent', 'max'),
            ('mem_available_gb', 'system/memory_available_gb', 'min'),
            ('swap_used_gb', 'system/swap_used_gb', 'max'),
        )
        for source, target, operation in fields:
            values = self._nested_values(records, 'system', source)
            if not values:
                continue
            if operation == 'mean':
                payload[target] = sum(values) / len(values)
            elif operation == 'min':
                payload[target] = min(values)
            else:
                payload[target] = max(values)

        gpu_samples = {}
        for record in records:
            for gpu in record.get('gpus') or ():
                if not isinstance(gpu, dict) or \
                        not isinstance(gpu.get('index'), int):
                    continue
                samples = gpu_samples.setdefault(
                    gpu['index'], {'util': [], 'memory': []})
                for source, target in (
                        ('util_percent', 'util'),
                        ('mem_used_gb', 'memory')):
                    value = gpu.get(source)
                    if isinstance(value, (int, float)) and \
                            math.isfinite(value):
                        samples[target].append(float(value))
        for index, samples in sorted(gpu_samples.items()):
            prefix = 'system/gpu{}'.format(index)
            if samples['util']:
                payload[prefix + '_util_mean_percent'] = \
                    sum(samples['util']) / len(samples['util'])
                payload[prefix + '_util_max_percent'] = max(samples['util'])
            if samples['memory']:
                payload[prefix + '_memory_max_gb'] = max(samples['memory'])
        return payload or None

    def add(self, record):
        kind = record.get('kind')
        if kind == 'update':
            self._updates.append(record)
            if len(self._updates) >= self.update_interval:
                return [self._flush_updates()]
        elif kind == 'system':
            if self._system_window_started is None:
                self._system_window_started = time.monotonic()
            self._systems.append(record)
            if self.system_interval_s == 0 or time.monotonic() - \
                    self._system_window_started >= self.system_interval_s:
                return [self._flush_system()]
        return []

    def flush(self):
        return [payload for payload in (
            self._flush_updates(), self._flush_system())
            if payload is not None]


def make_telemetry_stop(final_summary=None):
    return {
        TELEMETRY_CONTROL_KEY: 'stop',
        'final_summary': normalize_for_json(final_summary or {}),
    }


def configure_wandb_metrics(wandb_run, num_workers):
    wandb_run.define_metric('global_step')
    for prefix in ('train', 'episode', 'global', 'system', 'health'):
        wandb_run.define_metric(
            '{}/*'.format(prefix), step_metric='global_step')
    for worker_id in range(int(num_workers)):
        wandb_run.define_metric(
            'worker_{}/*'.format(worker_id), step_metric='global_step')


_HEALTH_METRICS = {
    'nan_gradient': 'health/nan_updates_total',
    'camera_timeout': 'health/camera_timeouts_total',
    'worker_crash': 'health/worker_crashes_total',
    'worker_restart': 'health/worker_restarts_total',
    'worker_give_up': 'health/workers_given_up_total',
    'rollback': 'health/rollbacks_total',
}


def run_telemetry_loop(telemetry_queue, run_output_dir, wandb_run=None,
                       update_interval=100, system_interval_s=60.0,
                       include_learning_rate=True,
                       include_entropy_coef=True, shared_counters=None,
                       startup_events=None):
    """Drain the telemetry queue until sentinel and own events.jsonl."""
    logs_dir = os.path.join(run_output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    events_path = os.path.join(logs_dir, 'events.jsonl')
    aggregator = TelemetryAggregator(
        update_interval, system_interval_s,
        include_learning_rate, include_entropy_coef)
    health_counts = {}

    def log_wandb(payload):
        if wandb_run is None or not payload:
            return
        try:
            wandb_run.log(payload)
        except Exception as error:
            counter = shared_counters.get('wandb_errors') \
                if isinstance(shared_counters, dict) else None
            _increment_counter(counter)
            count = _counter_value(counter) or 1
            if count == 1 or count % 100 == 0:
                print('[TELEMETRY] wandb.log failed (count={}): {}'.format(
                    count, error), flush=True)

    with open(events_path, 'a', buffering=1) as events_handle:
        def write_event(record):
            events_handle.write(json.dumps(
                normalize_for_json(record), allow_nan=False) + '\n')

        for event in startup_events or ():
            write_event(event)

        while True:
            record = telemetry_queue.get()
            if isinstance(record, dict) and \
                    record.get(TELEMETRY_CONTROL_KEY) == 'stop':
                for payload in aggregator.flush():
                    log_wandb(payload)
                final_summary = dict(record.get('final_summary') or {})
                final_summary.update(health_counts)
                if isinstance(shared_counters, dict):
                    for name, counter in shared_counters.items():
                        final_summary[name] = _counter_value(counter)
                if wandb_run is not None:
                    try:
                        wandb_run.summary.update(
                            normalize_for_json(final_summary))
                    except Exception as error:
                        print('[TELEMETRY] W&B summary failed: {}'.format(
                            error), flush=True)
                break
            if not isinstance(record, dict):
                continue

            kind = record.get('kind')
            if kind == 'event':
                write_event(record)
                metric = _HEALTH_METRICS.get(record.get('event'))
                if metric:
                    health_counts[metric] = health_counts.get(metric, 0) + 1
                    payload = {metric: health_counts[metric]}
                    _put_metric(payload, 'global_step',
                                record.get('global_t'))
                    log_wandb(payload)
            elif kind == 'episode':
                log_wandb(project_episode_to_wandb(record))
            else:
                for payload in aggregator.add(record):
                    log_wandb(payload)


def telemetry_process_main(telemetry_queue, run_output_dir,
                           wandb_enabled=False, wandb_config=None,
                           wandb_init_kwargs=None, update_interval=100,
                           system_interval_s=60.0, shared_counters=None,
                           session_id='legacy'):
    """Initialize W&B in its sole owner process and drain telemetry."""
    wandb_run = None
    startup_events = []
    config = dict(wandb_config or {})
    if wandb_enabled:
        try:
            os.environ.setdefault('WANDB_INSECURE_DISABLE_SSL', 'true')
            import wandb
            wandb_run = wandb.init(
                config=config, **dict(wandb_init_kwargs or {}))
        except Exception as error:
            startup_events.append(build_record(
                'event', session_id=session_id, worker_id=-1,
                data={'event': 'wandb_init_failed', 'error': str(error)}))
            counter = shared_counters.get('wandb_errors') \
                if isinstance(shared_counters, dict) else None
            _increment_counter(counter)
            wandb_run = None
        if wandb_run is not None:
            try:
                configure_wandb_metrics(
                    wandb_run, config.get('num_workers', 0))
            except Exception as error:
                startup_events.append(build_record(
                    'event', session_id=session_id, worker_id=-1,
                    data={
                        'event': 'wandb_metric_setup_failed',
                        'error': str(error),
                    }))
                counter = shared_counters.get('wandb_errors') \
                    if isinstance(shared_counters, dict) else None
                _increment_counter(counter)

    policy = wandb_update_policy(config)
    try:
        run_telemetry_loop(
            telemetry_queue, run_output_dir, wandb_run=wandb_run,
            update_interval=update_interval,
            system_interval_s=system_interval_s,
            shared_counters=shared_counters,
            startup_events=startup_events, **policy)
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception as error:
                counter = shared_counters.get('wandb_errors') \
                    if isinstance(shared_counters, dict) else None
                _increment_counter(counter)
                print('[TELEMETRY] wandb.finish failed: {}'.format(error),
                      flush=True)


class TrainingLogger:
    def __init__(self, run_output_dir, worker_id, log_steps=False,
                 log_update_arrays=False, session_id='legacy',
                 telemetry_queue=None, dropped_counter=None,
                 non_finite_counter=None, remote_enabled=False):
        self.run_output_dir = run_output_dir
        self.worker_id = worker_id
        self.session_id = session_id
        self.log_steps_enabled = log_steps
        self.log_update_arrays = log_update_arrays
        self.telemetry_queue = telemetry_queue
        self.dropped_counter = dropped_counter
        self.non_finite_counter = non_finite_counter
        self.remote_enabled = remote_enabled
        self.log_dir = os.path.join(
            run_output_dir, 'logs', 'worker_{}'.format(worker_id))
        self._files = {}

    def _open(self, name):
        handle = self._files.get(name)
        if handle is None:
            os.makedirs(self.log_dir, exist_ok=True)
            handle = open(
                os.path.join(self.log_dir, name), 'a', buffering=1)
            self._files[name] = handle
        return handle

    def _record(self, kind, data):
        return build_record(
            kind, session_id=self.session_id, worker_id=self.worker_id,
            data=data, non_finite_counter=self.non_finite_counter)

    def _write(self, name, record):
        self._open(name).write(
            json.dumps(record, allow_nan=False) + '\n')
        return record

    def _publish(self, record, required=False):
        if not required and not self.remote_enabled:
            return False
        return enqueue_telemetry(
            self.telemetry_queue, record, required=required,
            dropped_counter=self.dropped_counter)

    def log_step(self, global_t, local_t, global_episode, step_in_ep,
                 action, value, entropy, reward, done,
                 speed_kmh=None, route_dist=None, goal_dist=None,
                 maneuver=None, **extra):
        if not self.log_steps_enabled:
            return None
        data = {
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
        for name, field_value in (
                ('speed_kmh', speed_kmh), ('route_dist', route_dist),
                ('goal_dist', goal_dist), ('maneuver', maneuver)):
            if field_value is not None:
                data[name] = field_value
        data.update(extra)
        return self._write('steps.jsonl', self._record('step', data))

    def log_episode(self, global_episode, global_t, total_reward, steps,
                    duration_s=None, max_speed_kmh=None,
                    min_route_dist=None, goal_dist=None,
                    collisions=None, reached_goal=False,
                    action_counts=None, port=None, **extra):
        data = {
            'global_episode': global_episode,
            'global_t': global_t,
            'total_reward': total_reward,
            'steps': steps,
            'mean_reward': total_reward / steps if steps > 0 else 0.0,
            'reached_goal': reached_goal,
        }
        optional = {
            'duration_s': round(duration_s, 3)
            if duration_s is not None else None,
            'max_speed_kmh': max_speed_kmh,
            'min_route_dist': min_route_dist,
            'goal_dist': goal_dist,
            'collisions': collisions,
            'action_counts': action_counts,
            'port': port,
        }
        data.update({key: value for key, value in optional.items()
                     if value is not None})
        data.update(extra)
        record = self._write(
            'episodes.jsonl', self._record('episode', data))
        self._publish(record)
        return record

    def log_update(self, update_count, global_t, trajectory_length,
                   is_terminal, pi_loss, v_loss, total_loss, gradient_norm,
                   lr, advantages=None, values=None, rewards=None,
                   entropies=None, **extra):
        import numpy as np

        data = {
            'update': update_count,
            'global_t': global_t,
            'trajectory_length': trajectory_length,
            'is_terminal': is_terminal,
            'pi_loss': pi_loss,
            'v_loss': v_loss,
            'total_loss': total_loss,
            'gradient_norm': gradient_norm,
            'lr': lr,
        }
        arrays = {
            'advantages': advantages,
            'values': values,
            'rewards': rewards,
            'entropies': entropies,
        }
        for name, array in arrays.items():
            if array is None:
                continue
            values_array = np.asarray(array)
            if name == 'advantages':
                data['advantages_mean'] = float(values_array.mean()) \
                    if len(values_array) else 0.0
                data['advantages_std'] = float(values_array.std()) \
                    if len(values_array) > 1 else 0.0
            elif name == 'values':
                data['val_mean'] = float(values_array.mean()) \
                    if len(values_array) else 0.0
                data['val_std'] = float(values_array.std()) \
                    if len(values_array) > 1 else 0.0
            elif name == 'rewards':
                data['rew_mean'] = float(values_array.mean()) \
                    if len(values_array) else 0.0
                data['rew_sum'] = float(values_array.sum()) \
                    if len(values_array) else 0.0
            else:
                data['ent_mean'] = float(values_array.mean()) \
                    if len(values_array) else 0.0
            if self.log_update_arrays:
                data[name] = array
        data.update(extra)
        record = self._write(
            'updates.jsonl', self._record('update', data))
        self._publish(record)
        return record

    def log_timing(self, timing_stats, window_updates=None):
        data = {'window_updates': window_updates, 'ops': {}}
        for name, (avg_ms, count, total_s) in timing_stats.items():
            data['ops'][name] = {
                'avg_ms': round(avg_ms, 4),
                'count': count,
                'total_s': round(total_s, 4),
            }
        return self._write(
            'timing.jsonl', self._record('timing', data))

    def log_event(self, event_type, **kwargs):
        data = dict(kwargs)
        data['event'] = event_type
        record = self._record('event', data)
        self._publish(record, required=True)
        return record

    def log_save(self, path, global_t, **kwargs):
        return self.log_event(
            'model_save', path=path, global_t=global_t, **kwargs)

    def log_load(self, path, global_t, **kwargs):
        return self.log_event(
            'model_load', path=path, global_t=global_t, **kwargs)

    def log_checkpoint(self, path, global_t, **kwargs):
        return self.log_event(
            'checkpoint_save', path=path, global_t=global_t, **kwargs)

    def log_crash_recovery(self, global_t, error=None, **kwargs):
        return self.log_event(
            'crash_recovery', global_t=global_t,
            error=str(error) if error else None, **kwargs)

    def log_nan(self, global_t, nan_count, nan_layers=None, **kwargs):
        return self.log_event(
            'nan_gradient', global_t=global_t,
            nan_count=nan_count, nan_layers=nan_layers, **kwargs)

    def close(self):
        for handle in self._files.values():
            try:
                handle.flush()
                handle.close()
            except (OSError, ValueError):
                pass
        self._files.clear()

    @staticmethod
    def write_metadata(run_output_dir, args_dict, model_name, n_params,
                       n_workers, **extra):
        logs_dir = os.path.join(run_output_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        metadata = {
            'start_time': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            'args': args_dict,
            'model': model_name,
            'n_params': n_params,
            'n_workers': n_workers,
        }
        metadata.update(extra)
        with open(os.path.join(logs_dir, 'metadata.json'), 'w') as handle:
            json.dump(
                normalize_for_json(metadata), handle,
                indent=2, allow_nan=False)

    @staticmethod
    def read_max_episode(run_output_dir, worker_id=None):
        import glob as globmod
        max_episode = 0
        logs_dir = os.path.join(run_output_dir, 'logs')
        if worker_id is not None:
            paths = [os.path.join(
                logs_dir, 'worker_{}'.format(worker_id), 'episodes.jsonl')]
        else:
            paths = globmod.glob(os.path.join(
                logs_dir, 'worker_*', 'episodes.jsonl'))
        for path in paths:
            if not os.path.exists(path):
                continue
            try:
                with open(path) as handle:
                    for line in handle:
                        try:
                            record = json.loads(line)
                        except (TypeError, ValueError):
                            continue
                        episode = record.get(
                            'global_episode', record.get('episode', 0))
                        if episode > max_episode:
                            max_episode = episode
            except OSError:
                pass
        return max_episode
