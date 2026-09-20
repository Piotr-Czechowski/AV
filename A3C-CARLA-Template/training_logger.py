"""Local JSONL logging and W&B telemetry for A3C.

Two independent paths, on purpose:

1. Local files.  Every training process owns a ``TrainingLogger`` that
   writes its own ``logs/worker_<id>/*.jsonl``.  Nothing is shared, so
   nothing needs locking, and these files are the complete record of a
   run.
2. Telemetry.  The same record objects are also pushed onto one
   ``multiprocessing.Queue``.  A single consumer process
   (``telemetry_process_main``) drains it and is the only writer of the
   run-global ``logs/events.jsonl`` and the only caller of ``wandb.*``.

Records travel as plain dicts carrying a ``kind`` field, which is all the
consumer needs in order to route them.  Values are forwarded to W&B one
for one: the telemetry path never averages, downsamples, or otherwise
alters what training produced.

Telemetry is best effort.  A full queue costs W&B points, never local
records and never training throughput.

``ResourceLogger`` runs only in runs launched with ``--log-resources``:
one daemon thread per process samples its own CPU/RAM usage, and the
main process additionally samples every visible GPU, into ``system``
records that travel the same two paths.
"""

import json
import math
import os
import queue as queue_module
import threading
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None


RECORD_KINDS = frozenset(
    ('step', 'episode', 'update', 'timing', 'system', 'event'))
TELEMETRY_CONTROL_KEY = '_telemetry_control'

# How long a worker may wait for queue space when publishing an event.
# Metrics never wait at all.
_EVENT_ENQUEUE_TIMEOUT_S = 1.0


def _increment_counter(counter):
    """Add one to a shared ``mp.Value``; ignore an absent counter."""
    if counter is None:
        return
    with counter.get_lock():
        counter.value += 1


def _counter_value(counter):
    """Read a shared ``mp.Value``; an absent counter reads as zero."""
    return int(counter.value) if counter is not None else 0


def normalize_for_json(value):
    """Convert runtime values into values ``json.dumps`` accepts.

    Handles NumPy scalars/arrays, torch tensors, and nested
    dicts/lists/tuples/sets.  ``NaN`` and infinities become ``None``
    because strict JSON cannot represent them.  Anything else falls back
    to its ``str()``.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None

    try:
        import numpy as np
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return normalize_for_json(float(value))
        if isinstance(value, np.ndarray):
            return normalize_for_json(value.tolist())
    except ImportError:
        pass

    try:
        import torch
        if isinstance(value, torch.Tensor):
            return normalize_for_json(value.detach().cpu().tolist())
    except ImportError:
        pass

    if isinstance(value, dict):
        return {str(key): normalize_for_json(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [normalize_for_json(item) for item in value]
    return str(value)


def _timestamp():
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]


def build_record(kind, worker_id=None, data=None):
    """Wrap ``data`` in the envelope shared by local logs and telemetry.

    ``kind`` is how the telemetry consumer routes the record and must be
    one of ``RECORD_KINDS``.  ``worker_id`` is the source worker, or
    ``-1``/``None`` for run-level records.  Runs that resume an existing
    output directory are told apart by the ``training_start`` and
    ``training_end`` events in ``events.jsonl``.
    """
    if kind not in RECORD_KINDS:
        raise ValueError('unsupported record kind: {}'.format(kind))
    record = dict(data or {})
    record['kind'] = kind
    record['ts'] = record.get('ts') or _timestamp()
    record['worker'] = worker_id
    return normalize_for_json(record)


def enqueue_telemetry(telemetry_queue, record, is_event=False,
                      dropped_counter=None):
    """Hand one record to the telemetry process without blocking training.

    Metrics are dropped the moment the queue is full.  Events matter more,
    so they wait up to ``_EVENT_ENQUEUE_TIMEOUT_S`` for space before being
    dropped as well.  Every drop bumps ``dropped_counter``, which the final
    summary reports.  Returns whether the record was accepted.
    """
    if telemetry_queue is None:
        return False
    try:
        if is_event:
            telemetry_queue.put(
                record, timeout=_EVENT_ENQUEUE_TIMEOUT_S)
        else:
            telemetry_queue.put_nowait(record)
        return True
    except queue_module.Full:
        _increment_counter(dropped_counter)
        if is_event:
            print('[TELEMETRY] event dropped: queue full', flush=True)
        return False


def _put_metric(payload, name, value):
    """Add one W&B metric, skipping anything that is not a finite number."""
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


_UPDATE_METRICS = {
    'pi_loss': 'train/pi_loss',
    'v_loss': 'train/v_loss',
    'total_loss': 'train/total_loss',
    'gradient_norm': 'train/gradient_norm',
    'gradient_norm_pre_clip': 'train/gradient_norm_pre_clip',
    'grad_clipped': 'train/grad_clipped',
    'ent_mean': 'train/entropy',
    'advantages_mean': 'train/advantages_mean',
    'advantages_std': 'train/advantages_std',
    'val_mean': 'train/value_mean',
    'val_std': 'train/value_std',
    'rew_mean': 'train/reward_mean',
    'rew_sum': 'train/reward_sum',
    'trajectory_length': 'train/trajectory_length',
    'lr': 'train/lr',
    'entropy_coef': 'train/entropy_coef',
}

_SYSTEM_METRICS = ('proc_cpu_percent', 'proc_rss_gb')


def project_update_to_wandb(record):
    """Project one optimizer update onto W&B metrics, values untouched.

    Emits ``train/<metric>`` for the run as a whole plus
    ``worker_<id>/train/<metric>`` so a single misbehaving worker stays
    visible.  Reward components are forwarded under their own names.  Raw
    per-step arrays are skipped: ``_put_metric`` only accepts scalars.
    """
    payload = {}
    _put_metric(payload, 'global_step', record.get('global_t'))
    for source, target in _UPDATE_METRICS.items():
        _put_metric(payload, target, record.get(source))
    for name, value in record.items():
        if name.startswith('reward_') and name != 'reward_total_sum':
            _put_metric(payload, 'train/{}'.format(name), value)

    worker_id = record.get('worker')
    if worker_id is not None and int(worker_id) >= 0:
        prefix = 'worker_{}/'.format(worker_id)
        for name in [key for key in payload if key.startswith('train/')]:
            payload[prefix + name] = payload[name]
    return payload


def project_system_to_wandb(record):
    """Project one resource sample onto W&B metrics, values untouched.

    The main process reports its own CPU/RSS plus one sample per
    visible GPU under ``system/*``.  Workers, which own no GPU view,
    report their process numbers under ``worker_<id>/system/*``.
    """
    payload = {}
    _put_metric(payload, 'global_step', record.get('global_t'))
    worker_id = record.get('worker')
    prefix = ''
    if worker_id is not None and int(worker_id) >= 0:
        prefix = 'worker_{}/'.format(worker_id)
    for name in _SYSTEM_METRICS:
        _put_metric(
            payload, prefix + 'system/{}'.format(name), record.get(name))
    for gpu in record.get('gpus') or ():
        if not isinstance(gpu, dict) or \
                not isinstance(gpu.get('index'), int):
            continue
        prefix = 'system/gpu{}_'.format(gpu['index'])
        for name in ('util_percent', 'mem_used_gb'):
            _put_metric(payload, prefix + name, gpu.get(name))
    return payload


def make_telemetry_stop(final_summary=None):
    """Build the sentinel that tells the telemetry process to drain and exit."""
    return {
        TELEMETRY_CONTROL_KEY: 'stop',
        'final_summary': normalize_for_json(final_summary or {}),
    }


def configure_wandb_metrics(wandb_run, num_workers):
    """Pin W&B series to ``global_step`` for run-wide and per-worker metrics."""
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


_PROJECTIONS = {
    'episode': project_episode_to_wandb,
    'update': project_update_to_wandb,
    'system': project_system_to_wandb,
}


def _shared_counter(shared_counters, name):
    """Look one shared counter up; missing counters read as absent."""
    if not isinstance(shared_counters, dict):
        return None
    return shared_counters.get(name)


def run_telemetry_loop(telemetry_queue, run_output_dir, wandb_run=None,
                       shared_counters=None, startup_events=None):
    """Drain the telemetry queue until the stop sentinel arrives.

    This is the only writer of ``events.jsonl`` and the only place that
    calls ``wandb_run.log``.  Every record is forwarded as its own W&B
    point with the values training produced; nothing is batched or
    averaged.  W&B failures are counted and swallowed so that losing the
    network never costs a local event.

    ``wandb_run`` may be ``None``, in which case the loop still owns
    ``events.jsonl``.  ``startup_events`` are records created before the
    loop began and are written first to keep the file chronological.
    """
    logs_dir = os.path.join(run_output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    events_path = os.path.join(logs_dir, 'events.jsonl')
    health_counts = {}

    def log_wandb(payload):
        if wandb_run is None or not payload:
            return
        try:
            wandb_run.log(payload)
        except Exception as error:
            counter = _shared_counter(shared_counters, 'wandb_errors')
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
                final_summary = dict(record.get('final_summary') or {})
                final_summary.update(health_counts)
                if isinstance(shared_counters, dict):
                    for name, counter in shared_counters.items():
                        final_summary[name] = _counter_value(counter)
                write_event(build_record(
                    'event', worker_id=-1,
                    data=dict(final_summary, event='final_summary')))
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
                # Events go to disk first: a broken W&B must not cost one.
                write_event(record)
                metric = _HEALTH_METRICS.get(record.get('event'))
                if metric:
                    health_counts[metric] = health_counts.get(metric, 0) + 1
                    payload = {metric: health_counts[metric]}
                    _put_metric(payload, 'global_step',
                                record.get('global_t'))
                    log_wandb(payload)
            elif kind in _PROJECTIONS:
                log_wandb(_PROJECTIONS[kind](record))


def telemetry_process_main(telemetry_queue, run_output_dir,
                           wandb_enabled=False, wandb_config=None,
                           wandb_init_kwargs=None, shared_counters=None):
    """Entry point of the telemetry process: own W&B, then drain the queue.

    Kept separate from ``run_telemetry_loop`` so the loop itself can be
    exercised without W&B.  ``wandb`` is imported here and nowhere else,
    which keeps the SDK out of every training process.  A failed
    ``wandb.init`` degrades to local-only logging and is recorded as an
    event rather than raised.
    """
    wandb_run = None
    startup_events = []
    config = dict(wandb_config or {})

    def record_wandb_failure(event_type, error):
        startup_events.append(build_record(
            'event', worker_id=-1,
            data={'event': event_type, 'error': str(error)}))
        _increment_counter(
            _shared_counter(shared_counters, 'wandb_errors'))

    if wandb_enabled:
        try:
            import wandb
            wandb_run = wandb.init(
                config=config, **dict(wandb_init_kwargs or {}))
        except Exception as error:
            record_wandb_failure('wandb_init_failed', error)
            wandb_run = None
        if wandb_run is not None:
            try:
                configure_wandb_metrics(
                    wandb_run, config.get('num_workers', 0))
            except Exception as error:
                record_wandb_failure('wandb_metric_setup_failed', error)

    try:
        run_telemetry_loop(
            telemetry_queue, run_output_dir, wandb_run=wandb_run,
            shared_counters=shared_counters,
            startup_events=startup_events)
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception as error:
                _increment_counter(
                    _shared_counter(shared_counters, 'wandb_errors'))
                print('[TELEMETRY] wandb.finish failed: {}'.format(error),
                      flush=True)


class ResourceLogger:
    """Minimal resource sampler for ``--log-resources`` runs.

    One daemon thread per process samples the host process's own CPU
    usage and RSS.  The main process, the only one with a node-wide
    view, additionally samples every visible GPU through pynvml.
    Each sample becomes one ``system`` record handed to the
    ``TrainingLogger``, so it lands in the local ``resources.jsonl``
    and reaches W&B through the ordinary telemetry path.  Sampling
    errors are swallowed: monitoring must never disturb training.
    """

    def __init__(self, logger, interval=10.0, gpu_indices=None,
                 global_step_getter=None):
        self.logger = logger
        self.interval = interval
        self.gpu_indices = list(gpu_indices or ())
        self.global_step_getter = global_step_getter
        self._stop_event = threading.Event()
        self._thread = None
        self._nvml = None
        self._proc = None

    def start(self):
        """Start the sampler thread. No-op if psutil is missing."""
        if psutil is None:
            print('[RESOURCES] psutil unavailable; '
                  'resource logging disabled', flush=True)
            return
        self._proc = psutil.Process(os.getpid())
        self._proc.cpu_percent(interval=None)
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name='ResourceLogger')
        self._thread.start()

    def stop(self):
        """Join the sampler thread."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
        self._thread = None

    def _ensure_nvml(self):
        """Lazy-init pynvml; cache False if GPU sampling is unavailable."""
        if self._nvml is not None or not self.gpu_indices:
            return self._nvml
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
        except Exception:
            self._nvml = False
            print('[RESOURCES] pynvml unavailable; '
                  'GPU sampling disabled', flush=True)
        return self._nvml

    def _sample(self):
        """Return one CPU/RSS sample, plus GPU stats when NVML is available."""
        data = {
            'proc_cpu_percent': round(
                self._proc.cpu_percent(interval=None), 1),
            'proc_rss_gb': round(
                self._proc.memory_info().rss / 1e9, 3),
        }
        pynvml = self._ensure_nvml()
        if pynvml:
            gpus = []
            for index in self.gpu_indices:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpus.append({
                        'index': index,
                        'util_percent': int(util.gpu),
                        'mem_used_gb': round(mem.used / 1e9, 3),
                    })
                except Exception:
                    pass
            data['gpus'] = gpus
        return data

    def _loop(self):
        while not self._stop_event.is_set():
            try:
                data = self._sample()
                if self.global_step_getter is not None:
                    try:
                        data['global_t'] = self.global_step_getter()
                    except (AttributeError, TypeError, ValueError,
                            RuntimeError):
                        pass
                self.logger.log_resources(data)
            except Exception:
                pass
            if self._stop_event.wait(timeout=self.interval):
                break


class TrainingLogger:
    """Per-process writer of ``logs/worker_<id>/*.jsonl``.

    One instance lives in each worker plus one in the main process
    (``worker_id=-1``, events only).  Files belong to a single process, so
    writes need no locking, and they are opened lazily so that a logger
    which never writes leaves no empty directory behind.

    Records are written locally first and only then handed to telemetry,
    so local logs stay complete even when the queue is full.

    Args:
        log_steps: also write ``steps.jsonl``; very large, off by default.
        log_update_arrays: keep raw per-step arrays in update records.
        telemetry_queue: queue to the telemetry process, or ``None`` for
            local-only logging.
        dropped_counter: shared counter bumped when telemetry is dropped.
        publish_metrics: whether episode/update records are worth sending
            to telemetry.  Events are always sent regardless, because they
            are what ``events.jsonl`` is made of.
    """

    def __init__(self, run_output_dir, worker_id, log_steps=False,
                 log_update_arrays=False,
                 telemetry_queue=None, dropped_counter=None,
                 publish_metrics=False):
        self.run_output_dir = run_output_dir
        self.worker_id = worker_id
        self.log_steps_enabled = log_steps
        self.log_update_arrays = log_update_arrays
        self.telemetry_queue = telemetry_queue
        self.dropped_counter = dropped_counter
        self.publish_metrics = publish_metrics
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
            kind, worker_id=self.worker_id, data=data)

    def _write(self, name, record):
        self._open(name).write(
            json.dumps(record, allow_nan=False) + '\n')
        return record

    def _publish(self, record, is_event=False):
        """Send a record to telemetry, unless metrics are local-only."""
        if not is_event and not self.publish_metrics:
            return False
        return enqueue_telemetry(
            self.telemetry_queue, record, is_event=is_event,
            dropped_counter=self.dropped_counter)

    def log_step(self, global_t, local_t, global_episode, step_in_ep,
                 action, value, entropy, reward, done,
                 speed_kmh=None, route_dist=None, goal_dist=None,
                 maneuver=None, **extra):
        """Write one env step to ``steps.jsonl`` when ``--log-steps`` is on."""
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
        """Write one episode record locally and publish it to telemetry."""
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
        """Write one optimizer-update record and publish it to telemetry."""
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
        """Write one timing window from ``TimingAccumulator.get_stats()``."""
        data = {'window_updates': window_updates, 'ops': {}}
        for name, (avg_ms, count, total_s) in timing_stats.items():
            data['ops'][name] = {
                'avg_ms': round(avg_ms, 4),
                'count': count,
                'total_s': round(total_s, 4),
            }
        return self._write(
            'timing.jsonl', self._record('timing', data))

    def log_resources(self, data):
        """Write one resource sample, then hand it to telemetry."""
        record = self._write(
            'resources.jsonl', self._record('system', data))
        self._publish(record)
        return record

    def log_event(self, event_type, **kwargs):
        """Emit a lifecycle/health event.

        Events are not written by this logger: the telemetry process owns
        the single shared ``events.jsonl``, so they only travel by queue.
        """
        data = dict(kwargs)
        data['event'] = event_type
        record = self._record('event', data)
        self._publish(record, is_event=True)
        return record

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
        """Flush and close every open JSONL handle."""
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
        """Write ``logs/metadata.json`` once at training start."""
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
