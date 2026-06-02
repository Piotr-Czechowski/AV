"""Worker supervisor and rollback logic.

Restarts crashed workers, caps per-worker restarts, detects rapid-crash
bursts, and rolls the global network back to the last non-NaN checkpoint
when that happens.
"""

import glob
import json
import os
import time
from datetime import datetime

import torch

from new_hogwild_a3c import A3CWorker, has_nan_params


def _ts():
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]


def find_latest_checkpoint(run_output_dir):
    """Return (path, step) for the most recent checkpoint, or (None, 0)."""
    candidates = []
    fixed = os.path.join(run_output_dir, 'checkpoint.pth')
    if os.path.exists(fixed):
        step = _read_step_file(
            os.path.join(run_output_dir, 'checkpoint_step.txt'))
        candidates.append((step, fixed))

    checkpoint_dir = os.path.join(run_output_dir, 'checkpoints')
    if os.path.isdir(checkpoint_dir):
        for entry in os.listdir(checkpoint_dir):
            worker_checkpoint_path = os.path.join(
                checkpoint_dir, entry, 'checkpoint.pth')
            if os.path.exists(worker_checkpoint_path):
                step = _read_step_file(
                    os.path.join(checkpoint_dir, entry,
                                 'checkpoint_step.txt'))
                candidates.append((step, worker_checkpoint_path))

    if not candidates:
        return None, 0
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], candidates[0][0]


def _read_step_file(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return 0


def rollback_global_network(global_network, run_output_dir, worker_idx=None):
    """Load the last non-NaN checkpoint back into the global network.

    Tries the crashing worker's own checkpoint first (if provided), then
    the top-level checkpoint, then every other per-worker checkpoint from
    newest to oldest. Returns True on success.
    """
    candidates = []
    if worker_idx is not None:
        worker_checkpoint_path = os.path.join(
            run_output_dir, 'checkpoints',
            'worker_{}'.format(worker_idx), 'checkpoint.pth')
        if os.path.exists(worker_checkpoint_path):
            step = _read_step_file(os.path.join(
                run_output_dir, 'checkpoints',
                'worker_{}'.format(worker_idx), 'checkpoint_step.txt'))
            candidates.append((step, worker_checkpoint_path))

    fixed = os.path.join(run_output_dir, 'checkpoint.pth')
    if os.path.exists(fixed):
        step = _read_step_file(
            os.path.join(run_output_dir, 'checkpoint_step.txt'))
        candidates.append((step, fixed))

    checkpoint_dir = os.path.join(run_output_dir, 'checkpoints')
    if os.path.isdir(checkpoint_dir):
        for entry in os.listdir(checkpoint_dir):
            if worker_idx is not None and entry == \
                    'worker_{}'.format(worker_idx):
                continue
            worker_checkpoint_path = os.path.join(
                checkpoint_dir, entry, 'checkpoint.pth')
            if os.path.exists(worker_checkpoint_path):
                step = _read_step_file(
                    os.path.join(checkpoint_dir, entry,
                                 'checkpoint_step.txt'))
                candidates.append((step, worker_checkpoint_path))

    candidates.sort(key=lambda x: x[0], reverse=True)
    for step, path in candidates:
        try:
            state = torch.load(path, map_location=global_network.device)
        except Exception as e:
            print('[ROLLBACK] {} unreadable: {}'.format(path, e), flush=True)
            continue
        with global_network.save_lock:
            try:
                if 'model' not in state:
                    print('[ROLLBACK] {} is missing the model key '
                          '(legacy checkpoint?); skipping'.format(path),
                          flush=True)
                    continue
                global_network.model.load_state_dict(state['model'])
            except Exception as e:
                print('[ROLLBACK] load_state_dict failed for {}: {}'.format(
                    path, e), flush=True)
                continue
            if has_nan_params(global_network.model):
                print('[ROLLBACK] {} has NaN params, skipping'.format(path),
                      flush=True)
                continue
            try:
                if 'optimizer' in state:
                    global_network.optimizer.load_state_dict(
                        state['optimizer'])
                    if hasattr(global_network.optimizer, 'share_memory'):
                        global_network.optimizer.share_memory()
            except Exception as e:
                print('[ROLLBACK] optimizer restore failed: {}'.format(e),
                      flush=True)
        print('[ROLLBACK] global network restored from {} (step {})'.format(
            path, step), flush=True)
        return True

    print('[ROLLBACK] no usable checkpoint found', flush=True)
    return False


def _start_worker(worker_id, global_network, config, port, device,
                  run_output_dir, shutdown_event, log_queue, run_id):
    w = A3CWorker(
        worker_id=worker_id,
        global_network=global_network,
        config=config,
        port=port,
        device=device,
        run_output_dir=run_output_dir,
        shutdown_event=shutdown_event,
        log_queue=log_queue,
        run_id=run_id,
    )
    w.start()
    return w


def _append_event(run_output_dir, **kwargs):
    try:
        path = os.path.join(run_output_dir, 'logs', 'events.jsonl')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        kwargs['ts'] = _ts()
        kwargs['worker'] = kwargs.get('worker', -1)
        with open(path, 'a') as f:
            f.write(json.dumps(kwargs) + '\n')
    except Exception:
        pass


def run_with_restart(global_network, config, run_output_dir, shutdown_event,
                     log_queue=None, run_id=''):
    """Start N workers and keep the run alive.

    Restart dead workers, cap per-worker restart counts, and on
    rapid-crash detection roll the global network back to the last
    non-NaN checkpoint.
    """
    workers = {}
    restart_counts = {i: 0 for i in range(config.num_workers)}
    last_crash_step = {i: None for i in range(config.num_workers)}
    rapid_crash_count = {i: 0 for i in range(config.num_workers)}

    for i in range(config.num_workers):
        port = config.start_port + config.port_step * i
        device = config.worker_gpus[i]
        workers[i] = _start_worker(
            i, global_network, config, port, device, run_output_dir,
            shutdown_event, log_queue, run_id)
        _append_event(run_output_dir, event='worker_start', worker=i,
                      port=port, device=device)

    try:
        while not shutdown_event.is_set():
            if config.steps > 0 and \
                    global_network.global_step.value >= config.steps:
                shutdown_event.set()
                break
            time.sleep(config.worker_check_interval)

            for i in range(config.num_workers):
                w = workers[i]
                if w.is_alive():
                    continue

                w.join(timeout=2)
                if shutdown_event.is_set():
                    break

                restart_counts[i] += 1
                current_step = global_network.global_step.value
                if last_crash_step[i] is not None and \
                        current_step - last_crash_step[i] \
                        < config.rapid_crash_window_steps:
                    rapid_crash_count[i] += 1
                else:
                    rapid_crash_count[i] = 0
                last_crash_step[i] = current_step

                print('[RESTART] W{} died (restart #{}) at step {}'.format(
                    i, restart_counts[i], current_step), flush=True)
                _append_event(run_output_dir, event='worker_restart',
                              worker=i,
                              restart_count=restart_counts[i],
                              global_t=current_step)

                # Remove local core dumps so repeated CARLA crashes do not
                # fill the job directory.
                for core_file in glob.glob('core.*'):
                    try:
                        os.remove(core_file)
                    except Exception:
                        pass

                if restart_counts[i] >= config.max_restarts_per_worker:
                    print('[RESTART] W{} exceeded max restarts ({}), '
                          'giving up'.format(
                              i, config.max_restarts_per_worker),
                          flush=True)
                    _append_event(run_output_dir, event='worker_give_up',
                                  worker=i,
                                  restart_count=restart_counts[i])
                    continue

                if rapid_crash_count[i] >= config.rapid_crash_threshold:
                    print('[ROLLBACK_TRIGGER] W{} crashed {} times rapidly'
                          .format(i, rapid_crash_count[i]), flush=True)
                    ok = rollback_global_network(
                        global_network, run_output_dir, worker_idx=i)
                    _append_event(run_output_dir, event='rollback',
                                  worker=i,
                                  rapid_crash_count=rapid_crash_count[i],
                                  global_t=current_step, success=ok)
                    if ok:
                        rapid_crash_count[i] = 0

                base_wait = float(config.carla_server_start_period)
                max_wait = float(getattr(config, 'carla_restart_backoff_max',
                                         max(base_wait, 300.0)))
                wait_time = min(max_wait, base_wait * (
                    2 ** max(0, restart_counts[i] - 1)))
                print('[RESTART] W{} waiting {:.1f}s before relaunch'.format(
                    i, wait_time), flush=True)
                if shutdown_event.wait(wait_time):
                    break

                port = config.start_port + config.port_step * i
                device = config.worker_gpus[i]
                workers[i] = _start_worker(
                    i, global_network, config, port, device, run_output_dir,
                    shutdown_event, log_queue, run_id)

    except KeyboardInterrupt:
        print('[SUPERVISOR] KeyboardInterrupt, stopping workers', flush=True)
        shutdown_event.set()

    for w in workers.values():
        if w.is_alive():
            w.join(timeout=10)
        if w.is_alive():
            w.terminate()
            w.join(timeout=5)

    return restart_counts
