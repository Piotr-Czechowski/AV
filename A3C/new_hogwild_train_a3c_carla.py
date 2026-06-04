"""A3C multi-GPU CARLA training entry point.

The CLI exposes run-shaping arguments. Stable algorithm, reward, and recovery
defaults live as uppercase module-level variables below so normal SLURM
launches do not need to pass a long list of constant values.
"""

import argparse
import json
import os
import random
import signal
import sys
import time
import uuid
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
import torch.multiprocessing as mp

from ACTIONS import ACTIONS as ac
from new_hogwild_prepare_output_dir import prepare_output_dir
from new_hogwild_training_logger import TrainingLogger
from new_hogwild_system_monitor import RunMonitor
from new_hogwild_a3c import GlobalNetwork
from new_hogwild_run_a3c import run_with_restart, find_latest_checkpoint

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# Run shape
DEFAULT_NUM_WORKERS = 2
DEFAULT_WORKERS_PER_GPU = 2
DEFAULT_WORKER_GPU_START = 0
DEFAULT_START_PORT = 2000
DEFAULT_PORT_STEP = 100
DEFAULT_SCENARIO = 14
DEFAULT_CAMERA = 'semantic'
DEFAULT_RES = 250
DEFAULT_MP_DENSITY = 25
DEFAULT_SEED = 52

# Algorithm
DEFAULT_OPTIMIZER = 'shared-rmsprop'
DEFAULT_ROLLOUT_LENGTH = 20
DEFAULT_GAMMA = 0.99
DEFAULT_LR = 1e-4
DEFAULT_BETA_START = 0.02
DEFAULT_BETA_END = 0.002
DEFAULT_BETA_ANNEAL_FRAC = 0.6
DEFAULT_VALUE_LOSS_COEF = 1.0
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_MAX_GRAD_NORM = 5.0
DEFAULT_NORMALIZE_ADVANTAGES = True
DEFAULT_REWARD_SCALE = 0.0
DEFAULT_STEPS = 10_000_000
DEFAULT_SYNC_EVERY_N_UPDATES = 1
DEFAULT_HOGWILD_LOCK_UPDATES = False
DEFAULT_GC_INTERVAL = 10
DEFAULT_TESTING = False

# Optimizer internals
DEFAULT_RMSPROP_ALPHA = 0.99
DEFAULT_RMSPROP_EPS = 1e-5
DEFAULT_ADAM_BETA1 = 0.9
DEFAULT_ADAM_BETA2 = 0.999
DEFAULT_ADAM_EPS = 1e-8

# Recovery
DEFAULT_CARLA_TIMEOUT_WAIT = 60.0
DEFAULT_CARLA_SERVER_START_PERIOD = 30.0
DEFAULT_CARLA_RESTART_BACKOFF_MAX = 300.0
DEFAULT_WORKER_CHECK_INTERVAL = 5.0
DEFAULT_MAX_RESTARTS_PER_WORKER = 160
DEFAULT_RAPID_CRASH_THRESHOLD = 3
DEFAULT_RAPID_CRASH_WINDOW_STEPS = 100
DEFAULT_MAX_CONNECT_RETRIES = 5
DEFAULT_CONNECT_RETRY_WAIT = 30.0

# Logging and monitoring
DEFAULT_LOG_STEPS = False
DEFAULT_LOG_UPDATE_ARRAYS = False
DEFAULT_DIAG_LOG_INTERVAL = 100
DEFAULT_DIAG_LOG_WALL_S = 60.0
DEFAULT_MONITOR_INTERVAL = 10.0
DEFAULT_NO_SYSTEM_MONITOR = False
DEFAULT_NO_GPU_MONITOR = False
DEFAULT_VERBOSE_ENV_LOGS = False
DEFAULT_WANDB_PROJECT = 'a3c-carla'
DEFAULT_WANDB_RUN_NAME = None
DEFAULT_WANDB_ENTITY = None
DEFAULT_NO_WANDB = False
DEFAULT_SAVE_EPISODES = None
DEFAULT_SAVE_EPISODE_INTERVAL = 0

# Checkpointing
DEFAULT_SAVE_FREQUENCY = 100000
DEFAULT_SAVE_WORKER_CHECKPOINTS = False
DEFAULT_OUTDIR = None
DEFAULT_RESUME = None

# CARLA step/reset behavior
DEFAULT_ACTION_REPEAT = 2
DEFAULT_EPISODE_MAX_DECISIONS = 100
DEFAULT_WORLD_RELOAD_INTERVAL = 0
DEFAULT_REWARD_MODE = 'legacy'#'shaped'

# Reward shaping
DEFAULT_REWARD_PROGRESS_COEF = 1.0
DEFAULT_REWARD_TARGET_SPEED_COEF = 1.0
DEFAULT_REWARD_ROUTE_PENALTY_COEF = 0.1
DEFAULT_REWARD_TIME_PENALTY = 0.01
DEFAULT_REWARD_GOAL_BONUS = 50.0
DEFAULT_REWARD_COLLISION_PENALTY = 50.0
DEFAULT_REWARD_OFFROUTE_PENALTY = 25.0
DEFAULT_REWARD_LANE_INVASION_PENALTY = 5.0
DEFAULT_REWARD_TARGET_SPEED_KMH = 20.0
DEFAULT_REWARD_OFFROUTE_THRESHOLD = 10.0
DEFAULT_REWARD_CLIP = 50.0

# Fixed implementation choices
DEFAULT_ACTION_TYPE = 'discrete'


# ---------------------------------------------------------------------------
# W&B subprocess (non-blocking I/O path)
# ---------------------------------------------------------------------------

def wandb_logger_process(log_queue, shutdown_event, config, run_output_dir,
                        wandb_id):
    if not HAS_WANDB or config.no_wandb:
        return
    os.environ['WANDB_INSECURE_DISABLE_SSL'] = 'true'
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name or 'a3c-{}w-{}'.format(
            config.num_workers, wandb_id),
        id=wandb_id,
        resume='allow',
        dir=run_output_dir,
        config=_config_to_dict(config),
    )
    try:
        while not shutdown_event.is_set():
            try:
                record = log_queue.get(timeout=1.0)
            except Exception:
                continue
            if record is None:
                break
            worker_id = record.pop('worker_id', None)
            metrics = {}
            for k, v in record.items():
                if worker_id is not None and k not in \
                        ('episode', 'global_step'):
                    metrics['worker/{}/{}'.format(worker_id, k)] = v
                else:
                    metrics[k] = v
            try:
                wandb.log(metrics)
            except Exception:
                pass
    except Exception:
        pass
    finally:
        try:
            wandb.finish()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description='A3C multi-GPU training for CARLA autonomous driving')

    # Run shape: these are the values normally changed from SLURM.
    p.add_argument('--num-workers', type=int,
                   default=DEFAULT_NUM_WORKERS)
    p.add_argument('--workers-per-gpu', type=int,
                   default=DEFAULT_WORKERS_PER_GPU)
    p.add_argument('--worker-gpu-start', type=int,
                   default=DEFAULT_WORKER_GPU_START)
    p.add_argument('--start-port', type=int,
                   default=DEFAULT_START_PORT)
    p.add_argument('--port-step', type=int,
                   default=DEFAULT_PORT_STEP)
    p.add_argument('--scenario', type=int, nargs='+',
                   default=[DEFAULT_SCENARIO])
    p.add_argument('--camera', type=str, default=DEFAULT_CAMERA,
                   choices=['rgb', 'semantic'])
    p.add_argument('--res', type=int, default=DEFAULT_RES)
    p.add_argument('--seed', type=int, default=DEFAULT_SEED)

    # Algorithm knobs kept as CLI overrides for experiments; their normal
    # values live in the uppercase constants above.
    p.add_argument('--optimizer', type=str,
                   default=DEFAULT_OPTIMIZER,
                   choices=['shared-rmsprop', 'shared-adam'])
    p.add_argument('--rollout-length', type=int,
                   default=DEFAULT_ROLLOUT_LENGTH,
                   dest='rollout_length',
                   help='Worker rollout length before each optimizer update '
                        '(was --t-max in earlier revisions).')
    p.add_argument('--gamma', type=float, default=DEFAULT_GAMMA)
    p.add_argument('--lr', type=float, default=DEFAULT_LR)
    p.add_argument('--beta', type=float, default=None,
                   dest='entropy_coef',
                   help='Constant entropy coefficient. Overrides beta '
                        'schedule.')
    p.add_argument('--beta-start', type=float,
                   default=DEFAULT_BETA_START)
    p.add_argument('--beta-end', type=float,
                   default=DEFAULT_BETA_END)
    p.add_argument('--beta-anneal-frac', type=float,
                   default=DEFAULT_BETA_ANNEAL_FRAC)
    p.add_argument('--value-loss-coef', type=float,
                   default=DEFAULT_VALUE_LOSS_COEF)
    p.add_argument('--weight-decay', type=float,
                   default=DEFAULT_WEIGHT_DECAY)
    p.add_argument('--max-grad-norm', type=float,
                   default=DEFAULT_MAX_GRAD_NORM)
    p.add_argument('--no-normalize-advantages', action='store_false',
                   dest='normalize_advantages',
                   default=DEFAULT_NORMALIZE_ADVANTAGES)
    p.add_argument('--reward-scale', type=float,
                   default=DEFAULT_REWARD_SCALE)
    p.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    p.add_argument('--sync-every-n-updates', type=int,
                   default=DEFAULT_SYNC_EVERY_N_UPDATES)
    p.add_argument('--hogwild-lock-updates', action='store_true',
                   default=DEFAULT_HOGWILD_LOCK_UPDATES)
    p.add_argument('--gc-interval', type=int,
                   default=DEFAULT_GC_INTERVAL)
    p.add_argument('--testing', action='store_true',
                   default=DEFAULT_TESTING)

    # Checkpointing / resume.
    p.add_argument('--save-frequency', type=int,
                   default=DEFAULT_SAVE_FREQUENCY)
    p.add_argument('--save-worker-checkpoints', action='store_true',
                   default=DEFAULT_SAVE_WORKER_CHECKPOINTS)
    p.add_argument('--outdir', type=str, default=DEFAULT_OUTDIR)
    p.add_argument('--resume', type=str, default=DEFAULT_RESUME)

    # Recovery / supervisor.
    p.add_argument('--carla-timeout-wait', type=float,
                   default=DEFAULT_CARLA_TIMEOUT_WAIT)
    p.add_argument('--carla-server-start-period', type=float,
                   default=DEFAULT_CARLA_SERVER_START_PERIOD)
    p.add_argument('--carla-restart-backoff-max', type=float,
                   default=DEFAULT_CARLA_RESTART_BACKOFF_MAX)
    p.add_argument('--worker-check-interval', type=float,
                   default=DEFAULT_WORKER_CHECK_INTERVAL)
    p.add_argument('--max-restarts-per-worker', type=int,
                   default=DEFAULT_MAX_RESTARTS_PER_WORKER)
    p.add_argument('--rapid-crash-threshold', type=int,
                   default=DEFAULT_RAPID_CRASH_THRESHOLD)
    p.add_argument('--rapid-crash-window-steps', type=int,
                   default=DEFAULT_RAPID_CRASH_WINDOW_STEPS)
    p.add_argument('--max-connect-retries', type=int,
                   default=DEFAULT_MAX_CONNECT_RETRIES)
    p.add_argument('--connect-retry-wait', type=float,
                   default=DEFAULT_CONNECT_RETRY_WAIT)

    # Logging / monitoring.
    p.add_argument('--log-steps', action='store_true',
                   default=DEFAULT_LOG_STEPS)
    p.add_argument('--log-update-arrays', action='store_true',
                   default=DEFAULT_LOG_UPDATE_ARRAYS)
    p.add_argument('--diag-log-interval', type=int,
                   default=DEFAULT_DIAG_LOG_INTERVAL)
    p.add_argument('--diag-log-wall-s', type=float,
                   default=DEFAULT_DIAG_LOG_WALL_S)
    p.add_argument('--monitor-interval', type=float,
                   default=DEFAULT_MONITOR_INTERVAL)
    p.add_argument('--no-system-monitor', action='store_true',
                   default=DEFAULT_NO_SYSTEM_MONITOR)
    p.add_argument('--no-gpu-monitor', action='store_true',
                   default=DEFAULT_NO_GPU_MONITOR)
    p.add_argument('--verbose-env-logs', action='store_true',
                   default=DEFAULT_VERBOSE_ENV_LOGS)
    p.add_argument('--wandb-project', type=str,
                   default=DEFAULT_WANDB_PROJECT)
    p.add_argument('--wandb-run-name', type=str,
                   default=DEFAULT_WANDB_RUN_NAME)
    p.add_argument('--wandb-entity', type=str,
                   default=DEFAULT_WANDB_ENTITY)
    p.add_argument('--no-wandb', action='store_true',
                   default=DEFAULT_NO_WANDB)
    p.add_argument('--save-episodes', type=int, nargs='+',
                   default=DEFAULT_SAVE_EPISODES)
    p.add_argument('--save-episode-interval', type=int,
                   default=DEFAULT_SAVE_EPISODE_INTERVAL)

    # CARLA stepping and reset behavior. Reward coefficients are fixed globals
    # in uppercase constants; only legacy-vs-shaped mode remains a run option.
    p.add_argument('--action-repeat', type=int,
                   default=DEFAULT_ACTION_REPEAT)
    p.add_argument('--episode-max-decisions', type=int,
                   default=DEFAULT_EPISODE_MAX_DECISIONS)
    p.add_argument('--world-reload-interval', type=int,
                   default=DEFAULT_WORLD_RELOAD_INTERVAL)
    p.add_argument('--reward-mode', type=str,
                   default=DEFAULT_REWARD_MODE,
                   choices=['shaped', 'legacy'])

    return p


def _config_to_dict(config):
    if isinstance(config, dict):
        return config
    if hasattr(config, '__dict__'):
        return {k: v for k, v in vars(config).items()
                if not k.startswith('_')}
    return {}


def _apply_config_defaults(args):
    # Internal defaults below are kept out of the normal CLI because they are
    # rarely changed and define the current CARLA A3C setup.
    if not hasattr(args, 'mp_density'):
        args.mp_density = DEFAULT_MP_DENSITY
    if not hasattr(args, 'rmsprop_alpha'):
        args.rmsprop_alpha = DEFAULT_RMSPROP_ALPHA
    if not hasattr(args, 'rmsprop_eps'):
        args.rmsprop_eps = DEFAULT_RMSPROP_EPS
    if not hasattr(args, 'adam_beta1'):
        args.adam_beta1 = DEFAULT_ADAM_BETA1
    if not hasattr(args, 'adam_beta2'):
        args.adam_beta2 = DEFAULT_ADAM_BETA2
    if not hasattr(args, 'adam_eps'):
        args.adam_eps = DEFAULT_ADAM_EPS
    if not hasattr(args, 'reward_progress_coef'):
        args.reward_progress_coef = DEFAULT_REWARD_PROGRESS_COEF
    if not hasattr(args, 'reward_target_speed_coef'):
        args.reward_target_speed_coef = DEFAULT_REWARD_TARGET_SPEED_COEF
    if not hasattr(args, 'reward_route_penalty_coef'):
        args.reward_route_penalty_coef = DEFAULT_REWARD_ROUTE_PENALTY_COEF
    if not hasattr(args, 'reward_time_penalty'):
        args.reward_time_penalty = DEFAULT_REWARD_TIME_PENALTY
    if not hasattr(args, 'reward_goal_bonus'):
        args.reward_goal_bonus = DEFAULT_REWARD_GOAL_BONUS
    if not hasattr(args, 'reward_collision_penalty'):
        args.reward_collision_penalty = DEFAULT_REWARD_COLLISION_PENALTY
    if not hasattr(args, 'reward_offroute_penalty'):
        args.reward_offroute_penalty = DEFAULT_REWARD_OFFROUTE_PENALTY
    if not hasattr(args, 'reward_lane_invasion_penalty'):
        args.reward_lane_invasion_penalty = \
            DEFAULT_REWARD_LANE_INVASION_PENALTY
    if not hasattr(args, 'reward_target_speed_kmh'):
        args.reward_target_speed_kmh = DEFAULT_REWARD_TARGET_SPEED_KMH
    if not hasattr(args, 'reward_offroute_threshold'):
        args.reward_offroute_threshold = DEFAULT_REWARD_OFFROUTE_THRESHOLD
    if not hasattr(args, 'reward_clip'):
        args.reward_clip = DEFAULT_REWARD_CLIP
    if not hasattr(args, 'action_type'):
        args.action_type = DEFAULT_ACTION_TYPE
    return args


def _timestamp():
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]


def _read_resume_state(run_output_dir):
    state = {}
    state_path = os.path.join(run_output_dir, 'resume_state.json')
    try:
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
    except (OSError, ValueError, TypeError):
        state = {}

    if 'elapsed_training_s' not in state:
        events_path = os.path.join(run_output_dir, 'logs', 'events.jsonl')
        elapsed = None
        try:
            with open(events_path) as f:
                for line in f:
                    try:
                        log_record = json.loads(line)
                    except ValueError:
                        continue
                    if log_record.get('event') == 'training_end':
                        value = log_record.get('cumulative_elapsed_s')
                        if value is not None:
                            elapsed = max(float(value), elapsed or 0.0)
        except (OSError, TypeError, ValueError):
            elapsed = None
        if elapsed is not None:
            state['elapsed_training_s'] = elapsed
    return state


def _write_resume_state(run_output_dir, config, global_network,
                        elapsed_training_s, last_session_elapsed_s,
                        session_start_ts, session_end_ts):
    state = {
        'global_step': global_network.global_step.value,
        'global_episode': global_network.global_episode.value,
        'total_updates': global_network.total_updates.value,
        'elapsed_training_s': float(elapsed_training_s),
        'last_session_elapsed_s': float(last_session_elapsed_s),
        'last_session_start_ts': session_start_ts,
        'last_session_end_ts': session_end_ts,
        'training_args': _config_to_dict(config),
        'timestamp': session_end_ts,
    }
    state_path = os.path.join(run_output_dir, 'resume_state.json')
    tmp_path = state_path + '.tmp'
    try:
        with open(tmp_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp_path, state_path)
    except OSError as e:
        print('[RESUME] failed to write {}: {}'.format(state_path, e),
              flush=True)


def _install_signal_handlers(shutdown_event):
    def _handle_signal(signum, _frame):
        try:
            name = signal.Signals(signum).name
        except Exception:
            name = str(signum)
        print('[SIGNAL] received {}; requesting graceful shutdown'.format(
            name), flush=True)
        shutdown_event.set()

    for signame in ('SIGTERM', 'SIGUSR1', 'SIGINT'):
        sig = getattr(signal, signame, None)
        if sig is not None:
            signal.signal(sig, _handle_signal)


def _assign_worker_gpus(num_workers, workers_per_gpu, worker_gpu_start):
    n_gpus = torch.cuda.device_count()
    if workers_per_gpu <= 0 or n_gpus == 0:
        return ['cpu'] * num_workers
    if worker_gpu_start < 0 or worker_gpu_start >= n_gpus:
        raise ValueError(
            '--worker-gpu-start={} is outside visible CUDA device range '
            '0..{}'.format(worker_gpu_start, n_gpus - 1))
    if n_gpus <= 1:
        return ['cuda:0'] * num_workers
    worker_gpu_ids = list(range(worker_gpu_start, n_gpus))
    expanded = [f'cuda:{g}' for g in worker_gpu_ids
                for _ in range(workers_per_gpu)]
    if len(expanded) < num_workers:
        # Repeat the same GPU assignment pattern if there are more workers
        # than configured worker slots.
        while len(expanded) < num_workers:
            expanded.extend(expanded)
    return expanded[:num_workers]


def main():
    args = _apply_config_defaults(build_parser().parse_args())

    if args.entropy_coef is not None:
        # --beta is the simple "constant entropy" shortcut. Internally the
        # rest of the code always reads beta_start/beta_end.
        args.beta_start = float(args.entropy_coef)
        args.beta_end = float(args.entropy_coef)
    else:
        args.entropy_coef = args.beta_start

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.n_actions = len(ac.ACTIONS_NAMES)

    args.worker_gpus = _assign_worker_gpus(
        args.num_workers, args.workers_per_gpu, args.worker_gpu_start)

    user_run_output_dir = args.resume or args.outdir
    run_output_dir = prepare_output_dir(args, user_run_output_dir,
                                        resume=bool(args.resume))
    print('[RUN] outdir: {}'.format(run_output_dir), flush=True)

    config = SimpleNamespace(**vars(args))

    mp.set_start_method('spawn', force=True)
    shutdown_event = mp.Event()
    _install_signal_handlers(shutdown_event)

    log_queue = mp.Queue(maxsize=1000) \
        if (HAS_WANDB and not args.no_wandb) else None

    global_network = GlobalNetwork(
        config, state_shape=[args.res, args.res, 3],
        action_shape=args.n_actions, critic_shape=1)

    elapsed_offset = 0.0
    if args.resume:
        checkpoint_path, checkpoint_step_number = find_latest_checkpoint(
            args.resume)
        if checkpoint_path:
            print('[RESUME] loading {} (step {})'.format(
                checkpoint_path, checkpoint_step_number), flush=True)
            global_network.load(checkpoint_path)

        resume_state = _read_resume_state(run_output_dir)
        if resume_state:
            elapsed_offset = resume_state.get('elapsed_training_s', 0.0)
            saved_args = resume_state.get('training_args', {})
            for key in ('steps', 'lr', 'beta_start', 'beta_end',
                        'beta_anneal_frac', 'rollout_length', 'gamma',
                        'weight_decay', 'optimizer'):
                saved = saved_args.get(key)
                current = getattr(config, key, None)
                if saved is not None and current is not None \
                        and saved != current:
                    print('[RESUME] WARNING: --{} changed {} -> {}'.format(
                        key, saved, current), flush=True)

    n_params_model = sum(p.numel()
                         for p in global_network.model.parameters())
    model_name = 'SharedActorCritic'
    model_extra = {'model_params': n_params_model}
    total_params = n_params_model
    TrainingLogger.write_metadata(
        run_output_dir, _config_to_dict(config),
        model_name=model_name,
        n_params=total_params,
        n_workers=args.num_workers,
        **model_extra
    )

    _events = TrainingLogger(run_output_dir, worker_id=-1, log_steps=False)
    _events.log_event('training_start',
                      global_t=global_network.global_step.value,
                      n_workers=args.num_workers,
                      resumed=bool(args.resume),
                      worker_gpus=args.worker_gpus)
    _events.close()

    run_monitor = None
    if not args.no_system_monitor:
        run_monitor = RunMonitor(run_output_dir,
                                 interval=args.monitor_interval,
                                 track_carla=True,
                                 track_gpu=not args.no_gpu_monitor)
        run_monitor.start()

    wandb_id = None
    wandb_logger_process_handle = None
    if HAS_WANDB and not args.no_wandb:
        wandb_id_file = os.path.join(run_output_dir, 'wandb_run_id.txt')
        if args.resume and os.path.exists(wandb_id_file):
            with open(wandb_id_file) as f:
                wandb_id = f.read().strip()
        else:
            wandb_id = uuid.uuid4().hex[:8]
            try:
                with open(wandb_id_file, 'w') as f:
                    f.write(wandb_id)
            except OSError:
                pass

        wandb_logger_process_handle = mp.Process(
            target=wandb_logger_process,
            args=(log_queue, shutdown_event, config, run_output_dir,
                  wandb_id),
            name='WandBLogger')
        wandb_logger_process_handle.start()

    run_id = os.path.basename(run_output_dir.rstrip('/'))
    start_steps = global_network.global_step.value
    session_start_ts = _timestamp()
    start_time = time.time()
    try:
        run_with_restart(global_network, config, run_output_dir,
                         shutdown_event,
                         log_queue=log_queue, run_id=run_id)
    except KeyboardInterrupt:
        shutdown_event.set()
    finally:
        shutdown_event.set()
        if log_queue is not None:
            try:
                log_queue.put(None)
            except Exception:
                pass
        if wandb_logger_process_handle is not None:
            wandb_logger_process_handle.join(timeout=15)
            if wandb_logger_process_handle.is_alive():
                wandb_logger_process_handle.terminate()
        if run_monitor is not None:
            run_monitor.stop()

    session_elapsed = time.time() - start_time
    cumulative = session_elapsed + elapsed_offset
    final_steps = global_network.global_step.value
    session_steps = max(0, final_steps - start_steps)
    session_end_ts = _timestamp()
    _write_resume_state(run_output_dir, config, global_network, cumulative,
                        session_elapsed, session_start_ts, session_end_ts)
    print('[BENCHMARK] session: {} steps in {:.1f}s '
          '({:.2f} steps/s); cumulative active: {} steps in {:.1f}s '
          '({:.2f} steps/s)'.format(
              session_steps, session_elapsed,
              session_steps / session_elapsed if session_elapsed > 0 else 0,
              final_steps, cumulative,
              final_steps / cumulative if cumulative > 0 else 0),
          flush=True)

    _events = TrainingLogger(run_output_dir, worker_id=-1, log_steps=False)
    _events.log_event('training_end',
                      global_t=final_steps,
                      session_steps=session_steps,
                      session_elapsed_s=round(session_elapsed, 2),
                      cumulative_elapsed_s=round(cumulative, 2),
                      active_time_accounting=True)
    _events.close()


if __name__ == '__main__':
    main()
