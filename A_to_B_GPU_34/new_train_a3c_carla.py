"""A3C multi-GPU CARLA training – entry point.

Exposes every training / logging / recovery knob from async-rl on the
CLI. Designed to be launched from SLURM; pass --outdir to a job-unique
path and --resume <same path> to continue a previous run.

Example:
    python new_train_a3c_carla.py \
        --num-workers 2 --workers-per-gpu 2 \
        --start-port 2000 --port-step 100 \
        --scenario 14 --camera semantic --res 250 \
        --t-max 20 --lr 1e-4 --beta 0.01 --gamma 0.99 \
        --value-loss-coef 1.0 --weight-decay 0.0 \
        --max-grad-norm 40.0 \
        --steps 10000000 --save-frequency 20000 \
        --outdir ./runs/myrun \
        --wandb-project A_to_B --wandb-run-name myrun
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
import torch.multiprocessing as mp

from ACTIONS import ACTIONS as ac
from new_prepare_output_dir import prepare_output_dir
from new_training_logger import TrainingLogger
from new_system_monitor import RunMonitor
from new_a3c import GlobalNetwork
from new_run_a3c import run_with_restart, find_latest_checkpoint

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ---------------------------------------------------------------------------
# W&B subprocess (non-blocking I/O path)
# ---------------------------------------------------------------------------

def wandb_logger_process(log_queue, shutdown_event, cfg, outdir, wandb_id):
    if not HAS_WANDB or cfg.no_wandb:
        return
    os.environ['WANDB_INSECURE_DISABLE_SSL'] = 'true'
    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name or 'a3c-{}w-{}'.format(cfg.num_workers,
                                                       wandb_id),
        id=wandb_id,
        resume='allow',
        dir=outdir,
        config=_cfg_to_dict(cfg),
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

    # topology / GPU
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--workers-per-gpu', type=int, default=2,
                   help='How many workers share one GPU (CARLA-side counter).')
    p.add_argument('--global-device', type=str, default='cuda:0',
                   help='Device hosting the global network (parameter server).')
    p.add_argument('--worker-gpu-start', type=int, default=1,
                   help='First CUDA index used by workers (0 if only 1 GPU).')

    # CARLA
    p.add_argument('--start-port', type=int, default=2000)
    p.add_argument('--port-step', type=int, default=100)
    p.add_argument('--scenario', type=int, nargs='+', default=[14])
    p.add_argument('--camera', type=str, default='semantic',
                   choices=['rgb', 'semantic'])
    p.add_argument('--res', type=int, default=250)
    p.add_argument('--action-type', type=str, default='discrete')
    p.add_argument('--mp-density', type=int, default=25)
    p.add_argument('--n-actions', type=int, default=None,
                   help='Override action count (defaults to len(ACTIONS_NAMES)).')

    # training
    p.add_argument('--t-max', type=int, default=20)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--beta', type=float, default=0.01,
                   dest='entropy_coef',
                   help='Entropy coefficient (async-rl name: beta).')
    p.add_argument('--value-loss-coef', type=float, default=1.0)
    p.add_argument('--weight-decay', type=float, default=1e-2,
                   help='Adam weight decay (matches a3c_improved.py default).')
    p.add_argument('--max-grad-norm', type=float, default=40.0,
                   help='Gradient clipping norm (<=0 disables).')
    p.add_argument('--reward-scale', type=float, default=0.0,
                   help='Divide raw rewards by this (0 = disabled).')
    p.add_argument('--steps', type=int, default=10_000_000,
                   help='Total env-step budget across all workers.')
    p.add_argument('--sync-every-n-updates', type=int, default=1)
    p.add_argument('--gc-interval', type=int, default=10)
    p.add_argument('--testing', action='store_true',
                   help='Run in evaluation mode (argmax, no updates).')

    # checkpointing / resume
    p.add_argument('--save-frequency', type=int, default=20000,
                   help='Save a checkpoint every N global steps (0 disables).')
    p.add_argument('--outdir', type=str, default=None)
    p.add_argument('--resume', type=str, default=None,
                   help='Resume from this output directory.')

    # recovery / supervisor
    p.add_argument('--max-retries', type=int, default=3)
    p.add_argument('--carla-timeout-wait', type=float, default=60.0)
    p.add_argument('--carla-server-start-period', type=float, default=30.0)
    p.add_argument('--worker-check-interval', type=float, default=5.0)
    p.add_argument('--max-restarts-per-worker', type=int, default=160)
    p.add_argument('--rapid-crash-threshold', type=int, default=3)
    p.add_argument('--rapid-crash-window-steps', type=int, default=100)
    p.add_argument('--max-connect-retries', type=int, default=5)
    p.add_argument('--connect-retry-wait', type=float, default=30.0)

    # logging / monitoring
    p.add_argument('--log-steps', action='store_true',
                   help='Write one line per env step to steps.jsonl.')
    p.add_argument('--log-update-arrays', action='store_true',
                   help='Include full per-step arrays in updates.jsonl.')
    p.add_argument('--diag-log-interval', type=int, default=100,
                   help='Flush timing + diag stats every N episodes (0 off).')
    p.add_argument('--monitor-interval', type=float, default=10.0)
    p.add_argument('--no-system-monitor', action='store_true')
    p.add_argument('--no-gpu-monitor', action='store_true')

    # W&B
    p.add_argument('--wandb-project', type=str, default='A_to_B')
    p.add_argument('--wandb-run-name', type=str, default=None)
    p.add_argument('--no-wandb', action='store_true')

    # image saving
    p.add_argument('--save-episodes', type=int, nargs='+', default=None,
                   help='Save frames for these episodes across any worker.')
    p.add_argument('--save-episode-interval', type=int, default=0,
                   help='Also save frames every N episodes (0 disables).')

    # determinism
    p.add_argument('--seed', type=int, default=52)

    return p


def _cfg_to_dict(cfg):
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, '__dict__'):
        return {k: v for k, v in vars(cfg).items()
                if not k.startswith('_')}
    return {}


def _assign_worker_gpus(num_workers, workers_per_gpu, worker_gpu_start):
    n_gpus = torch.cuda.device_count()
    if n_gpus <= 1:
        return ['cuda:0'] * num_workers
    worker_gpu_ids = list(range(worker_gpu_start, n_gpus))
    if not worker_gpu_ids:
        worker_gpu_ids = [0]
    expanded = [f'cuda:{g}' for g in worker_gpu_ids
                for _ in range(workers_per_gpu)]
    if len(expanded) < num_workers:
        # recycle
        while len(expanded) < num_workers:
            expanded.extend(expanded)
    return expanded[:num_workers]


def main():
    args = build_parser().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.n_actions is None:
        args.n_actions = len(ac.ACTIONS_NAMES)

    args.worker_gpus = _assign_worker_gpus(
        args.num_workers, args.workers_per_gpu, args.worker_gpu_start)

    user_outdir = args.resume or args.outdir
    outdir = prepare_output_dir(args, user_outdir,
                                resume=bool(args.resume))
    print('[RUN] outdir: {}'.format(outdir), flush=True)

    cfg = SimpleNamespace(**vars(args))

    mp.set_start_method('spawn', force=True)
    shutdown_event = mp.Event()

    log_queue = mp.Queue(maxsize=1000) \
        if (HAS_WANDB and not args.no_wandb) else None

    global_network = GlobalNetwork(
        cfg, state_shape=[args.res, args.res, 3],
        action_shape=args.n_actions, critic_shape=1)

    elapsed_offset = 0.0
    if args.resume:
        ckpt, ckpt_step = find_latest_checkpoint(args.resume)
        if ckpt:
            print('[RESUME] loading {} (step {})'.format(ckpt, ckpt_step),
                  flush=True)
            global_network.load(ckpt)

        resume_state_path = os.path.join(outdir, 'resume_state.json')
        if os.path.exists(resume_state_path):
            with open(resume_state_path) as f:
                rs = json.load(f)
            elapsed_offset = rs.get('elapsed_training_s', 0.0)
            saved_args = rs.get('training_args', {})
            for key in ('steps', 'lr', 'entropy_coef', 't_max',
                        'gamma', 'weight_decay'):
                saved = saved_args.get(key)
                current = getattr(cfg, key, None)
                if saved is not None and current is not None \
                        and saved != current:
                    print('[RESUME] WARNING: --{} changed {} -> {}'.format(
                        key, saved, current), flush=True)

    n_params_actor = sum(p.numel() for p in global_network.actor.parameters())
    n_params_critic = sum(p.numel() for p in global_network.critic.parameters())
    TrainingLogger.write_metadata(
        outdir, _cfg_to_dict(cfg),
        model_name='DiscreteActor+Critic',
        n_params=n_params_actor + n_params_critic,
        n_workers=args.num_workers,
        actor_params=n_params_actor,
        critic_params=n_params_critic,
    )

    _events = TrainingLogger(outdir, worker_id=-1, log_steps=False)
    _events.log_event('training_start',
                      global_t=global_network.global_step.value,
                      n_workers=args.num_workers,
                      resumed=bool(args.resume),
                      worker_gpus=args.worker_gpus)
    _events.close()

    run_mon = None
    if not args.no_system_monitor:
        run_mon = RunMonitor(outdir, interval=args.monitor_interval,
                             track_carla=True,
                             track_gpu=not args.no_gpu_monitor)
        run_mon.start()

    wandb_id = None
    wandb_proc = None
    if HAS_WANDB and not args.no_wandb:
        wandb_id_file = os.path.join(outdir, 'wandb_run_id.txt')
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

        wandb_proc = mp.Process(
            target=wandb_logger_process,
            args=(log_queue, shutdown_event, cfg, outdir, wandb_id),
            name='WandBLogger')
        wandb_proc.start()

    run_id = os.path.basename(outdir.rstrip('/'))
    start_time = time.time()
    try:
        run_with_restart(global_network, cfg, outdir, shutdown_event,
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
        if wandb_proc is not None:
            wandb_proc.join(timeout=15)
            if wandb_proc.is_alive():
                wandb_proc.terminate()
        if run_mon is not None:
            run_mon.stop()

    session_elapsed = time.time() - start_time
    cumulative = session_elapsed + elapsed_offset
    final_steps = global_network.global_step.value
    print('[BENCHMARK] {} steps in {:.1f}s cumulative '
          '({:.2f} steps/s overall)'.format(
              final_steps, cumulative,
              final_steps / cumulative if cumulative > 0 else 0),
          flush=True)

    _events = TrainingLogger(outdir, worker_id=-1, log_steps=False)
    _events.log_event('training_end',
                      global_t=final_steps,
                      session_elapsed_s=round(session_elapsed, 2),
                      cumulative_elapsed_s=round(cumulative, 2))
    _events.close()


if __name__ == '__main__':
    main()
