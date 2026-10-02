# Launch

Two pipelines are supported. Each one is a single script that starts the CARLA servers, trains, and cleans up.

| | Docker | Apptainer |
|---|---|---|
| Script | `examples/docker/train.sh` | `examples/apptainer/train.slurm` |
| Use it on | Linux workstation or server | Slurm cluster |
| CARLA runs in | containers with `--network host` | `apptainer exec --nv` |
| Training runs | on the host | in the batch job |
| Needs | Docker, NVIDIA Container Toolkit | Apptainer |

Both need a Python 3.10 environment with this template installed (`pip install -e .`). Activate it before you start the script, or set `VENV` so the script activates it.

## What a pipeline does

`train_a3c.py` never starts the simulator. `carla_multiserver_launcher.py` starts and supervises the servers. The pipeline script only puts the two processes together:

1. Read `.env` and the script's own options. Every other argument goes to `train_a3c.py` unchanged.
2. Create the run directory `runs/a3c_<N>w_<date>[_<jobid>]`.
3. Start the launcher in the background, in its own session. Ctrl+C then does not reach it, and the servers stay up until training has stopped.
4. Start `train_a3c.py`. Training waits until every CARLA server answers (at most `SERVER_READY_TIMEOUT_S` = 600 s, a constant in `train_a3c.py`), then starts the workers.
5. On exit or on a signal: stop training first and give it `GRACEFUL_SHUTDOWN_WAIT` (75 s) to write its last checkpoint. Then stop the launcher, which stops the servers.
6. Fail fast: when the launcher dies, the pipeline prints the end of the launcher log and exits with an error.

Options that the scripts read:

| Option | Meaning |
|---|---|
| `-w N`, `--workers N` | A3C workers = CARLA servers (default 1) |
| `--servers-per-gpu N` | CARLA servers on one GPU (default 1) |
| `--start-port N` | first CARLA RPC port |
| `--outdir DIR` | directory of a new run |
| `-r DIR`, `--resume DIR` | continue the run in `DIR` |

Everything else is a `train_a3c.py` argument: `--scenario 14 15`, `--steps 2000000`, `--init-from PATH`, `--workers-per-gpu 2`, and so on. Do not pass `--port-step` this way: the launcher would not get it. Set `CARLA_PORT_STEP` in `.env`.

Files in the run directory that come from the pipeline: `a3c_training.log` (training output), `carla_servers.log` (launcher output), and `server_logs/` (launcher events and raw CARLA output). The Apptainer script adds `gpu_dmon.log`, and under Slurm a `slurm.log` link to the job log.

## Docker

Get CARLA once per machine:

```bash
docker pull carlasim/carla:0.9.15
```

Set `CARLA_CONTAINER_IMAGE=carlasim/carla:0.9.15` in `.env`, then:

```bash
./examples/docker/train.sh -w 1
./examples/docker/train.sh -w 2 --servers-per-gpu 2    # two servers on one GPU
```

Ctrl+C stops training first, then the servers.

The containers use `--network host` and `--ipc=host`, so the RPC port and the two ports after it are host ports. This needs Linux. Docker Desktop on macOS does not share host networking that way.

Each container is named `a3c-carla-<user>-<rpc port>`. The launcher removes it with `docker rm -f` when it stops a server and before it starts one. A container that outlived a killed launcher is removed at the next start. Because the removal goes by name, two Docker runs of one user on one machine must use different ports: give the second run its own `--start-port`.

## Apptainer

Build the image once. Put the cache and the temporary directory on scratch, because the build needs much more disk space than a home quota usually allows:

```bash
export APPTAINER_CACHEDIR=/scratch/$USER/apptainer_cache
export APPTAINER_TMPDIR=/scratch/$USER/apptainer_tmp
apptainer build carla_0.9.15.sif docker://carlasim/carla:0.9.15
```

Set `CARLA_CONTAINER_IMAGE=/path/to/carla_0.9.15.sif` in `.env`. Edit `#SBATCH --partition` and `--account` in `examples/apptainer/train.slurm` (replace `CHANGE_ME`). Submit from the template root:

```bash
sbatch examples/apptainer/train.slurm -w 1
sbatch --gpus=2 --cpus-per-task=16 --mem=50G examples/apptainer/train.slurm -w 2
```

The same script runs without Slurm: `bash examples/apptainer/train.slurm -w 1`.

**Time limit.** `#SBATCH --signal=B:USR1@120` sends SIGUSR1 to the batch shell 120 s before the time limit. The `B:` matters: without it Slurm signals only job steps, and this job has none. The shell forwards SIGUSR1 to training, training writes its last checkpoint, and the job exits with code 138. Continue it with `-r <run directory>`.

**GPU placement.** Under Apptainer the launcher passes `-graphicsadapter = CUDA index + APPTAINER_ADAPTER_OFFSET`. The offset is `1` on the cluster this template was developed on, and it can differ on yours. Check with `nvidia-smi` that each `CarlaUE4` process runs on the GPU you expect. If it does not, change `APPTAINER_ADAPTER_OFFSET` at the top of `carla_multiserver_launcher.py`. Docker always uses adapter `0`, because each container sees one GPU.

**Bind mounts.** Set `APPTAINER_BIND` in `.env`, for example `APPTAINER_BIND=/scratch`. Apptainer reads this variable itself.

## Planning resources

The default `#SBATCH` header (7 CPUs, 25 GB RAM, 1 GPU) is sized for one worker with one server. Scale CPUs and RAM with the number of workers, as in the two-worker example above.

One CARLA server needs several GB of GPU memory. The number depends on the map and the quality level, so measure it: run one server and read `nvidia-smi` or `gpu_dmon.log`. Use `--servers-per-gpu N` only when N times that number fits on the GPU with room for the workers' models. To lower the load of a server, add CARLA flags such as `-quality-level=Low` to `CARLA_FLAGS` at the top of `carla_multiserver_launcher.py`.

By default worker models go one per GPU (`--workers-per-gpu 1`), and the pattern repeats when there are more workers than GPUs. Pass `--workers-per-gpu N` together with `--servers-per-gpu N` to keep each worker's model on the GPU of its server.

## Ports

Worker `i` talks to the server on port `start port + i * port step`. A server uses three ports: RPC, RPC+1 (streaming), and RPC+2 (secondary). The port step is 5 by default (`CARLA_PORT_STEP` in `.env`; the launcher and training must use the same value, so the scripts do not take `--port-step`).

Default start port:

- Docker: 2000, or `CARLA_START_PORT` from `.env`.
- Apptainer on Slurm: `10000 + (job id % 200) * 100`, so two jobs on one node get different port blocks. `--start-port` or `CARLA_START_PORT` overrides it.

Before it starts a server, the launcher checks that every port it needs is free. A busy port is a fatal error (`port(s) ... already in use`): the launcher exits and the pipeline stops. Pass another `--start-port`.

## Health checks

The launcher restarts a server that exits. It also checks every 30 s that the simulator answers a call that needs the game thread (`client.get_world()`). The RPC version call is not used for this, because it still answers when the simulator hangs.

- No strikes are counted until the simulator has answered once.
- After that, three missed checks in a row (about 90 s) mean a hung server. The launcher kills and restarts it, and the worker reconnects.
- A server that loads a map does not answer until the map is loaded. The three-check tolerance covers this.

The timing constants (`CHECK_INTERVAL`, `HANG_STRIKES`, `RPC_TIMEOUT`, `RESTART_DELAY`) are at the top of `carla_multiserver_launcher.py`.

