# Launch

Training and CARLA are separate processes. `train_a3c.py` only connects to `CARLA_HOST:start_port + i * port_step`.

Each example shows **both** steps. Only the server command changes. Default is one worker and one GPU (`-w 1`). Two workers need two GPUs: pass `-w 2` **and** `--gpus=2` on Slurm.

## Local (native `CarlaUE4.sh`)

```bash
cp env.example .env          # set CARLA_PATH=/path/to/CARLA_0.9.15
# terminal A
./examples/local/run_servers.sh -w 1 --outdir runs/demo
# terminal B
./examples/run_train.sh -w 1 --outdir runs/demo
```

`run_train.sh` waits until the RPC ports LISTEN (`lsof`, then `ss` is used inside the Python launcher). If you skip wait: `--no-wait`.

## Docker (Linux + NVIDIA Container Toolkit)

```bash
cp env.example .env          # set CARLA_CONTAINER_IMAGE=carlasim/carla:0.9.15
# pull the image once on the node: docker pull carlasim/carla:0.9.15
# terminal A
./examples/docker/run_servers.sh -w 1 --outdir runs/demo
# terminal B
./examples/run_train.sh -w 1 --outdir runs/demo
```

`--network host` keeps RPC and the streaming port (RPC+1) on the host. `--ipc=host` is baked into the launcher (CARLA shared-memory sensors). This recipe is for Linux; Docker Desktop on macOS does not share host networking that way.

`CHANGE_ME` in `CARLA_CONTAINER_IMAGE` is rejected.

## HPC / Apptainer (one job starts both layers)

Edit `#SBATCH --partition` / `--account` in `examples/hpc/train.slurm` (replace `CHANGE_ME`), set `CARLA_CONTAINER_IMAGE` in `.env`, then:

```bash
sbatch examples/hpc/train.slurm -w 1
# two workers:
# sbatch --gpus=2 --cpus-per-task=16 --mem=50G examples/hpc/train.slurm -w 2
```

`--no-carla` skips the launcher if servers are already up.

Apptainer `-graphicsadapter` is **1-based** (`cuda_index + 1`). Docker uses adapter `0` (one GPU in the container). Native uses 0-based `cuda_index`. Do not add a pin-GPU flag; this is intentional.

Slurm sends SIGUSR1 ~90s before walltime. The job forwards USR1 to Python so it can write `checkpoint.pth`, then cleans up. Timeout is not `exit 0`.

## Launcher directly

```bash
python carla_multiserver_launcher.py \
  --runtime native|docker|apptainer \
  --num-servers 1 \
  --start-port 2000 \
  --port-step 100 \
  --outdir runs/demo
```

`--port-step` must be `>= 2` (RPC uses `port` and `port+1` for streaming). `--image` is required for docker and apptainer. There is no `--runtime external`: if CARLA already listens, skip the launcher and run `examples/run_train.sh` (or `train_a3c.py`) alone.

The launcher restarts a server that exits. Hang detection: no hang-strikes until the RPC port has **listened once** (warmup). After that, three missed LISTEN checks (~90s) kill a hung process. `lsof` is fail-closed; if `lsof` cannot say LISTEN, the launcher tries `ss`, then assumes not listening.

Wait-for-ports scripts fail immediately if the launcher PID has died, and dump `carla_servers.log` / `server_logs`.

## Port grid

Worker `i` uses port `start_port + i * port_step` (default 2000, 2100, ...). Start as many servers as `--num-workers` / `-w`. `CARLA_START_PORT` and `CARLA_PORT_STEP` in `.env` feed `settings.py` and the launcher.
