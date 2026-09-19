# Launch

Training and CARLA are separate processes. `train_a3c.py` only connects to `CARLA_HOST:start_port + i * port_step`.

Each example shows **both** steps. Only the server command changes.

## Local (native `CarlaUE4.sh`)

```bash
cp env.example .env          # set CARLA_PATH=/path/to/CARLA_0.9.15
# terminal A
./examples/local/run_servers.sh -w 2 --outdir runs/demo
# terminal B
./examples/run_train.sh -w 2 --outdir runs/demo
```

## Docker (Linux + NVIDIA Container Toolkit)

```bash
cp env.example .env          # set CARLA_CONTAINER_IMAGE=carlasim/carla:0.9.15
# terminal A
./examples/docker/run_servers.sh -w 2 --outdir runs/demo
# terminal B
./examples/run_train.sh -w 2 --outdir runs/demo
```

`--network host` keeps RPC and the streaming port (RPC+1) on the host. This recipe is for Linux; Docker Desktop on macOS does not share host networking that way.

## HPC / Apptainer (one job starts both layers)

Edit `#SBATCH --partition` / `--account` in `examples/hpc/train.slurm`, set `CARLA_CONTAINER_IMAGE` in `.env`, then:

```bash
sbatch examples/hpc/train.slurm -w 2 --workers-per-gpu 1 --servers-per-gpu 1
```

`--no-carla` skips the launcher if servers are already up.

## Launcher directly

```bash
python carla_multiserver_launcher.py \
  --runtime native|docker|apptainer \
  --num-servers 2 \
  --start-port 2000 \
  --port-step 100 \
  --outdir runs/demo
```

`--image` is required for docker and apptainer. There is no `--runtime external`: if CARLA already listens, skip the launcher and run `examples/run_train.sh` (or `train_a3c.py`) alone.

The launcher restarts a server that exits and kills one whose RPC port stops listening.

## Port grid

Worker `i` uses port `start_port + i * port_step` (default 2000, 2100, ...). Start as many servers as `--num-workers` / `-w`.
