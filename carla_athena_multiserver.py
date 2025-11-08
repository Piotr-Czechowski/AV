"""
Run multiple CARLA servers, each inside its own Apptainer container.
If any server exits, it gets relaunched after `DELAY` seconds.

PORT MAP
────────
instance 0 → 2000, gpu 0
instance 1 → 2100, gpu 0
instance 2 → 2200, gpu 1
instance 3 → 2300, gpu 2
"""

import multiprocessing as mp
import os
import signal
import subprocess
from time import sleep
from typing import List
import torch


# MAIN SETTINGS
NUM_SERVERS: int = 4
SERVERS_PER_GPU = 1

# COMMON - SHOULD BE GOOD
CONTAINER_IMAGE: str = (
    "/net/tscratch/people/plgbartoszkawa/carla_0.9.15.sif"
)
CARLA_BINARY: List[str] = [
    "/home/carla/CarlaUE4.sh",
    "-RenderOffScreen",
    "-nosound",
    "--carla-server",
]

START_PORT: int = 2000
PORT_STEP: int = 100
DELAY: int = 0.1  # seconds
BASE_CMD: List[str] = ["apptainer", "exec", "--nv", CONTAINER_IMAGE]

# Map instance index → GPU ID (1–4)
NUM_GPUS = torch.cuda.device_count()
GPU_MAP = {i: i // SERVERS_PER_GPU + 1 for i in range(NUM_GPUS * SERVERS_PER_GPU)}


def supervise(idx: int) -> None:
    """Launch & babysit one CARLA instance identified by `idx`."""
    port = START_PORT + (idx * PORT_STEP)
    gpu_id = GPU_MAP.get(idx, 1)  # default to GPU 1 if not mapped
    cmd = BASE_CMD + CARLA_BINARY + [
        f"-carla-rpc-port={port}",
        f"-graphicsadapter={gpu_id}",
    ]

    while True:
        try:
            print(f"[{idx}] Launching: {' '.join(cmd)}", flush=True)
            proc = subprocess.Popen(cmd, preexec_fn=os.setsid)
            proc.wait()

            print(
                f"[{idx}] Carla (port {port}, GPU {gpu_id}) exited with code {proc.returncode}. "
                f"Restarting in {DELAY}s…",
                flush=True,
            )
            sleep(DELAY)

        except KeyboardInterrupt:
            print(f"[{idx}] Keyboard interrupt → terminating server…", flush=True)
            _kill_process_tree(proc)
            break

        except Exception as exc:
            print(f"[{idx}] Error: {exc}. Restarting in {DELAY}s…", flush=True)
            sleep(DELAY)


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """SIGTERM the entire process group started via `setsid`."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass


# main
def main() -> None:
    print(f"Starting {NUM_SERVERS} CARLA servers (first port {START_PORT})")
    print(f'Using GPU map {GPU_MAP}')

    workers: List[mp.Process] = []
    for idx in range(NUM_SERVERS):
        p = mp.Process(target=supervise, args=(idx,), daemon=True)
        p.start()
        workers.append(p)

    try:
        for p in workers:
            p.join()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt – shutting down all servers…", flush=True)
        for p in workers:
            p.terminate()
        for p in workers:
            p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
