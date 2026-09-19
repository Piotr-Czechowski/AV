"""Launch and supervise a grid of CARLA servers.

One supervisor thread per server:
  * starts the process on the assigned GPU / RPC port,
  * restarts it when the process exits (crash),
  * kills and restarts it when the process is alive but its RPC port
    stopped listening (hung server).

Spawn backends (--runtime):
  * apptainer — ``apptainer exec --nv IMAGE BINARY ...``
  * docker    — ``docker run --rm --network host --gpus device=N IMAGE ...``
  * native    — host ``CarlaUE4.sh`` / binary

This script does not start training. ``train_a3c.py`` connects to ports that
are already listening.

All start/crash/restart/stop events go to ``<outdir>/server_logs/servers.log``
and to stdout. Raw CARLA output of server i is kept in
``<outdir>/server_logs/carla_server_<i>.log``.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
from time import time

from settings import load_dotenv


CHECK_INTERVAL = 30.0
HANG_STRIKES = 3
RESTART_DELAY = 5.0
LSOF_TIMEOUT = 5.0
CARLA_FLAGS = ["-RenderOffScreen", "-nosound", "--carla-server"]
DEFAULT_BINARY = "/home/carla/CarlaUE4.sh"

LOG = logging.getLogger("carla_servers")
LOG_DIR = ""
SERVER_PROCS = {}
STOP = threading.Event()
CONFIG = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Start and supervise CARLA servers for A3C workers")
    parser.add_argument(
        "--runtime",
        choices=("apptainer", "docker", "native"),
        default=os.environ.get("CARLA_RUNTIME", "native"),
    )
    parser.add_argument("--num-servers", type=int, default=1)
    parser.add_argument("--servers-per-gpu", type=int, default=2)
    parser.add_argument(
        "--server-gpu-start",
        type=int,
        default=0,
        help="First visible CUDA index to place servers on (0-based).",
    )
    parser.add_argument("--start-port", type=int, default=2000)
    parser.add_argument("--port-step", type=int, default=100)
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument(
        "--image",
        default=os.environ.get("CARLA_CONTAINER_IMAGE", ""),
        help="Container image (.sif or docker tag). Required for apptainer/docker.",
    )
    parser.add_argument(
        "--binary",
        default=os.environ.get("CARLA_BINARY", "") or DEFAULT_BINARY,
        help="CARLA entrypoint inside the image or on the host.",
    )
    parser.add_argument(
        "--carla-path",
        default=os.environ.get("CARLA_PATH", ""),
        help="Host directory containing the native CARLA binary.",
    )
    return parser.parse_args(argv)


def resolve_binary(binary, carla_path, runtime):
    binary = binary or DEFAULT_BINARY
    if runtime == "native" and carla_path:
        if not os.path.isabs(binary):
            return os.path.join(carla_path, binary)
        candidate = os.path.join(carla_path, os.path.basename(binary))
        if os.path.isfile(candidate):
            return candidate
    return binary


def cuda_index_for(server_idx, num_gpus, servers_per_gpu, gpu_start):
    """0-based CUDA index among currently visible GPUs."""
    if num_gpus <= 0:
        return 0
    servers_per_gpu = max(1, int(servers_per_gpu))
    wanted = int(server_idx) // servers_per_gpu + int(gpu_start)
    if wanted < 0:
        return 0
    if wanted >= num_gpus:
        LOG.warning(
            "server %d: more servers than %d GPU(s) can hold; clamping to GPU %d",
            server_idx,
            num_gpus,
            num_gpus - 1,
        )
        return num_gpus - 1
    return wanted


def docker_gpu_device(cuda_index):
    """Host GPU id for ``docker --gpus device=``. Honours CUDA_VISIBLE_DEVICES."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        ids = [item.strip() for item in cvd.split(",") if item.strip() != ""]
        if ids:
            return ids[min(int(cuda_index), len(ids) - 1)]
    return str(cuda_index)


def graphics_adapter_for(runtime, cuda_index):
    """Unreal -graphicsadapter index for this spawn backend.

    Apptainer keeps the historical 1-based adapter used on HPC.
    Docker exposes a single GPU inside the container, so adapter 0.
    Native uses the 0-based visible CUDA index.
    """
    if runtime == "docker":
        return 0
    if runtime == "apptainer":
        return int(cuda_index) + 1
    return int(cuda_index)


def build_cmd(runtime, port, cuda_index, image, binary, extra_args=None):
    """Return argv that starts one CARLA server. Pure function for tests."""
    extra_args = list(extra_args or [])
    rpc = "-carla-rpc-port={}".format(port)
    adapter = "-graphicsadapter={}".format(
        graphics_adapter_for(runtime, cuda_index))

    if runtime == "apptainer":
        if not image:
            raise ValueError(
                "CARLA_CONTAINER_IMAGE / --image is required for apptainer")
        return (
            ["apptainer", "exec", "--nv", image, binary]
            + CARLA_FLAGS
            + [rpc, adapter]
            + extra_args
        )
    if runtime == "docker":
        if not image:
            raise ValueError(
                "CARLA_CONTAINER_IMAGE / --image is required for docker")
        return (
            [
                "docker", "run", "--rm", "--network", "host",
                "--gpus", "device={}".format(docker_gpu_device(cuda_index)),
                "--entrypoint", binary,
                image,
            ]
            + CARLA_FLAGS
            + [rpc, adapter]
            + extra_args
        )
    if runtime == "native":
        return [binary] + CARLA_FLAGS + [rpc, adapter] + extra_args
    raise ValueError("unsupported runtime: {}".format(runtime))


def setup_logging(outdir):
    global LOG_DIR
    LOG_DIR = os.path.join(outdir, "server_logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "servers.log"))
    file_handler.setFormatter(formatter)
    LOG.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOG.addHandler(stream_handler)
    return LOG_DIR


def num_visible_gpus():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        return max(1, len(cvd.split(",")))
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return max(1, len(result.stdout.strip().splitlines()))
    except Exception:
        LOG.warning("could not count GPUs; assuming 1")
        return 1


def port_listening(port):
    try:
        result = subprocess.run(
            ["lsof", "-nP", "-iTCP:{}".format(port), "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=LSOF_TIMEOUT,
            check=False,
        )
    except Exception as exc:
        LOG.warning("port %d: lsof check failed (%s); assuming listening", port, exc)
        return True
    if result.returncode not in (0, 1):
        LOG.warning("port %d: lsof rc=%d; assuming listening", port, result.returncode)
        return True
    return result.returncode == 0


def terminate(proc, grace=5.0):
    """SIGTERM the whole process group (start_new_session -> pgid == pid)."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def supervise(idx, num_gpus):
    port = CONFIG.start_port + idx * CONFIG.port_step
    cuda_index = cuda_index_for(
        idx, num_gpus, CONFIG.servers_per_gpu, CONFIG.server_gpu_start)
    cmd = build_cmd(
        CONFIG.runtime, port, cuda_index, CONFIG.image, CONFIG.binary)
    server_log_path = os.path.join(LOG_DIR, "carla_server_{}.log".format(idx))

    restarts = 0
    proc = None
    try:
        while not STOP.is_set():
            LOG.info(
                "server %d: starting (port %d, CUDA %d, runtime=%s, restart #%d)",
                idx,
                port,
                cuda_index,
                CONFIG.runtime,
                restarts,
            )
            try:
                with open(server_log_path, "ab") as server_log:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=server_log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
            except Exception as exc:
                LOG.error(
                    "server %d: failed to start: %s; retrying in %.0fs",
                    idx,
                    exc,
                    RESTART_DELAY,
                )
                restarts += 1
                STOP.wait(RESTART_DELAY)
                continue
            SERVER_PROCS[idx] = proc
            LOG.info("server %d: running (pid %d)", idx, proc.pid)

            hang_strikes = 0
            next_check = time() + CHECK_INTERVAL
            while not STOP.is_set():
                returncode = proc.poll()
                if returncode is not None:
                    restarts += 1
                    LOG.error(
                        "server %d: exited with code %s; restarting in %.0fs (restart #%d)",
                        idx,
                        returncode,
                        RESTART_DELAY,
                        restarts,
                    )
                    STOP.wait(RESTART_DELAY)
                    break
                if time() >= next_check:
                    next_check = time() + CHECK_INTERVAL
                    if port_listening(port):
                        hang_strikes = 0
                    else:
                        hang_strikes += 1
                        if hang_strikes >= HANG_STRIKES:
                            restarts += 1
                            LOG.warning(
                                "server %d: alive but port %d not listening for %.0fs; killing as hung (restart #%d)",
                                idx,
                                port,
                                HANG_STRIKES * CHECK_INTERVAL,
                                restarts,
                            )
                            terminate(proc)
                            STOP.wait(RESTART_DELAY)
                            break
                        LOG.info(
                            "server %d: port %d not listening yet (check %d/%d)",
                            idx,
                            port,
                            hang_strikes,
                            HANG_STRIKES,
                        )
                STOP.wait(1.0)
    finally:
        if proc is not None and proc.poll() is None:
            terminate(proc)
            LOG.info("server %d: stopped", idx)
        SERVER_PROCS.pop(idx, None)


def main(argv=None):
    global CONFIG
    load_dotenv()
    args = parse_args(argv)
    args.binary = resolve_binary(args.binary, args.carla_path, args.runtime)
    if args.runtime in ("apptainer", "docker"):
        if not args.image or "CHANGE_ME" in args.image:
            sys.exit(
                "ERROR: set --image or CARLA_CONTAINER_IMAGE for runtime={}".format(
                    args.runtime)
            )
    if args.runtime == "native" and not os.path.isfile(args.binary):
        sys.exit(
            "ERROR: native CARLA binary not found: {}. "
            "Set CARLA_PATH or --binary / CARLA_BINARY.".format(args.binary)
        )
    CONFIG = args

    setup_logging(args.outdir)
    num_gpus = num_visible_gpus()
    LOG.info(
        "starting %d CARLA server(s) runtime=%s on %d GPU(s) "
        "(servers-per-gpu=%d, gpu-start=%d), ports %d+%d",
        args.num_servers,
        args.runtime,
        num_gpus,
        args.servers_per_gpu,
        args.server_gpu_start,
        args.start_port,
        args.port_step,
    )
    LOG.info("binary: %s", args.binary)
    if args.image:
        LOG.info("image: %s", args.image)
    LOG.info("log directory: %s", LOG_DIR)

    threads = []
    for idx in range(args.num_servers):
        thread = threading.Thread(
            target=supervise, args=(idx, num_gpus), daemon=True,
            name="server-{}".format(idx),
        )
        thread.start()
        threads.append(thread)

    def shutdown(signum, _frame):
        LOG.info("received signal %s; stopping all servers", signum)
        STOP.set()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        STOP.set()
        for thread in threads:
            thread.join(timeout=15.0)
    LOG.info("all servers stopped")


if __name__ == "__main__":
    main()
