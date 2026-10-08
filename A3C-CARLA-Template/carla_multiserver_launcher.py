"""Launch and supervise a grid of CARLA servers.

One supervisor thread per server:
  * starts the process on the assigned GPU / RPC port,
  * restarts it when the process exits (crash),
  * kills and restarts it when the process is alive but the simulator
    stopped answering RPC calls (hung server).

Spawn backends (--runtime):
  * apptainer — ``apptainer exec --nv IMAGE BINARY ...``
  * docker    — ``docker run --rm --name NAME --network host --ipc=host --gpus device=N IMAGE ...``

This script does not start training. The pipeline scripts in ``examples/``
start it next to ``train_a3c.py``, which waits for the servers itself.

All start/crash/restart/stop events go to ``<outdir>/server_logs/servers.log``
and to stdout. Raw CARLA output of server i is kept in
``<outdir>/server_logs/carla_server_<i>.log``.
"""

import argparse
import getpass
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
from time import time

from settings import load_dotenv, CARLA_HOST, PORT, PORT_STEP


CHECK_INTERVAL = 30.0
HANG_STRIKES = 3
RESTART_DELAY = 5.0
RPC_TIMEOUT = 5.0  # one probe call; keep it short
# One server uses its RPC port, RPC+1 (streaming), and RPC+2 (secondary).
PORTS_PER_SERVER = 3
# CARLA command-line flags. Add your own here, e.g. "-quality-level=Low".
CARLA_FLAGS = ["-RenderOffScreen", "-nosound"]
DEFAULT_BINARY = "/home/carla/CarlaUE4.sh"
# Apptainer: Unreal -graphicsadapter = visible CUDA index + this offset.
# 1 is right on the cluster this template comes from; yours can differ.
# Check with nvidia-smi that each CarlaUE4 runs on the GPU you expect.
APPTAINER_ADAPTER_OFFSET = 1
# Docker containers are named <prefix>-<user>-<rpc port>.
DOCKER_NAME_PREFIX = "a3c-carla"

LOG = logging.getLogger("carla_servers")
LOG_DIR = ""
SERVER_PROCS = {}
STOP = threading.Event()
CONFIG = None


def parse_args(argv=None):
    """CLI for runtime, server count, ports, image, and binary paths."""
    parser = argparse.ArgumentParser(
        description="Start and supervise CARLA servers for A3C workers")
    parser.add_argument(
        "--runtime",
        choices=("apptainer", "docker"),
        required=True,
    )
    parser.add_argument("--num-servers", type=int, default=1)
    parser.add_argument("--servers-per-gpu", type=int, default=1)
    parser.add_argument(
        "--server-gpu-start",
        type=int,
        default=0,
        help="First visible CUDA index to place servers on (0-based).",
    )
    parser.add_argument("--start-port", type=int, default=PORT)
    parser.add_argument("--port-step", type=int, default=PORT_STEP)
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument(
        "--image",
        default=os.environ.get("CARLA_CONTAINER_IMAGE", ""),
        help="Container image (.sif or docker tag).",
    )
    parser.add_argument(
        "--binary",
        default=os.environ.get("CARLA_BINARY", "") or DEFAULT_BINARY,
        help="CARLA entrypoint inside the image.",
    )
    return parser.parse_args(argv)


def hang_warmup_allows_miss(ever_alive):
    """True while the simulator has never answered; do not count hang strikes."""
    return not bool(ever_alive)


def launcher_config_error(args):
    """Return a fatal config message, or None if the launcher may start."""
    if args.port_step < PORTS_PER_SERVER:
        return (
            "ERROR: --port-step must be >= {} "
            "(a server uses its RPC port and the next two)"
            .format(PORTS_PER_SERVER)
        )
    if not args.image or "CHANGE_ME" in args.image:
        return (
            "ERROR: set --image or CARLA_CONTAINER_IMAGE for runtime={}"
            .format(args.runtime)
        )
    return None


def port_is_free(port):
    """True when nothing on this host listens on TCP ``port``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        # Ignore connections in TIME_WAIT from an earlier run.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("", port))
        except OSError:
            return False
    return True


def busy_ports(start_port, port_step, num_servers):
    """Ports of the server grid (RPC, +1, +2 per server) that are taken."""
    return [
        start_port + idx * port_step + offset
        for idx in range(num_servers)
        for offset in range(PORTS_PER_SERVER)
        if not port_is_free(start_port + idx * port_step + offset)
    ]


def _rpc_client(port):
    import carla
    client = carla.Client(CARLA_HOST, port)
    client.set_timeout(RPC_TIMEOUT)
    return client


def carla_rpc_up(port):
    """True when the RPC server answers. Use for readiness.

    ``get_server_version`` is answered by the RPC thread pool even when the
    game thread hangs, so it cannot detect a hung simulator.
    """
    try:
        _rpc_client(port).get_server_version()
        return True
    except RuntimeError:
        return False


def carla_sim_alive(port):
    """True when the simulator answers. Use for the health check.

    ``get_world`` needs the game thread, so a hung simulator fails it.
    """
    try:
        _rpc_client(port).get_world()
        return True
    except RuntimeError:
        return False


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


def docker_container_name(port):
    """Container name of the server on ``port``: unique per host and user."""
    return "{}-{}-{}".format(DOCKER_NAME_PREFIX, getpass.getuser(), port)


def remove_container(name):
    """``docker rm -f``: stop this server's container or clear a leftover.

    Killing the ``docker run`` client leaves the container running, and a
    killed launcher cannot clean up. The next start removes it by name.
    """
    subprocess.run(
        ["docker", "rm", "-f", name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def graphics_adapter_for(runtime, cuda_index):
    """Unreal -graphicsadapter index for this spawn backend.

    Docker exposes a single GPU inside the container, so adapter 0.
    Apptainer: see APPTAINER_ADAPTER_OFFSET.
    """
    if runtime == "docker":
        return 0
    return int(cuda_index) + APPTAINER_ADAPTER_OFFSET


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
                "docker", "run", "--rm",
                "--name", docker_container_name(port),
                "--network", "host", "--ipc=host",
                "--gpus", "device={}".format(docker_gpu_device(cuda_index)),
                "--entrypoint", binary,
                image,
            ]
            + CARLA_FLAGS
            + [rpc, adapter]
            + extra_args
        )
    raise ValueError("unsupported runtime: {}".format(runtime))


def setup_logging(outdir):
    """Write ``servers.log`` under ``<outdir>/server_logs`` and echo to stdout."""
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
    """Count GPUs from ``CUDA_VISIBLE_DEVICES`` or ``nvidia-smi -L``."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        ids = [item.strip() for item in cvd.split(",") if item.strip() != ""]
        if ids:
            return max(1, len(ids))
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


def terminate(proc, container=None, grace=5.0):
    """SIGTERM the whole process group (start_new_session -> pgid == pid).

    Docker: remove the container first; killing the client leaves it running.
    """
    if container:
        remove_container(container)
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
    """Start one CARLA process and restart it on crash or hung simulator."""
    port = CONFIG.start_port + idx * CONFIG.port_step
    cuda_index = cuda_index_for(
        idx, num_gpus, CONFIG.servers_per_gpu, CONFIG.server_gpu_start)
    cmd = build_cmd(
        CONFIG.runtime, port, cuda_index, CONFIG.image, CONFIG.binary)
    container = docker_container_name(port) \
        if CONFIG.runtime == "docker" else None
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
            if container:
                remove_container(container)
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
            ever_alive = False
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
                    if carla_sim_alive(port):
                        hang_strikes = 0
                        ever_alive = True
                    elif hang_warmup_allows_miss(ever_alive):
                        LOG.info(
                            "server %d: waiting for the simulator on port %d",
                            idx,
                            port,
                        )
                    else:
                        hang_strikes += 1
                        if hang_strikes >= HANG_STRIKES:
                            restarts += 1
                            LOG.warning(
                                "server %d: alive but the simulator on port %d did not answer for %.0fs; killing as hung (restart #%d)",
                                idx,
                                port,
                                HANG_STRIKES * CHECK_INTERVAL,
                                restarts,
                            )
                            terminate(proc, container)
                            STOP.wait(RESTART_DELAY)
                            break
                        LOG.info(
                            "server %d: simulator on port %d did not answer (check %d/%d)",
                            idx,
                            port,
                            hang_strikes,
                            HANG_STRIKES,
                        )
                STOP.wait(1.0)
    finally:
        if proc is not None and (container or proc.poll() is None):
            terminate(proc, container)
            LOG.info("server %d: stopped", idx)
        SERVER_PROCS.pop(idx, None)


def main(argv=None):
    """Parse args, spawn one supervisor thread per server, wait until stop."""
    global CONFIG
    load_dotenv()
    args = parse_args(argv)
    config_error = launcher_config_error(args)
    if config_error:
        sys.exit(config_error)
    try:
        import carla  # noqa: F401  the health check needs it
    except ImportError as error:
        sys.exit(
            "ERROR: cannot import carla ({}). Run the launcher with the "
            "same Python environment as training.".format(error))
    CONFIG = args

    setup_logging(args.outdir)
    if args.runtime == "docker":
        # Containers left by a killed launcher still hold the ports.
        for idx in range(args.num_servers):
            remove_container(docker_container_name(
                args.start_port + idx * args.port_step))
    busy = busy_ports(args.start_port, args.port_step, args.num_servers)
    if busy:
        LOG.error(
            "port(s) %s already in use, probably by another job on this "
            "node. Pass a different --start-port.", busy)
        sys.exit(1)
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
