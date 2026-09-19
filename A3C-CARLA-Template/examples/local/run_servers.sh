#!/usr/bin/env bash
# Native CARLA on the host. Training is a separate process.
#
# 1. cp env.example .env  and set CARLA_PATH to the CARLA_0.9.15 directory
#    (the folder that contains CarlaUE4.sh).
#
# 2. Terminal A — servers:
#      ./examples/local/run_servers.sh -w 2 --outdir runs/demo
#
# 3. Terminal B — training (same -w / ports / outdir):
#      ./examples/run_train.sh -w 2 --outdir runs/demo
#
# Ctrl+C in terminal A stops the supervisor and the CARLA processes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
if [[ -f "${ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${ROOT}/.env"
    set +a
fi

NUM_SERVERS=1
SERVERS_PER_GPU="${SERVERS_PER_GPU:-1}"
SERVER_GPU_START=0
START_PORT="${CARLA_START_PORT:-2000}"
PORT_STEP="${CARLA_PORT_STEP:-100}"
OUTDIR="${ROOT}/runs/demo"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -w|--workers|--num-servers) NUM_SERVERS="$2"; shift 2 ;;
        --servers-per-gpu) SERVERS_PER_GPU="$2"; shift 2 ;;
        --server-gpu-start) SERVER_GPU_START="$2"; shift 2 ;;
        --start-port) START_PORT="$2"; shift 2 ;;
        --port-step) PORT_STEP="$2"; shift 2 ;;
        --outdir) OUTDIR="$2"; shift 2 ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${CARLA_PATH:-}" && ! -x "${CARLA_BINARY:-}" && ! -f "${CARLA_BINARY:-}" ]]; then
    echo "ERROR: set CARLA_PATH in .env to the host CARLA install" >&2
    echo "       (directory containing CarlaUE4.sh), or set CARLA_BINARY" >&2
    exit 1
fi

mkdir -p "${OUTDIR}"
echo "[local] starting ${NUM_SERVERS} native CARLA server(s), ports ${START_PORT}+${PORT_STEP}"
echo "[local] then run: ./examples/run_train.sh -w ${NUM_SERVERS} --start-port ${START_PORT} --port-step ${PORT_STEP} --outdir ${OUTDIR}"

LAUNCHER=(python -u carla_multiserver_launcher.py
    --runtime native
    --num-servers "${NUM_SERVERS}"
    --servers-per-gpu "${SERVERS_PER_GPU}"
    --server-gpu-start "${SERVER_GPU_START}"
    --start-port "${START_PORT}"
    --port-step "${PORT_STEP}"
    --outdir "${OUTDIR}"
)
if [[ -n "${CARLA_PATH:-}" ]]; then
    LAUNCHER+=(--carla-path "${CARLA_PATH}")
fi
if [[ -n "${CARLA_BINARY:-}" ]]; then
    LAUNCHER+=(--binary "${CARLA_BINARY}")
fi
exec "${LAUNCHER[@]}"
