#!/usr/bin/env bash
# Training is the same no matter how CARLA was started.
# Start servers in another terminal first, then from the template root:
#   ./examples/run_train.sh -w 2 --outdir runs/demo
#
# Matching ports: worker i uses START_PORT + i * PORT_STEP (2000, 2100, ...).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
if [[ -f "${ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${ROOT}/.env"
    set +a
fi

NUM_WORKERS=1
WORKERS_PER_GPU=1
WORKER_GPU_START=0
SCENARIO=14
START_PORT="${CARLA_START_PORT:-2000}"
PORT_STEP="${CARLA_PORT_STEP:-100}"
OUTDIR="${ROOT}/runs/demo"
NO_WAIT=""
PYTHON_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -w|--workers) NUM_WORKERS="$2"; shift 2 ;;
        --workers-per-gpu) WORKERS_PER_GPU="$2"; shift 2 ;;
        --worker-gpu-start) WORKER_GPU_START="$2"; shift 2 ;;
        -s|--scenario) SCENARIO="$2"; shift 2 ;;
        --start-port) START_PORT="$2"; shift 2 ;;
        --port-step) PORT_STEP="$2"; shift 2 ;;
        --outdir) OUTDIR="$2"; shift 2 ;;
        --no-wait) NO_WAIT="1"; shift ;;
        --) shift; PYTHON_EXTRA_ARGS+=("$@"); break ;;
        -h|--help)
            sed -n '2,6p' "$0"
            echo "  -w N                 workers (must match number of CARLA servers)"
            echo "  --start-port N       first RPC port (default 2000)"
            echo "  --port-step N        port stride (default 100)"
            echo "  --outdir DIR"
            echo "  --no-wait            do not poll ports before training"
            echo "  --                   extra args for train_a3c.py"
            exit 0 ;;
        *)
            echo "Unknown argument: $1 (use -- before train_a3c.py flags)" >&2
            exit 1 ;;
    esac
done

mkdir -p "${OUTDIR}"

if [[ -z "${NO_WAIT}" ]]; then
    echo "[train] waiting for ${NUM_WORKERS} CARLA port(s) starting at ${START_PORT}..."
    retries=60
    delay=5
    for ((r=1; r<=retries; r++)); do
        all_ok=true
        for ((i=0; i<NUM_WORKERS; i++)); do
            port=$((START_PORT + i * PORT_STEP))
            if ! lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
                all_ok=false
                break
            fi
        done
        if ${all_ok}; then
            echo "[train] all ports LISTEN after ${r} check(s)."
            break
        fi
        if [[ "${r}" -eq "${retries}" ]]; then
            echo "[train] CARLA ports not listening. Start servers first:" >&2
            echo "  ./examples/local/run_servers.sh -w ${NUM_WORKERS} --outdir ${OUTDIR}" >&2
            echo "  ./examples/docker/run_servers.sh -w ${NUM_WORKERS} --outdir ${OUTDIR}" >&2
            exit 1
        fi
        sleep "${delay}"
    done
fi

TRAIN_CMD=(python -u train_a3c.py
    --num-workers "${NUM_WORKERS}"
    --workers-per-gpu "${WORKERS_PER_GPU}"
    --worker-gpu-start "${WORKER_GPU_START}"
    --start-port "${START_PORT}"
    --port-step "${PORT_STEP}"
    --outdir "${OUTDIR}"
    --scenario
)
# shellcheck disable=SC2206
SCENARIO_VALUES=(${SCENARIO})
TRAIN_CMD+=("${SCENARIO_VALUES[@]}")
if [[ ${#PYTHON_EXTRA_ARGS[@]} -gt 0 ]]; then
    TRAIN_CMD+=("${PYTHON_EXTRA_ARGS[@]}")
fi

echo "[train] ${TRAIN_CMD[*]}"
exec "${TRAIN_CMD[@]}"
