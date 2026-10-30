#!/bin/bash
#SBATCH --partition=CHANGEME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=25G
#SBATCH --time=05:00:00
#SBATCH --gpus=1
#SBATCH --account=CHANGEME
#SBATCH --job-name=basic-a2c
#SBATCH --output=basic-a2c-%j.log
#SBATCH --error=basic-a2c-%j.log
#SBATCH --signal=B:USR1@120

# Submit from AV or this directory:
#   sbatch A_to_B_GPU_34/train_basic_a2c.slurm --scenario 10
# Override the allocation duration at submission time:
#   sbatch --time=02:30:00 A_to_B_GPU_34/train_basic_a2c.slurm --scenario 13
# Supported Town03 scenario IDs: 1-8 and 10-16.

set -uo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SUBMIT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
    if [ -f "${SUBMIT_DIR}/A_to_B_GPU_34/a2c_rgb_try2.py" ]; then
        AV_ROOT="${SUBMIT_DIR}"
        SCRIPT_DIR="${AV_ROOT}/A_to_B_GPU_34"
    elif [ -f "${SUBMIT_DIR}/a2c_rgb_try2.py" ] && \
         [ -f "${SUBMIT_DIR}/train_basic_a2c.slurm" ]; then
        SCRIPT_DIR="${SUBMIT_DIR}"
        AV_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    else
        echo "ERROR: submit from AV or A_to_B_GPU_34; SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
        exit 1
    fi
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    AV_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
TEMPLATE_DIR="${AV_ROOT}/A3C-CARLA-Template"
CARLA_LAUNCHER="${TEMPLATE_DIR}/carla_multiserver_launcher.py"

SCENARIO=10
while [ "$#" -gt 0 ]; do
    case "$1" in
        --scenario)
            if [ "$#" -lt 2 ]; then
                echo "ERROR: --scenario requires an ID." >&2
                exit 2
            fi
            SCENARIO="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '14,19p' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            echo "Usage: sbatch [--time=HH:MM:SS] $0 [--scenario ID]" >&2
            exit 2
            ;;
    esac
done

case "${SCENARIO}" in
    1|2|3|4|5|6|7|8|10|11|12|13|14|15|16) ;;
    *)
        echo "ERROR: unsupported scenario '${SCENARIO}' (valid IDs: 1-8, 10-16)." >&2
        exit 2
        ;;
esac

if [ ! -f "${TEMPLATE_DIR}/.env" ]; then
    echo "ERROR: missing ${TEMPLATE_DIR}/.env" >&2
    exit 1
fi
if [ ! -f "${CARLA_LAUNCHER}" ]; then
    echo "ERROR: CARLA launcher not found: ${CARLA_LAUNCHER}" >&2
    exit 1
fi

set -a
# shellcheck disable=SC1090
source "${TEMPLATE_DIR}/.env"
set +a

module load Python/3.10.4 || {
    echo "ERROR: could not load Python/3.10.4." >&2
    exit 1
}

if [ -z "${VENV:-}" ] || [ ! -x "${VENV}/bin/python" ]; then
    echo "ERROR: VENV must point to a usable Python environment in ${TEMPLATE_DIR}/.env." >&2
    exit 1
fi
if [ -z "${CARLA_CONTAINER_IMAGE:-}" ] || [ ! -f "${CARLA_CONTAINER_IMAGE}" ]; then
    echo "ERROR: CARLA_CONTAINER_IMAGE in ${TEMPLATE_DIR}/.env must point to an existing SIF." >&2
    exit 1
fi
CARLA_BINARY="${CARLA_BINARY:-/home/carla/CarlaUE4.sh}"
for command_name in lsof setsid; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "ERROR: required command '${command_name}' is not on PATH." >&2
        exit 1
    fi
done

# shellcheck disable=SC1090
source "${VENV}/bin/activate" || exit 1
if ! python -c 'import torch, carla, numpy, cv2, wandb' >/dev/null 2>&1; then
    echo "ERROR: the selected venv is missing an A2C runtime dependency." >&2
    exit 1
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_API_KEY is missing from ${TEMPLATE_DIR}/.env; W&B logging is required." >&2
    exit 1
fi
if ! python -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
    echo "ERROR: PyTorch cannot access the allocated GPU." >&2
    exit 1
fi

if [ -n "${SLURM_JOB_ID:-}" ]; then
    START_PORT=$((10000 + (SLURM_JOB_ID % 200) * 100))
else
    START_PORT=2000
fi
export A2C_SCENARIO="${SCENARIO}"
export CARLA_START_PORT="${START_PORT}"
export CARLA_PORT_STEP=5
export CARLA_SERVER_START_PERIOD=30
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

JOB_ID="${SLURM_JOB_ID:-local}"
RUN_DIR="${SCRIPT_DIR}/runs/basic_a2c_s${SCENARIO}_$(date +%Y%m%d_%H%M%S)_${JOB_ID}"
mkdir -p "${RUN_DIR}" || exit 1
export A2C_RUN_DIR="${RUN_DIR}"
TRAIN_LOG="${RUN_DIR}/a2c_training.log"
CARLA_LOG="${RUN_DIR}/carla_servers.log"
cd "${AV_ROOT}" || exit 1

echo "[A2C] job=${JOB_ID} host=$(hostname) scenario=${SCENARIO} port=${START_PORT}"
echo "[A2C] python=$(command -v python) version=$(python --version 2>&1)"
echo "[A2C] output=${RUN_DIR}"

TRAIN_PID=""
CARLA_PID=""
TAIL_PID=""

alive() {
    [ -n "$1" ] && kill -0 "$1" 2>/dev/null
}

stop_training() {
    if ! alive "${TRAIN_PID}"; then
        return
    fi
    echo "[A2C] requesting graceful stop; current episode will finish and save."
    kill -USR1 "${TRAIN_PID}" 2>/dev/null || true
}

cleanup() {
    local code=$?
    trap '' EXIT INT TERM USR1
    if alive "${TRAIN_PID}"; then
        stop_training
        local waited=0
        while alive "${TRAIN_PID}" && [ "${waited}" -lt 105 ]; do
            sleep 1
            waited=$((waited + 1))
        done
        if alive "${TRAIN_PID}"; then
            echo "[A2C] graceful stop timed out; terminating the training process group." >&2
            kill -TERM -- "-${TRAIN_PID}" 2>/dev/null || true
            sleep 5
            kill -KILL -- "-${TRAIN_PID}" 2>/dev/null || true
        fi
        wait "${TRAIN_PID}" 2>/dev/null || true
    fi
    if alive "${TAIL_PID}"; then
        kill -TERM "${TAIL_PID}" 2>/dev/null || true
        wait "${TAIL_PID}" 2>/dev/null || true
    fi
    if alive "${CARLA_PID}"; then
        echo "[A2C] stopping CARLA supervisor."
        kill -INT "${CARLA_PID}" 2>/dev/null || true
        local waited=0
        while alive "${CARLA_PID}" && [ "${waited}" -lt 15 ]; do
            sleep 1
            waited=$((waited + 1))
        done
        alive "${CARLA_PID}" && kill -TERM "${CARLA_PID}" 2>/dev/null
        wait "${CARLA_PID}" 2>/dev/null || true
    fi
    exit "${code}"
}

trap cleanup EXIT
trap stop_training INT TERM USR1

setsid python -u "${CARLA_LAUNCHER}" \
    --runtime apptainer \
    --num-servers 1 \
    --servers-per-gpu 1 \
    --start-port "${START_PORT}" \
    --port-step 5 \
    --outdir "${RUN_DIR}" \
    --image "${CARLA_CONTAINER_IMAGE}" \
    --binary "${CARLA_BINARY}" \
    > "${CARLA_LOG}" 2>&1 &
CARLA_PID=$!

echo "[A2C] CARLA launcher PID=${CARLA_PID}; waiting for port ${START_PORT}."
server_ready=false
for attempt in $(seq 1 60); do
    if ! alive "${CARLA_PID}"; then
        echo "[A2C] CARLA launcher exited. Last log lines:" >&2
        tail -n 60 "${CARLA_LOG}" >&2
        exit 1
    fi
    if lsof -nP -iTCP:"${START_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
        server_ready=true
        break
    fi
    sleep 10
done
if [ "${server_ready}" != true ]; then
    echo "[A2C] CARLA did not open port ${START_PORT} in time. Last log lines:" >&2
    tail -n 60 "${CARLA_LOG}" >&2
    exit 1
fi

echo "[A2C] CARLA is ready; starting a2c_rgb_try2.py."
setsid python -u "${SCRIPT_DIR}/a2c_rgb_try2.py" > "${TRAIN_LOG}" 2>&1 &
TRAIN_PID=$!
tail -n 0 -f "${TRAIN_LOG}" &
TAIL_PID=$!

while alive "${TRAIN_PID}"; do
    if ! alive "${CARLA_PID}"; then
        echo "[A2C] CARLA launcher died. Last log lines:" >&2
        tail -n 60 "${CARLA_LOG}" >&2
        exit 1
    fi
    sleep 1
done

wait "${TRAIN_PID}"
TRAIN_STATUS=$?
TRAIN_PID=""
exit "${TRAIN_STATUS}"
