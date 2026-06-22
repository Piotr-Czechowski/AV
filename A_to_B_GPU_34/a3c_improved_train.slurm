#!/bin/bash
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=25G
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --account=plgdyplomanci7-gpu-a100
#SBATCH --job-name=new-hogwild-a3c-carla
#SBATCH --output=new-hogwild-a3c-carla-log-%J.txt
#SBATCH --error=new-hogwild-a3c-carla-log-%J.txt
#SBATCH --signal=SIGUSR1@90


CONFIG_FILE="$(dirname "${BASH_SOURCE[0]}")/a3c_improved_train_paths.json"
export CONFIG_FILE
if [[ -f "${CONFIG_FILE}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD=python3
    else
        PYTHON_CMD=python
    fi
    eval "$("${PYTHON_CMD}" - <<'PY'
import json, os, shlex
path = os.environ['CONFIG_FILE']
with open(path, 'r', encoding='utf-8') as f:
    cfg = json.load(f)
for key in ('VENV', 'PROJECT_DIR', 'MULTISERVER_SCRIPT', 'WANDB_PROJECT', 'WANDB_ENTITY'):
    if key not in cfg:
        raise SystemExit(f'Missing key {key} in config file {path}')
    if cfg[key] is not None:
        print(f"{key}={shlex.quote(str(cfg[key]))}")
    else:
        print(f"{key}=")
PY
)"
else
    echo "ERROR: missing config file ${CONFIG_FILE}" >&2
    exit 1
fi



set -uo pipefail

# ----- defaults --------------------------------------------------------------
NUM_WORKERS=1
WORKERS_PER_GPU=1
WORKER_GPU_START=0
SERVER_GPU_START=0
SCENARIO=14
START_PORT=2000
PORT_STEP=100

TESTING=""

RESUME=""
OUTDIR_OVERRIDE=""
SEED=""
SERVERS_PER_GPU=""
NO_CARLA=""
PYTHON_EXTRA_ARGS=()
WANDB_NAME=""

# ----- parse named args ------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -w|--workers) NUM_WORKERS="$2"; shift 2 ;;
        --workers-per-gpu) WORKERS_PER_GPU="$2"; shift 2 ;;
        --worker-gpu-start) WORKER_GPU_START="$2"; shift 2 ;;
        --server-gpu-start) SERVER_GPU_START="$2"; shift 2 ;;
        --servers-per-gpu) SERVERS_PER_GPU="$2"; shift 2 ;;
        -s|--scenario) SCENARIO="$2"; shift 2 ;;
        --start-port) START_PORT="$2"; shift 2 ;;
        --port-step) PORT_STEP="$2"; shift 2 ;;
        --testing) TESTING="--testing"; shift ;;
        --wandb-name) WANDB_NAME="$2"; shift 2;;
        -r|--resume) RESUME="$2"; shift 2 ;;
        --outdir) OUTDIR_OVERRIDE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --no-carla) NO_CARLA="1"; shift ;;
        --venv) VENV="$2"; shift 2 ;;
        --project-dir) PROJECT_DIR="$2"; shift 2 ;;
        --multiserver-script) MULTISERVER_SCRIPT="$2"; shift 2 ;;
        --python-arg) PYTHON_EXTRA_ARGS+=("$2"); shift 2 ;;
        --) shift; PYTHON_EXTRA_ARGS+=("$@"); break ;;

        -h|--help)
            sed -n '14,55p' "$0"; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Use -- before rare Python-only arguments, e.g. -- --lr 5e-5" >&2
            exit 1 ;;
    esac
done

if [ -z "${SERVERS_PER_GPU}" ]; then
    SERVERS_PER_GPU="${WORKERS_PER_GPU}"
fi

# ----- HPC environment -------------------------------------------------------
if [ -n "${TMPDIR:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    ln -sfn "$TMPDIR" "/tmp/${SLURM_JOB_ID}" || true
    export TMPDIR="/tmp/${SLURM_JOB_ID}"
fi

module load GCCcore/13.2.0 Python/3.11.5 2>/dev/null || true
module load CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0 2>/dev/null || true

if [ -d "${VENV}" ]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ----- output dir ------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -n "${RESUME}" ]; then
    OUTPUT_DIR="${RESUME}"
    echo "[SLURM] resuming from: ${OUTPUT_DIR}"
elif [ -n "${OUTDIR_OVERRIDE}" ]; then
    OUTPUT_DIR="$(pwd)/${OUTDIR_OVERRIDE}"
    mkdir -p "${OUTPUT_DIR}"
else
    OUTPUT_DIR="${PROJECT_DIR}/runs/a3c_improved_1_${NUM_WORKERS}w_${TIMESTAMP}_${SLURM_JOB_ID:-local}"
    mkdir -p "${OUTPUT_DIR}"
fi
# ----- wandb-file ------------------------------------------------------------
if [ -z "${WANDB_NAME}" ]; then
    WANDB_NAME="a3c_improved_1_${NUM_WORKERS}w_${TIMESTAMP}"
fi

RUN_LOG="${OUTPUT_DIR}/a3c_training.log"
GPU_LOG="${OUTPUT_DIR}/gpu_dmon.log"
CARLA_LOG="${OUTPUT_DIR}/carla_servers.log"

echo "==================================================================="
echo "=== A3C CARLA training (a3c_improved_* pipeline) ==="
echo "==================================================================="
echo "  job id            : ${SLURM_JOB_ID:-local}"
echo "  host              : $(hostname)"
echo "  visible GPUs      : ${CUDA_VISIBLE_DEVICES:-all}"
echo "  outdir            : ${OUTPUT_DIR}"
echo "  project dir       : ${PROJECT_DIR}"
echo "  venv              : ${VENV}"
echo "-------------------------------------------------------------------"
echo "  workers           : ${NUM_WORKERS}"
echo "  workers-per-gpu   : ${WORKERS_PER_GPU}"
echo "  worker-gpu-start  : ${WORKER_GPU_START}"
echo "  server-gpu-start  : ${SERVER_GPU_START}"
echo "  servers-per-gpu   : ${SERVERS_PER_GPU}"
echo "  start-port        : ${START_PORT} (+${PORT_STEP} per server)"
echo "  scenario          : ${SCENARIO}"
echo "-------------------------------------------------------------------"
echo "  steps override    : ${STEPS:-<python default>}"
echo "  wandb             : enabled"
echo "  resume            : ${RESUME:-<fresh>}"
if [ "${#PYTHON_EXTRA_ARGS[@]}" -gt 0 ]; then
    printf '  python extra args :'
    printf ' %q' "${PYTHON_EXTRA_ARGS[@]}"
    echo ""
fi
echo "==================================================================="

cd "${PROJECT_DIR}"

# ----- process bookkeeping + cleanup -----------------------------------------
CARLA_PID=""
GPU_DMON_PID=""
TRAIN_PID=""
TAIL_PID=""

cleanup() {
    local code=$?
    trap - SIGINT SIGTERM SIGUSR1 EXIT
    echo ""
    echo "[CLEANUP] exit code=${code}; stopping child processes..."

    local train_pid="${TRAIN_PID:-}"

    for pid_var in TAIL_PID CARLA_PID GPU_DMON_PID; do
        local pid="${!pid_var:-}"
        if [ -n "${pid}" ] && ps -p "${pid}" > /dev/null 2>&1; then
            echo "[CLEANUP] SIGTERM ${pid_var}=${pid}"
            kill -TERM "${pid}" 2>/dev/null || true
        fi
    done

    if [ -n "${train_pid}" ] && ps -p "${train_pid}" > /dev/null 2>&1; then
        echo "[CLEANUP] SIGKILL TRAIN_PID=${train_pid}"
        kill -KILL "${train_pid}" 2>/dev/null || true
    fi

    sleep 3
    for pid_var in TAIL_PID CARLA_PID GPU_DMON_PID; do
        local pid="${!pid_var:-}"
        if [ -n "${pid}" ] && ps -p "${pid}" > /dev/null 2>&1; then
            echo "[CLEANUP] SIGKILL ${pid_var}=${pid}"
            kill -KILL "${pid}" 2>/dev/null || true
        fi
    done
    pkill -TERM -f CarlaUE4 2>/dev/null || true
    echo "[CLEANUP] done."
    exit "${code}"
}
trap cleanup SIGINT SIGTERM SIGUSR1 EXIT

# ----- start GPU watchdog (for post-mortem) ----------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi dmon -s pucvmet -o DT > "${GPU_LOG}" 2>&1 &
    GPU_DMON_PID=$!
    echo "[SLURM] gpu watchdog PID=${GPU_DMON_PID} -> ${GPU_LOG}"
    nvidia-smi || true
fi

# ----- start CARLA servers ---------------------------------------------------
if [ -z "${NO_CARLA}" ]; then
    echo "[SLURM] starting ${NUM_WORKERS} CARLA server(s)..."
    nohup python -u "${MULTISERVER_SCRIPT}" \
        --num-servers "${NUM_WORKERS}" \
        --servers-per-gpu "${SERVERS_PER_GPU}" \
        --server-gpu-start "${SERVER_GPU_START}" \
        --start-port "${START_PORT}" \
        --port-step "${PORT_STEP}" \
        --outdir "${OUTPUT_DIR}" \
        > "${CARLA_LOG}" 2>&1 &
    CARLA_PID=$!
    echo "[SLURM] CARLA supervisor PID=${CARLA_PID} -> ${CARLA_LOG}"

    EXPECTED_PORTS=()
    for ((i=0; i<NUM_WORKERS; i++)); do
        EXPECTED_PORTS+=($((START_PORT + i * PORT_STEP)))
    done
    echo "[SLURM] waiting for ports: ${EXPECTED_PORTS[*]}"

    retries=60
    delay=10
    for ((r=1; r<=retries; r++)); do
        all_ok=true
        for port in "${EXPECTED_PORTS[@]}"; do
            if ! lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
                all_ok=false; break
            fi
        done
        if ${all_ok}; then
            echo "[SLURM] all CARLA ports LISTEN after ${r} check(s)."
            break
        fi
        if [ "${r}" -eq "${retries}" ]; then
            echo "[SLURM] FAILED to bring up all CARLA servers in time." >&2
            echo "[SLURM] last 80 lines of ${CARLA_LOG}:" >&2
            tail -n 80 "${CARLA_LOG}" >&2 || true
            exit 1
        fi
        sleep "${delay}"
    done
else
    echo "[SLURM] --no-carla: skipping CARLA startup."
fi

# ----- build training command ------------------------------------------------
TRAIN_CMD=(python -u a3c_improved_1.py
    --num-workers "${NUM_WORKERS}"
    --workers-per-gpu "${WORKERS_PER_GPU}"
    --worker-gpu-start "${WORKER_GPU_START}"
    --start-port "${START_PORT}"
    --port-step "${PORT_STEP}"
    --outdir "${OUTPUT_DIR}"
    --wandb-name "${WANDB_NAME}"
    --scenario "${SCENARIO}"
)

[ -n "${SEED}" ] && TRAIN_CMD+=(--seed "${SEED}")
[ -n "${RESUME}" ] && TRAIN_CMD+=(--resume "${RESUME}")
[ -n "${TESTING}" ] && TRAIN_CMD+=(--testing)

if [ "${#PYTHON_EXTRA_ARGS[@]}" -gt 0 ]; then
    TRAIN_CMD+=("${PYTHON_EXTRA_ARGS[@]}")
fi

echo "[SLURM] training command:"
printf '  %q ' "${TRAIN_CMD[@]}"; echo ""
echo ""

touch "${RUN_LOG}"
"${TRAIN_CMD[@]}" >> "${RUN_LOG}" 2>&1 &
TRAIN_PID=$!
echo "[SLURM] training PID=${TRAIN_PID} -> ${RUN_LOG}"
echo ""
echo "==================================================================="
echo "=== tailing training log (Ctrl+C stops training AND this job) ==="
echo "==================================================================="
tail -n +1 -f "${RUN_LOG}" --pid=${TRAIN_PID} &
TAIL_PID=$!

wait "${TRAIN_PID}"
TRAIN_RC=$?
echo ""
echo "[SLURM] training exited with code ${TRAIN_RC}"

kill -TERM "${TAIL_PID}" 2>/dev/null || true

exit "${TRAIN_RC}"