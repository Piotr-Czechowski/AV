#!/bin/bash
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=25G
#SBATCH --time=20:50:00
#SBATCH --gpus=1
#SBATCH --account=plgdyplomanci7-gpu-a100
#SBATCH --job-name=new-a3c-carla
#SBATCH --output=new-a3c-carla-log-%J.txt
#SBATCH --error=new-a3c-carla-log-%J.txt
#SBATCH --signal=SIGUSR1@90

# =============================================================================
# new_train.slurm -- full pipeline for AV/A_to_B_GPU_34/new_* implementation.
#
# What it does (one job, one sbatch call):
#   1. Parse named CLI args for every training knob.
#   2. Start N CARLA servers via carla_athena_multiserver_v3.py
#      (one per worker, placed on GPUs via --workers-per-gpu).
#   3. Wait until all N CARLA ports are LISTEN.
#   4. Launch new_train_a3c_carla.py against those servers.
#   5. Tail the training log, clean up both CARLA + training on exit.
#
# Usage:
#   sbatch [SBATCH_FLAGS] new_train.slurm [OPTIONS]
#
# Training / infra args (all optional, have defaults):
#   -w,  --workers NUM                number of A3C workers = CARLA servers   (default: 1)
#        --workers-per-gpu NUM        workers sharing one GPU (CARLA side)    (default: 1)
#        --worker-gpu-start NUM       first CUDA idx for workers              (default: 0)
#        --global-device DEV          device for global net                   (default: cuda:0)
#   -s,  --scenario NUM               CARLA scenario id (space-sep for many)  (default: 14)
#        --camera {rgb,semantic}      camera modality                         (default: semantic)
#        --res NUM                    square image resolution                 (default: 250)
#        --start-port NUM             first CARLA rpc port                    (default: 2000)
#        --port-step NUM              port increment between servers          (default: 100)
#
# Algorithm args:
#        --t-max NUM                  rollout length per update               (default: 20)
#        --gamma FLOAT                discount factor                         (default: 0.99)
#        --lr FLOAT                   starting learning rate                  (default: 1e-4)
#        --beta FLOAT                 entropy coefficient                     (default: 0.01)
#        --value-loss-coef FLOAT      value loss weight                       (default: 1.0)
#        --weight-decay FLOAT         Adam weight decay                       (default: 0.0)
#        --max-grad-norm FLOAT        grad-norm clip, <=0 disables            (default: 40.0)
#        --reward-scale FLOAT         divide rewards by this (0 = disabled)   (default: 0.0)
#        --steps NUM                  global env-step budget                  (default: 10000000)
#        --save-frequency NUM         checkpoint every N steps (0 off)        (default: 20000)
#        --sync-every-n-updates NUM   sync workers every N local updates      (default: 1)
#        --gc-interval NUM            gc/empty_cache every N episodes         (default: 10)
#
# Recovery args:
#        --max-restarts-per-worker NUM                                        (default: 20)
#        --rapid-crash-threshold NUM                                          (default: 3)
#        --rapid-crash-window-steps NUM                                       (default: 100)
#        --carla-timeout-wait FLOAT                                           (default: 60)
#        --carla-server-start-period FLOAT                                    (default: 30)
#        --max-connect-retries NUM                                            (default: 5)
#        --connect-retry-wait FLOAT                                           (default: 30)
#
# Logging args:
#        --diag-log-interval NUM      flush timing every N episodes           (default: 100)
#        --monitor-interval FLOAT     psutil monitor period (s)               (default: 10)
#        --log-steps                  also write steps.jsonl (big)
#        --log-update-arrays          include raw per-step arrays in updates
#        --no-system-monitor          disable RunMonitor/WorkerMonitor
#        --no-gpu-monitor             disable pynvml GPU sampling
#        --wandb-project NAME         W&B project                             (default: a3c-carla)
#        --wandb-run-name NAME        W&B run name                            (default: auto)
#        --no-wandb                   disable W&B subprocess
#        --save-episodes "EP1 EP2..." episodes whose frames get saved to disk (default: none)
#        --save-episode-interval NUM  also save frames every N episodes       (default: 0)
#        --testing                    evaluation mode (argmax, no updates)
#
# Lifecycle args:
#   -r,  --resume DIR                 resume from previous outdir             (default: fresh run)
#        --outdir DIR                 explicit outdir override                (default: auto)
#        --seed NUM                   RNG seed                                (default: 52)
#        --servers-per-gpu NUM        CARLA side only; defaults to --workers-per-gpu
#        --no-carla                   skip launching CARLA (assume up already)
#
# Example: single worker, single GPU, 5M steps, fresh run
#   sbatch new_train.slurm -w 1 --workers-per-gpu 1 --steps 5000000
#
# Example: 4 workers on 2 GPUs, resume from previous
#   sbatch --gpus=2 --mem=50G new_train.slurm -w 4 --workers-per-gpu 2 \
#          -r /net/tscratch/people/plgbartoszkawa/AV/A_to_B_GPU_34/runs/prev_run
# =============================================================================

set -uo pipefail

# ----- defaults --------------------------------------------------------------
NUM_WORKERS=1
WORKERS_PER_GPU=1
WORKER_GPU_START=0
GLOBAL_DEVICE=""
SCENARIO=14
CAMERA=semantic
RES=250
START_PORT=2000
PORT_STEP=100

T_MAX=20
GAMMA=0.99
LR=1e-4
BETA=0.01
VALUE_LOSS_COEF=1.0
WEIGHT_DECAY=1e-2
MAX_GRAD_NORM=40.0
# REWARD_SCALE empty by default → flag is NOT forwarded → no scaling.
# Set explicitly (e.g. --reward-scale 100) to enable.
REWARD_SCALE=""
STEPS=10000000
SAVE_FREQUENCY=20000
SYNC_EVERY_N_UPDATES=1
GC_INTERVAL=10

MAX_RESTARTS_PER_WORKER=160
RAPID_CRASH_THRESHOLD=3
RAPID_CRASH_WINDOW_STEPS=100
CARLA_TIMEOUT_WAIT=60
CARLA_SERVER_START_PERIOD=30
MAX_CONNECT_RETRIES=5
CONNECT_RETRY_WAIT=30

DIAG_LOG_INTERVAL=100
MONITOR_INTERVAL=10
LOG_STEPS=""
LOG_UPDATE_ARRAYS=""
NO_SYSTEM_MONITOR=""
NO_GPU_MONITOR=""
WANDB_PROJECT="a3c-carla"
WANDB_RUN_NAME=""
NO_WANDB=""
SAVE_EPISODES=""
SAVE_EPISODE_INTERVAL=0
TESTING=""

RESUME=""
OUTDIR_OVERRIDE=""
SEED=52
SERVERS_PER_GPU=""
NO_CARLA=""

VENV="/net/tscratch/people/plgbartoszkawa/venv"
PROJECT_DIR="/net/tscratch/people/plgbartoszkawa/AV/A_to_B_GPU_34"
MULTISERVER_SCRIPT="/net/tscratch/people/plgbartoszkawa/carla_athena_multiserver_v3.py"

# ----- parse named args ------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -w|--workers) NUM_WORKERS="$2"; shift 2 ;;
        --workers-per-gpu) WORKERS_PER_GPU="$2"; shift 2 ;;
        --worker-gpu-start) WORKER_GPU_START="$2"; shift 2 ;;
        --global-device) GLOBAL_DEVICE="$2"; shift 2 ;;
        -s|--scenario) SCENARIO="$2"; shift 2 ;;
        --camera) CAMERA="$2"; shift 2 ;;
        --res) RES="$2"; shift 2 ;;
        --start-port) START_PORT="$2"; shift 2 ;;
        --port-step) PORT_STEP="$2"; shift 2 ;;

        --t-max) T_MAX="$2"; shift 2 ;;
        --gamma) GAMMA="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --beta) BETA="$2"; shift 2 ;;
        --value-loss-coef) VALUE_LOSS_COEF="$2"; shift 2 ;;
        --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
        --max-grad-norm) MAX_GRAD_NORM="$2"; shift 2 ;;
        --reward-scale) REWARD_SCALE="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --save-frequency) SAVE_FREQUENCY="$2"; shift 2 ;;
        --sync-every-n-updates) SYNC_EVERY_N_UPDATES="$2"; shift 2 ;;
        --gc-interval) GC_INTERVAL="$2"; shift 2 ;;

        --max-restarts-per-worker) MAX_RESTARTS_PER_WORKER="$2"; shift 2 ;;
        --rapid-crash-threshold) RAPID_CRASH_THRESHOLD="$2"; shift 2 ;;
        --rapid-crash-window-steps) RAPID_CRASH_WINDOW_STEPS="$2"; shift 2 ;;
        --carla-timeout-wait) CARLA_TIMEOUT_WAIT="$2"; shift 2 ;;
        --carla-server-start-period) CARLA_SERVER_START_PERIOD="$2"; shift 2 ;;
        --max-connect-retries) MAX_CONNECT_RETRIES="$2"; shift 2 ;;
        --connect-retry-wait) CONNECT_RETRY_WAIT="$2"; shift 2 ;;

        --diag-log-interval) DIAG_LOG_INTERVAL="$2"; shift 2 ;;
        --monitor-interval) MONITOR_INTERVAL="$2"; shift 2 ;;
        --log-steps) LOG_STEPS="--log-steps"; shift ;;
        --log-update-arrays) LOG_UPDATE_ARRAYS="--log-update-arrays"; shift ;;
        --no-system-monitor) NO_SYSTEM_MONITOR="--no-system-monitor"; shift ;;
        --no-gpu-monitor) NO_GPU_MONITOR="--no-gpu-monitor"; shift ;;
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        --wandb-run-name) WANDB_RUN_NAME="$2"; shift 2 ;;
        --no-wandb) NO_WANDB="--no-wandb"; shift ;;
        --save-episodes|--episodes-to-save) SAVE_EPISODES="$2"; shift 2 ;;
        --save-episode-interval) SAVE_EPISODE_INTERVAL="$2"; shift 2 ;;
        --testing) TESTING="--testing"; shift ;;

        -r|--resume) RESUME="$2"; shift 2 ;;
        --outdir) OUTDIR_OVERRIDE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --servers-per-gpu) SERVERS_PER_GPU="$2"; shift 2 ;;
        --no-carla) NO_CARLA="1"; shift ;;
        --venv) VENV="$2"; shift 2 ;;
        --project-dir) PROJECT_DIR="$2"; shift 2 ;;
        --multiserver-script) MULTISERVER_SCRIPT="$2"; shift 2 ;;

        -h|--help)
            sed -n '14,75p' "$0"; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2; exit 1 ;;
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
export CARLA_SERVER_START_PERIOD

# ----- output dir ------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -n "${RESUME}" ]; then
    OUTPUT_DIR="${RESUME}"
    echo "[SLURM] resuming from: ${OUTPUT_DIR}"
elif [ -n "${OUTDIR_OVERRIDE}" ]; then
    OUTPUT_DIR="${OUTDIR_OVERRIDE}"
    mkdir -p "${OUTPUT_DIR}"
else
    OUTPUT_DIR="${PROJECT_DIR}/runs/a3c_${NUM_WORKERS}w_${TIMESTAMP}_${SLURM_JOB_ID:-local}"
    mkdir -p "${OUTPUT_DIR}"
fi

RUN_LOG="${OUTPUT_DIR}/a3c_training.log"
GPU_LOG="${OUTPUT_DIR}/gpu_dmon.log"
CARLA_LOG="${OUTPUT_DIR}/carla_servers.log"

echo "==================================================================="
echo "=== A3C CARLA training (new_* pipeline) ==="
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
echo "  servers-per-gpu   : ${SERVERS_PER_GPU}"
echo "  start-port        : ${START_PORT} (+${PORT_STEP} per server)"
echo "  scenario / camera : ${SCENARIO} / ${CAMERA}@${RES}x${RES}"
echo "-------------------------------------------------------------------"
echo "  steps             : ${STEPS}"
echo "  t-max / gamma     : ${T_MAX} / ${GAMMA}"
echo "  lr / beta / vloss : ${LR} / ${BETA} / ${VALUE_LOSS_COEF}"
echo "  weight-decay      : ${WEIGHT_DECAY}"
echo "  max-grad-norm     : ${MAX_GRAD_NORM}"
echo "  reward-scale      : ${REWARD_SCALE}"
echo "  save-frequency    : ${SAVE_FREQUENCY}"
echo "  wandb project     : ${WANDB_PROJECT}${NO_WANDB:+ (DISABLED)}"
echo "  resume            : ${RESUME:-<fresh>}"
echo "==================================================================="

cd "${PROJECT_DIR}"

# ----- process bookkeeping + cleanup -----------------------------------------
CARLA_PID=""
GPU_DMON_PID=""
TRAIN_PID=""

cleanup() {
    local code=$?
    echo ""
    echo "[CLEANUP] exit code=${code}; stopping child processes…"
    for pid_var in TRAIN_PID CARLA_PID GPU_DMON_PID; do
        pid="${!pid_var:-}"
        if [ -n "${pid}" ] && ps -p "${pid}" > /dev/null 2>&1; then
            echo "[CLEANUP] killing ${pid_var}=${pid}"
            kill -TERM "${pid}" 2>/dev/null || true
        fi
    done
    sleep 3
    for pid_var in TRAIN_PID CARLA_PID GPU_DMON_PID; do
        pid="${!pid_var:-}"
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
    echo "[SLURM] starting ${NUM_WORKERS} CARLA server(s)…"
    nohup python -u "${MULTISERVER_SCRIPT}" \
        --num-servers "${NUM_WORKERS}" \
        --servers-per-gpu "${SERVERS_PER_GPU}" \
        --start-port "${START_PORT}" \
        --port-step "${PORT_STEP}" \
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
RESUME_FLAG=""
[ -n "${RESUME}" ] && RESUME_FLAG="--resume ${RESUME}"

WANDB_RUN_FLAG=""
[ -n "${WANDB_RUN_NAME}" ] && WANDB_RUN_FLAG="--wandb-run-name ${WANDB_RUN_NAME}"

GLOBAL_DEVICE_FLAG=""
[ -n "${GLOBAL_DEVICE}" ] && GLOBAL_DEVICE_FLAG="--global-device ${GLOBAL_DEVICE}"

SAVE_EPISODES_FLAG=""
[ -n "${SAVE_EPISODES}" ] && SAVE_EPISODES_FLAG="--save-episodes ${SAVE_EPISODES}"

# Only forward --reward-scale when the user actually set REWARD_SCALE
# (empty default ⇒ flag absent ⇒ Python uses its default of 0.0 = off).
REWARD_SCALE_FLAG=""
[ -n "${REWARD_SCALE}" ] && REWARD_SCALE_FLAG="--reward-scale ${REWARD_SCALE}"

# SCENARIO and SAVE_EPISODES are intentionally passed unquoted to let
# space-separated multi-value tokens expand into nargs='+' lists.
# shellcheck disable=SC2086
TRAIN_CMD=(python -u new_train_a3c_carla.py
    --num-workers "${NUM_WORKERS}"
    --workers-per-gpu "${WORKERS_PER_GPU}"
    --worker-gpu-start "${WORKER_GPU_START}"
    --start-port "${START_PORT}"
    --port-step "${PORT_STEP}"
    --scenario ${SCENARIO}
    --camera "${CAMERA}"
    --res "${RES}"
    --t-max "${T_MAX}"
    --gamma "${GAMMA}"
    --lr "${LR}"
    --beta "${BETA}"
    --value-loss-coef "${VALUE_LOSS_COEF}"
    --weight-decay "${WEIGHT_DECAY}"
    --max-grad-norm "${MAX_GRAD_NORM}"
    --steps "${STEPS}"
    --save-frequency "${SAVE_FREQUENCY}"
    --sync-every-n-updates "${SYNC_EVERY_N_UPDATES}"
    --gc-interval "${GC_INTERVAL}"
    --max-restarts-per-worker "${MAX_RESTARTS_PER_WORKER}"
    --rapid-crash-threshold "${RAPID_CRASH_THRESHOLD}"
    --rapid-crash-window-steps "${RAPID_CRASH_WINDOW_STEPS}"
    --carla-timeout-wait "${CARLA_TIMEOUT_WAIT}"
    --carla-server-start-period "${CARLA_SERVER_START_PERIOD}"
    --max-connect-retries "${MAX_CONNECT_RETRIES}"
    --connect-retry-wait "${CONNECT_RETRY_WAIT}"
    --diag-log-interval "${DIAG_LOG_INTERVAL}"
    --monitor-interval "${MONITOR_INTERVAL}"
    --save-episode-interval "${SAVE_EPISODE_INTERVAL}"
    --wandb-project "${WANDB_PROJECT}"
    --seed "${SEED}"
    --outdir "${OUTPUT_DIR}"
    ${GLOBAL_DEVICE_FLAG}
    ${LOG_STEPS}
    ${LOG_UPDATE_ARRAYS}
    ${NO_SYSTEM_MONITOR}
    ${NO_GPU_MONITOR}
    ${NO_WANDB}
    ${TESTING}
    ${WANDB_RUN_FLAG}
    ${SAVE_EPISODES_FLAG}
    ${REWARD_SCALE_FLAG}
    ${RESUME_FLAG}
)

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
