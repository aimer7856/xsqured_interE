#!/bin/bash
#SBATCH --job-name=cc_x2
#SBATCH --array=0-9                     # <-- adjust this based on line count - 2
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs_cc/%x_%A_%a.out
#SBATCH --error=logs_cc/%x_%A_%a.err
#SBATCH --mail-user=doyeon.k@unb.ca
#SBATCH --mail-type=ALL

mkdir -p logs_cc

# Conda activation
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py310_env

unset PYTHONPATH
export PYTHONNOUSERSITE=True

# Read parameter line
PARAM_FILE="param_list_cc.txt"
# If param_list_qq.txt has a header, skip it; otherwise, remove 'tail -n +2'
LINE=$(tail -n +2 "$PARAM_FILE" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")

IFS=',' read -r MODE MX MY X0 VX0 NX XMIN XMAX NY YMIN YMAX Y0 VY0 SIGMAY TOTAL_TIME TIMESTEPS LAMBDA N_EIG FILENAME <<< "$LINE"

OUT_DIR="results_x2/${MODE}/${FILENAME}"
mkdir -p "$OUT_DIR"

export MEM_LOG_FILE="${OUT_DIR}/mem_log.txt"

RUN_LOG="${OUT_DIR}/run.log"
START=$(date +%s)

python RunSimulation_x2.py \
  --mode "$MODE" --mx "$MX" --my "$MY" \
  --x0 "$X0" --vx0 "$VX0" \ 
  --nx "$NX" --xmin "$XMIN" --xmax "$XMAX" \
  --ny "$NY" --ymin "$YMIN" --ymax "$YMAX" \
  --y0 "$Y0" --vy0 "$VY0" --sigmay "$SIGMAY" \
  --total_time "$TOTAL_TIME" --timesteps "$TIMESTEPS" \
  --lambda_ "$LAMBDA" --N_eig "$N_EIG" \
  --base "$FILENAME" --output_dir "$OUT_DIR" \
  2>&1 | tee "$RUN_LOG"

END=$(date +%s)
echo "Finished at: $(date)"          | tee -a "$RUN_LOG"
echo "Elapsed time: $((END-START))s" | tee -a "$RUN_LOG"

# Save metadata
cat > "${OUT_DIR}/params.txt" << EOF
base=$FILENAME
mode=$MODE
mx=$MX
my=$MY
x0=$X0
vx0=$VX0
xmin=$XMIN
xmax=$XMAX
y0=$Y0
vy0=$VY0
ymin=$YMIN
ymax=$YMAX
nx=$NX
ny=$NY
sigmay=$SIGMAY
timesteps=$TIMESTEPS
total_time=$TOTAL_TIME
lambda=$LAMBDA
N_eig=$N_EIG
output_dir=$OUT_DIR
EOF

touch "${OUT_DIR}/done.txt"