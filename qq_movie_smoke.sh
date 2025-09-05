#!/bin/bash
#SBATCH --job-name=qq_movie_smoke
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-user=doyeon.k@unb.ca
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail

# ---- Clean environment; load a stack that has NumPy + Matplotlib + ffmpeg ----
module --force purge
module load StdEnv/2020

# Scientific Python stack (provides numpy, scipy, matplotlib, etc.)
module load scipy-stack/2020b    # if 2020b not present, try 2020a (see note below)

# FFmpeg under this stack
module load ffmpeg/4.3.2

# (Optional) quick sanity checks
python - <<'PY'
import sys, numpy as np
print("Python:", sys.version.split()[0], "NumPy:", np.__version__)
PY
which ffmpeg
ffmpeg -version | head -n 1

# ---- Stage code from PROJECT to node-local SSD and run from there ----
CODE_DIR=/home/dykim/projects/def-sseahra/dykim/xsqured_interE
WORK_DIR=$SLURM_TMPDIR/xsqured_interE
mkdir -p "$WORK_DIR"
rsync -a "$CODE_DIR/" "$WORK_DIR/"
cd "$WORK_DIR"

# Run the small movie driver (writes $SLURM_TMPDIR/qq_smoke.mp4)
python qq_make_mp4.py

# Copy only the MP4 to SCRATCH
DEST="$SCRATCH/qq_movie_smoke_$SLURM_JOB_ID"
mkdir -p "$DEST"
cp "$SLURM_TMPDIR/qq_smoke.mp4" "$DEST/"
echo "MP4 saved to: $DEST/qq_smoke.mp4"