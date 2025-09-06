#!/bin/bash
#SBATCH --job-name=qq_movie_smoke
#SBATCH --time=00:45:00
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

# ---- Stage only what's required for the smoke run ----
echo "[`date +%F' '%T`] staging code..." >&2
CODE_DIR=/home/dykim/projects/def-sseahra/dykim/xsqured_interE
WORK_DIR=$SLURM_TMPDIR/xsqured_interE
mkdir -p "$WORK_DIR"

# copy only the driver and module (avoid rsync of the whole repo)
cp -f "$CODE_DIR/qq_make_mp4.py" "$WORK_DIR/"
# prefer QQ_wavefunc.py; also bring QQmodule.py as fallback for import
cp -f "$CODE_DIR/QQ_wavefunc.py" "$WORK_DIR/" || true
cp -f "$CODE_DIR/QQmodule.py"     "$WORK_DIR/" || true

cd "$WORK_DIR"
echo "[`date +%F' '%T`] staged to $WORK_DIR" >&2

# ---- Run Python unbuffered so prints flush to .out ----
echo "[`date +%F' '%T`] starting python..." >&2
python -u qq_make_mp4.py
echo "[`date +%F' '%T`] python finished, copying mp4..." >&2

# ---- Copy only the MP4 to SCRATCH ----
DEST="$SCRATCH/qq_movie_smoke_$SLURM_JOB_ID"
mkdir -p "$DEST"
cp "$SLURM_TMPDIR/qq_smoke.mp4" "$DEST/"
echo "[`date +%F' '%T`] MP4 saved to: $DEST/qq_smoke.mp4" >&2