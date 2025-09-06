#!/bin/bash
#SBATCH --job-name=qq_movie_mx4_my0.25
#SBATCH --time=60:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-user=doyeon.k@unb.ca
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail

module --force purge
module load StdEnv/2020
module load scipy-stack/2020b
module load ffmpeg/4.3.2

echo "[`date +%F' '%T`] staging code..."
CODE_DIR=/home/dykim/projects/def-sseahra/dykim/xsqured_interE
WORK_DIR=$SLURM_TMPDIR/xsqured_interE
mkdir -p "$WORK_DIR"
cp -f "$CODE_DIR/qq_movie_run.py" "$WORK_DIR/"
cp -f "$CODE_DIR/QQ_wavefunc.py" "$WORK_DIR/" || true
cp -f "$CODE_DIR/QQmodule.py"     "$WORK_DIR/" || true
cd "$WORK_DIR"

echo "[`date +%F' '%T`] starting python..."
python -u qq_movie_run.py
echo "[`date +%F' '%T`] python finished, copying mp4..."

DEST="$SCRATCH/qq_movie_mx4_my0.25_$SLURM_JOB_ID"
mkdir -p "$DEST"
cp "$WORK_DIR/qq_movie_mx4_my0.25.mp4" "$DEST/"ß
echo "[`date +%F' '%T`] MP4 saved to: $DEST/qq_movie_mx4_my0.25.mp4"