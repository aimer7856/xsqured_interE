#!/bin/bash
#SBATCH --job-name=qq_movie_smoke
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module load StdEnv/2023
module load python/3.10
# Optional, if you want to stitch on-node (if available on your cluster image):
module load ffmpeg/6.1.1 || true

export OUTDIR="$SLURM_TMPDIR/frames_smoke"
mkdir -p "$OUTDIR"

python qq_movie_smoke.py

# Copy frames to scratch (and stitch)
DEST="$SCRATCH/qq_movie_smoke_$SLURM_JOB_ID"
mkdir -p "$DEST"
rsync -av "$OUTDIR/" "$DEST/"

if command -v ffmpeg &>/dev/null; then
  ffmpeg -r 12 -i "$DEST/frame_%05d.png" -pix_fmt yuv420p "$DEST/qq_smoke.mp4" || true
fi

echo "Smoke movie outputs: $DEST"
