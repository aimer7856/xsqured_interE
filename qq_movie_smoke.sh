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

# Bring a self-contained ffmpeg into PATH (avoids site module/version issues)
python -m pip install --user -q --upgrade imageio-ffmpeg
export FFMPEG_BIN=$(python - <<'PY'
import imageio_ffmpeg, sys
sys.stdout.write(imageio_ffmpeg.get_ffmpeg_exe())
PY
)
export PATH="$(dirname "$FFMPEG_BIN"):$PATH"

# Optional: verify ffmpeg is available
which ffmpeg
ffmpeg -version | head -n 1 || true


# Stage the code to the compute node
CODE_DIR=/home/dykim/projects/def-sseahra/dykim/xsqured_interE
WORK_DIR=$SLURM_TMPDIR/xsqured_interE
mkdir -p "$WORK_DIR"
rsync -a "$CODE_DIR/" "$WORK_DIR/"
cd "$WORK_DIR"

# Run the Python driver that streams frames and writes the MP4 to $SLURM_TMPDIR
python qq_make_mp4.py

# Stage only the final MP4 to SCRATCH
DEST="$SCRATCH/qq_movie_smoke_$SLURM_JOB_ID"
mkdir -p "$DEST"

# qq_make_mp4.py writes qq_smoke.mp4 to $SLURM_TMPDIR
cp "$SLURM_TMPDIR/qq_smoke.mp4" "$DEST/"

echo "MP4 saved to: $DEST/qq_smoke.mp4"