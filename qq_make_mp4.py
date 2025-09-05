import os
import sys
from pathlib import Path

# Ensure headless rendering works on clusters
os.environ.setdefault("MPLBACKEND", "Agg")

# Make sure this script's directory is importable when run via sbatch
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# Prefer QQ_wavefunc.py; fall back to QQmodule.py if needed
try:
    from QQ_wavefunc import Quantum_Bipartite_System  # preferred
except ModuleNotFoundError:
    from QQmodule import Quantum_Bipartite_System


def main():
    # Small, fast configuration (smoke test). Adjust to your production values later.
    qb = Quantum_Bipartite_System(
        nx=64, xmin=-8.0, xmax=8.0, mx=4.0,
        x0=2.0, vx0=0.0,
        ny=256, ymin=-12.0, ymax=40.0, my=0.25,
        y0=0.0, vy0=0.5,
        sigmax=0.5, sigmay=1.5,
        total_time=4.0, timesteps=96,
        lambda_=0.5,
    )

    # Output MP4 path (node-local SSD on SLURM; current dir otherwise)
    mp4_dir = os.environ.get("SLURM_TMPDIR", ".")
    os.makedirs(mp4_dir, exist_ok=True)
    mp4_path = os.path.join(mp4_dir, "qq_smoke.mp4")

    # Stream frames and stitch into MP4 (requires ffmpeg module on CC)
    try:
        qb.save_movie_mp4_streaming(
            mp4_path=mp4_path,
            every=4,   # save every 4th step to reduce I/O
            fps=12,
            keep_frames=False,  # delete PNGs after encoding
        )
    except FileNotFoundError as e:
        # Usually happens if ffmpeg isn't available on the node
        print("Error:", e)
        print("Hint: on CC load ffmpeg: module load ffmpeg/6.1.1")
        sys.exit(1)

    print("Wrote:", mp4_path)


if __name__ == "__main__":
    main()