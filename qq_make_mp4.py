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
    qb = Quantum_Bipartite_System(
        nx=256, xmin=-10.0, xmax=10.0, mx=4.0,
        x0=5.0, vx0=0.0,
        ny=4096, ymin=-15.0, ymax=130.0, my=0.25,
        y0=10.0, vy0=-1.0,
        sigmax=0.5, sigmay=3.0,
        total_time=30.0, timesteps=4096,
        lambda_=1.0,
        n_eig=256,
    )

    mp4_dir = os.environ.get("SLURM_TMPDIR", ".")
    mp4_path = os.path.join(mp4_dir, "qq_movie_mx4_my0.25.mp4")

    qb.save_movie_mp4_streaming(
        mp4_path=mp4_path,
        every=8,   # save every 8th step → ~512 frames
        fps=24,
        keep_frames=False,
    )

    print("Wrote:", mp4_path)

if __name__ == "__main__":
    main()