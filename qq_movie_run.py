import os, sys
from pathlib import Path
os.environ.setdefault("MPLBACKEND", "Agg")
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from QQ_wavefunc import Quantum_Bipartite_System
except ModuleNotFoundError:
    from QQmodule import Quantum_Bipartite_System

def main():
    qb = Quantum_Bipartite_System(
        nx=256, xmin=-10.0, xmax=10.0, mx=4.0,
        x0=5.0, vx0=0.0,
        ny=4096, ymin=-15.0, ymax=130.0, my=0.25,
        y0=10.0, vy0=-1.0,
        sigmax=0.35, sigmay=3.0,
        total_time=30.0, timesteps=4096,
        lambda_=1.0
    )
    mp4_dir = os.environ.get("SLURM_TMPDIR", ".")
    mp4_path = os.path.join(mp4_dir, "qq_movie_mx4_my0.25.mp4")
    qb.save_movie_mp4_streaming(mp4_path=mp4_path, every=8, fps=24, keep_frames=False)
    print("Wrote:", mp4_path)

if __name__ == "__main__":
    main()