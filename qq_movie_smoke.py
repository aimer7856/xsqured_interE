# qq_movie_smoke.py
import os
from QQmodule import Quantum_Bipartite_System

def main():
    # Tiny case for speed
    sys = Quantum_Bipartite_System(
        nx=64,  xmin=-8.0,  xmax=8.0,   mx=4.0,
        x0=2.0, vx0=0.0,
        ny=256, ymin=-12.0, ymax=40.0,  my=0.25,
        y0=0.0, vy0=0.5,
        sigmax=0.5, sigmay=1.5,
        total_time=4.0, timesteps=96,
        lambda_=0.5
    )
    psi_evol = sys.time_evolution_wavefunction()

    outdir = os.environ.get("OUTDIR", "frames_smoke")
    os.makedirs(outdir, exist_ok=True)

    # Write frames of |ψ|^2; save every 4th frame
    sys.save_wave_evolution_frames(psi_evol, outdir=outdir, every=4, vmin=0.0, vmax=None)

if __name__ == "__main__":
    main()