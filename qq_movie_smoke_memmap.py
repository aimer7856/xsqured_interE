# qq_movie_smoke_memmap.py
import os
import numpy as np
from QQmodule import Quantum_Bipartite_System

def main():
    # Small grid for a quick test; you can scale this up later
    sys = Quantum_Bipartite_System(
        nx=64,  xmin=-8.0,  xmax=8.0,   mx=4.0,
        x0=2.0, vx0=0.0,
        ny=256, ymin=-12.0, ymax=40.0,  my=0.25,
        y0=0.0, vy0=0.5,
        sigmax=0.5, sigmay=1.5,
        total_time=4.0, timesteps=96,
        lambda_=0.5
    )

    # Use node-local SSD on Narval if available
    tmpdir = os.environ.get("SLURM_TMPDIR", ".")
    k = sys.nx * sys.ny
    shape = (sys.timesteps, k)

    # Disk-backed storage (halve footprint with complex64)
    mmap_path = os.path.join(tmpdir, "psi_evol.dat")
    psi_evol = np.memmap(mmap_path, dtype=np.complex64, mode="w+", shape=shape)

    # Evolve directly into the memmap
    sys.time_evolution_wavefunction(out=psi_evol, dtype=np.complex64)

    # Write frames (reads from memmap lazily; low RAM)
    outdir = os.environ.get("OUTDIR", "frames_smoke")
    os.makedirs(outdir, exist_ok=True)
    sys.save_wave_evolution_frames(psi_evol, outdir=outdir, every=4, vmin=0.0, vmax=None)

if __name__ == "__main__":
    main()