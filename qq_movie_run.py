"""
Driver script for full QQ wavefunction evolution movie.
Adjust nx, ny, timesteps, coupling, etc. as needed.
"""

from QQ_wavefunc import Quantum_Bipartite_System

if __name__ == "__main__":
    # --- Simulation parameters ---
    nx = 256
    ny = 4096
    total_time = 40.0
    timesteps = 4096
    lambda_coupling = 1.0   # adjust for your interaction
    every = 8               # save every 8th step
    fps = 24                # smooth movie playback
    mp4_path = "qq_movie.mp4"

    # --- Build system ---
    qb = Quantum_Bipartite_System(
        nx=nx,
        ny=ny,
        total_time=total_time,
        timesteps=timesteps,
        lambda_=lambda_coupling,
        # also pass x0, y0, px0, py0 if you want displaced/coherent initial state
    )

    # --- Run and save MP4 ---
    qb.save_movie_mp4_streaming(
        mp4_path=mp4_path,
        every=every,
        fps=fps,
        keep_frames=False,
    )

    print(f"Wrote movie to {mp4_path}")