#!/usr/bin/env python3
"""
qq_density_driver_fixedparams.py
Build Quantum_Bipartite_System directly (no CSV), evolve with Crank–Nicolson,
and stream the probability density |psi|^2 to a memory-mapped file.

Default parameters are:
    qb = Quantum_Bipartite_System(
        nx=256, xmin=-10.0, xmax=10.0, mx=4.0,
        x0=5.0, vx0=0.0,
        ny=4096, ymin=-15.0, ymax=130.0, my=0.25,
        y0=10.0, vy0=-1.0,
        sigmax=0.35, sigmay=3.0,
        total_time=30.0, timesteps=4096,
        lambda_=1.0
    )

You can override any parameter via CLI flags.
Example (keep defaults, only customize outdir/prefix):
  python qq_density_driver_fixedparams.py --outdir outputs --outprefix mx4_my0p25

Outputs:
  <outdir>/<outprefix>_rho_txy.f32    # float32 memmap [T, nx, ny] of |psi|^2
  <outdir>/<outprefix>_meta.npz       # grids & metadata

Notes:
- This driver reconstructs the same CN system used in the QQ module by
  assembling H = Hx ⊕ Hy and the interaction λ Ux ⊗ Uy. If your QQmodule
  uses different accessors for H and U, adjust the two calls inside
  `build_CN_operators()` accordingly (search for "ADAPT HERE").
"""
import argparse
from pathlib import Path
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from QQmodule import Quantum_Bipartite_System

def build_system_from_args(args):
    qb = Quantum_Bipartite_System(
        nx=args.nx, xmin=args.xmin, xmax=args.xmax, mx=args.mx,
        x0=args.x0, vx0=args.vx0,
        ny=args.ny, ymin=args.ymin, ymax=args.ymax, my=args.my,
        y0=args.y0, vy0=args.vy0,
        sigmax=args.sigmax, sigmay=args.sigmay,
        total_time=args.total_time, timesteps=args.timesteps,
        lambda_=args.lambda_
    )
    return qb

def build_CN_operators(qb):
    """
    Construct sparse operators for CN:
      A = -i (H_sum + lambda * U_int)
      PNext = I - (dt/2) A
      PPrev = I + (dt/2) A

    ADAPT HERE if your QQmodule exposes H and U differently.
    """
    # Attempt 1: method accessors
    try:
        Hx = qb.oscillator.get_H()
        Hy = qb.projectile.get_H()
    except AttributeError:
        # Attempt 2: attributes
        Hx = getattr(qb.oscillator, 'H')
        Hy = getattr(qb.projectile, 'H')

    # Direct sum
    H_sum = sps.kronsum(Hx, Hy, format='csr')

    # Interaction operators Ux ⊗ Uy
    try:
        Ux = qb.oscillator.get_U()
        Uy = qb.projectile.get_U()
    except AttributeError:
        Ux = getattr(qb.oscillator, 'U')
        Uy = getattr(qb.projectile, 'U')

    U_int = sps.kron(Ux, Uy, format='csr')

    A = -1j * (H_sum + qb.lambda_ * U_int)

    k = qb.nx * qb.ny
    I = sps.identity(k, format='csr', dtype=complex)
    PNext = I - (qb.dt/2.0) * A
    PPrev = I + (qb.dt/2.0) * A
    return PNext, PPrev

def init_psi_vec(qb):
    # Prefer QQmodule's own initialization if available
    try:
        psi0 = qb.init_wavefunction()
        psi_vec = qb.normalized_init_wavefunction(psi0.reshape(qb.nx*qb.ny, order='C'))
        return psi_vec
    except AttributeError:
        # Fallback: assume qb.psi0 exists, normalize
        psi0 = getattr(qb, 'psi0').reshape(qb.nx*qb.ny, order='C')
        norm = np.sqrt((psi0.conj()*psi0).sum())
        return psi0 / norm

def stream_density(qb, outdir, outprefix):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nx, ny = qb.nx, qb.ny
    T = qb.T
    dt = qb.dt
    timesteps = qb.timesteps
    k = nx * ny

    # Output memmap
    mm_path = outdir / f"{outprefix}_rho_txy.f32"
    rho_mm = np.memmap(mm_path, dtype='float32', mode='w+', shape=(timesteps, nx, ny), order='C')

    # CN operators
    PNext, PPrev = build_CN_operators(qb)

    # Initial psi
    psi_vec = init_psi_vec(qb)

    # grids (1D)
    x_grid = qb.oscillator.grid_points.copy()
    y_grid = qb.projectile.grid_points.copy()

    # t=0
    psi_xy = psi_vec.reshape(nx, ny, order='C')
    rho_mm[0, :, :] = (psi_xy * psi_xy.conj()).real.astype('float32')

    # March
    for t in range(1, timesteps):
        rhs = PPrev.dot(psi_vec)
        psi_vec = spla.spsolve(PNext, rhs)
        psi_xy = psi_vec.reshape(nx, ny, order='C')
        rho_mm[t, :, :] = (psi_xy * psi_xy.conj()).real.astype('float32')

    # flush
    del rho_mm

    # Save metadata
    meta_path = outdir / f"{outprefix}_meta.npz"
    np.savez_compressed(
        meta_path,
        T=T.astype('float64'),
        x_grid=x_grid.astype('float64'),
        y_grid=y_grid.astype('float64'),
        nx=np.array(nx),
        ny=np.array(ny),
        dt=np.array(dt),
        lambda_=np.array(qb.lambda_),
        sigmax=np.array(qb.sigmax),
        sigmay=np.array(qb.sigmay),
    )
    return str(mm_path), str(meta_path)

def build_argparser():
    ap = argparse.ArgumentParser(description="Stream |psi|^2 from QQmodule (fixed parameters by default)")
    # Defaults match your requested qb=... block
    ap.add_argument('--nx', type=int, default=256)
    ap.add_argument('--xmin', type=float, default=-10.0)
    ap.add_argument('--xmax', type=float, default=10.0)
    ap.add_argument('--mx', type=float, default=4.0)
    ap.add_argument('--x0', type=float, default=5.0)
    ap.add_argument('--vx0', type=float, default=0.0)

    ap.add_argument('--ny', type=int, default=4096)
    ap.add_argument('--ymin', type=float, default=-15.0)
    ap.add_argument('--ymax', type=float, default=130.0)
    ap.add_argument('--my', type=float, default=0.25)
    ap.add_argument('--y0', type=float, default=10.0)
    ap.add_argument('--vy0', type=float, default=-1.0)

    ap.add_argument('--sigmax', type=float, default=0.35)
    ap.add_argument('--sigmay', type=float, default=3.0)

    ap.add_argument('--total_time', type=float, default=30.0)
    ap.add_argument('--timesteps', type=int, default=4096)
    ap.add_argument('--lambda_', type=float, default=1.0)

    ap.add_argument('--outdir', type=str, default='outputs')
    ap.add_argument('--outprefix', type=str, default='mx4_my0p25_density')
    return ap

def main():
    args = build_argparser().parse_args()
    qb = build_system_from_args(args)
    mm_path, meta_path = stream_density(qb, args.outdir, args.outprefix)
    print("Wrote density memmap:", mm_path)
    print("Wrote metadata:", meta_path)

if __name__ == '__main__':
    main()
