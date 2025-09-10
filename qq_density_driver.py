
#!/usr/bin/env python3
"""
qq_density_driver.py
Stream the wavefunction density |psi(x,y,t)|^2 from QQmodule's bipartite system
and store it efficiently without keeping the whole psi(t) array in RAM.

Usage examples:
  python qq_density_driver.py --param-file param_list_qq_x0_5.txt --mode qq --mx 4 --my 0.25 --outdir outputs
  python qq_density_driver.py --param-file param_list_cc_x0_5.txt --mode qq --mx 4 --my 0.25 --outdir outputs

Notes:
- If the param file does not include 'sigmax', we compute sigmax from the harmonic-oscillator
  ground-state width with ħ=ω=1:  sigmax = 1/sqrt(2*m_x).
- We stream density to a float32 memmap of shape (timesteps, nx, ny) and save metadata as NPZ.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

# Import classes from the local QQmodule.py
from QQmodule import Quantum_Bipartite_System

def load_params_row(param_file, mode, mx, my):
    """
    Return the first CSV row matching (mode, mx, my) with all fields as proper types.
    The CSV is expected to have a header like:
      mode,mx,my,x0,vx0,nx,xmin,xmax,ny,ymin,ymax,y0,vy0,sigmay,total_time,timesteps,lambda,n_eig,filename
    """
    def coerce(k, v):
        if k in {'mode','filename'}:
            return v
        # ints
        if k in {'nx','ny','timesteps','n_eig'}:
            return int(float(v))
        # floats
        return float(v)

    # First pass: try exact mode match
    rows = []
    with open(param_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        rmode = row.get('mode','').strip().lower()
        rmx = float(row.get('mx'))
        rmy = float(row.get('my'))
        if rmode == mode.lower() and abs(rmx - mx) < 1e-12 and abs(rmy - my) < 1e-12:
                return {k: coerce(k, v) for k, v in row.items()}
    # Second pass: ignore mode, match only mx,my
    for row in rows:
        rmx = float(row.get('mx'))
        rmy = float(row.get('my'))
        if abs(rmx - mx) < 1e-12 and abs(rmy - my) < 1e-12:
            return {k: coerce(k, v) for k, v in row.items()}
    raise ValueError(f"No row found for (mode={mode}, mx={mx}, my={my}) in {param_file}")

def compute_sigmax_from_ground_state(mx, hbar=1.0, omega=1.0):
    # σ_x,0 = sqrt(ħ / (2 m ω))
    return math.sqrt(hbar/(2.0*mx*omega))

def build_system(params):
    # Map CSV params to QQmodule initializer signature
    # Quantum_Bipartite_System(nx,xmin,xmax,mx,x0,vx0, ny,ymin,ymax,my,y0,vy0, sigmax,sigmay,total_time,timesteps, lambda_)
    nx = params['nx']
    ny = params['ny']
    xmin, xmax = params['xmin'], params['xmax']
    ymin, ymax = params['ymin'], params['ymax']
    mx, my = params['mx'], params['my']
    x0, y0 = params['x0'], params['y0']
    vx0, vy0 = params['vx0'], params['vy0']
    sigmay = params['sigmay']
    total_time = params['total_time']
    timesteps = params['timesteps']
    lambda_ = params['lambda'] if 'lambda' in params else params['lambda_']

    # If no sigmax in CSV, compute from ground-state width (ħ=ω=1)
    sigmax = params.get('sigmax', None)
    if sigmax is None or (isinstance(sigmax, float) and not np.isfinite(sigmax)):
        sigmax = compute_sigmax_from_ground_state(mx)

    system = Quantum_Bipartite_System(nx, xmin, xmax, mx, x0, vx0,
                                      ny, ymin, ymax, my, y0, vy0,
                                      sigmax, sigmay, total_time, timesteps,
                                      lambda_)
    return system

def stream_density(system, outdir, outprefix):
    """
    Perform CN time-stepping and write |psi|^2 at each step to a float32 memmap
    of shape (T, nx, ny). Also save grids and metadata to an .npz sidecar.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nx, ny = system.nx, system.ny
    T = system.T
    dt = system.dt
    timesteps = system.timesteps
    k = nx * ny

    # Prepare output memmap (float32 to save space)
    mm_path = outdir / f"{outprefix}_rho_txy.f32"
    rho_mm = np.memmap(mm_path, dtype='float32', mode='w+', shape=(timesteps, nx, ny), order='C')

    # Construct CN matrices using QQmodule's public Hamiltonian accessors
    H_sum = sps.kronsum(system.oscillator.get_H(), system.projectile.get_H())
    U_int = sps.kron(system.oscillator.get_U(), system.projectile.get_U(), format='csr')
    A = -1j * (H_sum + system.lambda_ * U_int)
    PNext = sps.identity(k, format='csr') - (dt/2.0) * A
    PPrev = sps.identity(k, format='csr') + (dt/2.0) * A

    # Initial ψ and normalization identical to QQmodule
    psi0 = system.init_wavefunction()
    psi_vec = system.normalized_init_wavefunction(psi0.reshape(k, order='C'))

    # Precompute grids for saving
    x_grid = system.oscillator.grid_points.copy()
    y_grid = system.projectile.grid_points.copy()

    # Write t=0 density
    rho_mm[0, :, :] = (psi_vec.reshape(nx, ny, order='C') * psi_vec.reshape(nx, ny, order='C').conj()).real.astype('float32')

    # March in time, write density each step
    for t in range(1, timesteps):
        rhs = PPrev.dot(psi_vec)
        psi_vec = spla.spsolve(PNext, rhs)

        # Density at step t
        psi_xy = psi_vec.reshape(nx, ny, order='C')
        rho_mm[t, :, :] = (psi_xy * psi_xy.conj()).real.astype('float32')

    # Ensure memmap contents are flushed
    del rho_mm

    # Save metadata and 1D grids
    meta_path = outdir / f"{outprefix}_meta.npz"
    np.savez_compressed(
        meta_path,
        T=T.astype('float64'),
        x_grid=x_grid.astype('float64'),
        y_grid=y_grid.astype('float64'),
        nx=np.array(nx),
        ny=np.array(ny),
        dt=np.array(dt),
        lambda_=np.array(system.lambda_),
        sigmax=np.array(system.sigmax),
        sigmay=np.array(system.sigmay),
    )

    return str(mm_path), str(meta_path)

def main():
    ap = argparse.ArgumentParser(description="Stream |psi|^2 from QQmodule and store to disk")
    ap.add_argument("--param-file", required=True, help="CSV parameter file (qq/cc style header)")
    ap.add_argument("--mode", default="qq", help="mode string to match in CSV (e.g., qq)")
    ap.add_argument("--mx", type=float, required=True, help="oscillator mass mx to match")
    ap.add_argument("--my", type=float, required=True, help="projectile mass my to match")
    ap.add_argument("--outdir", default="outputs", help="directory to write outputs")
    ap.add_argument("--outprefix", default=None, help="output prefix (defaults to filename or qq_mx{mx}_my{my})")
    args = ap.parse_args()

    params = load_params_row(args.param_file, args.mode, args.mx, args.my)
    system = build_system(params)

    outprefix = args.outprefix
    if not outprefix:
        base = params.get('filename', f"{args.mode}_mx{args.mx}_my{args.my}")
        outprefix = f"{base}_density"

    mm_path, meta_path = stream_density(system, args.outdir, outprefix)

    print("Wrote density memmap:", mm_path)
    print("Wrote metadata:", meta_path)

if __name__ == "__main__":
    main()
