"""
Profiled batch runner for three simulation types:
 - full quantum bipartite (QuantumSimulationModules)
 - classical–quantum mean-field (ClassicalQuantumEhrenfest)
 - purely classical RK45 (ClassicalSimulationModules)

Use --mode to pick one or all.
"""


from memory_profiler import profile

import os
import time
import psutil

import argparse
import time
from pathlib import Path
import numpy as np
import json
import cProfile
import pstats

# --- Simulation Modules ----------------------------------------------------

# Full quantum simulation
from QQmodule import Quantum_Bipartite_System
# Classical–Quantum mean-field (Ehrenfest)
from CQmodule import CQEhrenfestSystem
# Purely classical simulation
from CCmodule import ClassicalBipartiteSystem



# --- Helpers ---------------------------------------------------------------
START_TIME = time.time()

def log_peak_memory(note="", filename=None):
    """
    Log current peak memory usage and elapsed time to a file.
    Uses $MEM_LOG_FILE or falls back to 'func_mem_log.txt'.
    """
    if filename is None:
        filename = os.environ.get("MEM_LOG_FILE", "func_mem_log.txt")

    # Get memory in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024

    # Elapsed time since script started
    timestamp = time.time()
    elapsed = timestamp - START_TIME

    # Create header if file is new
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        with open(filename, "a") as f:
            f.write("timestamp,elapsed_seconds,note,memory_MB\n")

    # Append log line
    with open(filename, "a") as f:
        f.write(f"{timestamp:.2f},{elapsed:.2f},{note},{mem:.2f}\n")

def format_hms_str(total_seconds: float) -> str:
    hrs = int(total_seconds // 3600)
    mins = int((total_seconds % 3600) // 60)
    secs = total_seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"


def profile_and_dump(fn, name, *args, **kwargs):
    t0 = time.perf_counter()
    profiler = cProfile.Profile()
    profiler.enable()
    result = fn(*args, **kwargs)
    profiler.disable()
    runtime = time.perf_counter() - t0
    
   # Correct way to access 'params'
    params = kwargs.get('params')
    if not params and len(args) > 0 and isinstance(args[0], dict):
        params = args[0]
        
    output_dir = Path(params.get('output_dir', '.'))
    base = params.get('base', name)
    
    prof_file = output_dir / f"{base}.prof"
    profiler.dump_stats(prof_file)
    print(f"[PROFILE] Saved raw profile to {prof_file}")
    
    txt_file = output_dir /f"{base}_profile.txt"
    with open(txt_file, 'w') as rpt:
        stats = pstats.Stats(profiler, stream=rpt)
        stats.strip_dirs().sort_stats('cumtime').print_stats(50)
    print(f"[PROFILE] Saved text report to {txt_file}")

    return result, runtime

# --- Simulation Routines ---------------------------------------------------

@profile
def run_quantum(params: dict):
    """
    Execute the full quantum bipartite simulation.
    Returns (T, qdata_quantum, ext_quantum).
    """
    q_params = {
        'nx'        : params['nx'],
        'xmin'      : params['xmin'],
        'xmax'      : params['xmax'],
        'mx'        : params['mx'],
        'x0'        : params['x0'],
        'vx0'       : params['vx0'],
        'ny'        : params['ny'],
        'ymin'      : params['ymin'],
        'ymax'      : params['ymax'],
        'my'        : params['my'],
        'y0'        : params['y0'],
        'vy0'       : params['vy0'],
        'sigmax'    : params['sigmax'],
        'sigmay'    : params['sigmay'],
        'total_time': params['total_time'],
        'timesteps' : params['timesteps'],
        'lambda_'   : params['lambda_'],
    }
    log_peak_memory(note="before QQ system init")
    qsys = Quantum_Bipartite_System(**q_params)
    log_peak_memory(note="after QQ system init")
    T, qdata = qsys.get_expected_data(include=[
        'oscillator','projectile','vn_entropy',
        'inter_energy','std_inter_energy','linear_entropy','rho1_diag'
    ])
    log_peak_memory("📊 after get_expected_data()")
    # update metadata from the actual run
    q_params['total_time'] = T[-1]
    q_params['timesteps']  = len(T)
    
    return T, qdata, q_params


@profile
def run_cq(params: dict):
    """
    Run the Classical–Quantum Ehrenfest solver.
    Returns (T_cq, qdata_cq, ext_cq).
    """
    cq_params = {
        'N_eig'     : params['N_eig'],
        'xmin'      : params['xmin'],
        'xmax'      : params['xmax'],
        'mx'        : params['mx'],
        'x0'        : params['x0'],
        'vx0'       : params['vx0'],
        'ymin'      : params['ymin'],
        'ymax'      : params['ymax'],
        'my'        : params['my'],
        'y0'        : params['y0'],
        'vy0'       : params['vy0'],
        'sigmax'    : params['sigmax'],
        'total_time': params['total_time'],
        'timesteps' : params['timesteps'],
        'lambda_'   : params['lambda_'],
    }
    log_peak_memory(note="before CQ system init")
    cq = CQEhrenfestSystem(**cq_params)
    log_peak_memory(note="after CQ system init")
    output = cq.solve()
    log_peak_memory(note="after cq.solve")
    t = output.pop('t')
    cqdata = output
    return t, cqdata, cq_params

@profile
def run_classical(params: dict):
    """
    Run the purely classical RK45 simulation.
    Returns (T_cl, cdata).
    """
    c_params = {
        'x0'        : params['x0'],
        'y0'        : params['y0'],
        'vx0'       : params['vx0'],
        'vy0'       : params['vy0'],
        'mx'        : params['mx'],
        'my'        : params['my'],
        'lambda_'   : params['lambda_'],
        'total_time': params['total_time'],
        'timesteps' : params['timesteps'],
    }
    log_peak_memory(note="before CC system init")
    csys = ClassicalBipartiteSystem(**c_params)
    log_peak_memory(note="after CC system init")
    t_cl, x_c, px_c, y_c, py_c, Hx_c, Hy_c, Hint_c, Htot_c = csys.simulate_rk45_refined()
    cdata = {
        'x'   : x_c,
        'px'  : px_c,
        'y'   : y_c,
        'py'  : py_c,
        'Hx'  : Hx_c,
        'Hy'  : Hy_c,
        'Hint': Hint_c,
        'Htot': Htot_c
    }
    return t_cl, cdata,c_params

# --- Argument Parsing ------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profiled runner for quantum, CQ, and classical sims"
    )
    parser.add_argument(
        '--mode', choices=['qq','cq','cc','all'], default='all',
        help="Which simulation(s) to run"
    )
    
    parser.add_argument("--N_eig",       type=int,   default=16,  help="# Eigenbasis")
   # Spatial grid parameters
    parser.add_argument("--nx",       type=int,   default=256,  help="# x-grid points")
    parser.add_argument("--xmin",     type=float, default=-7.5, help="Min x")
    parser.add_argument("--xmax",     type=float, default=7.5,  help="Max x")
    parser.add_argument("--ny",       type=int,   default=2048, help="# y-grid points")
    parser.add_argument("--ymin",     type=float, default=-5.0, help="Min y")
    parser.add_argument("--ymax",     type=float, default=175.0,help="Max y")

    # Physical parameters
    parser.add_argument("--mx",       type=float, default=2.0,  help="Mass of oscillator")
    parser.add_argument("--my",       type=float, default=1.0,  help="Mass of projectile")
    parser.add_argument("--x0",       type=float, default=0.0,  help="Initial x-center")
    parser.add_argument("--y0",       type=float, default=20.0, help="Initial y-center")
    parser.add_argument("--vx0",       type=float, default=0.0,  help="Initial x-velocity")
    parser.add_argument("--vy0",       type=float, default=-5.0, help="Initial y-velocity")
    parser.add_argument("--sigmax",   type=float, default=None, help="Initial x-width (computed if None)")
    parser.add_argument("--sigmay",   type=float, default=3.0, help="Initial x-width (computed if None)")

    # Time evolution parameters
    parser.add_argument("--total_time", type=float, default=30.0, help="Total sim time")
    parser.add_argument("--timesteps",  type=int,   default=1024, help="# time steps")
    parser.add_argument("--lambda_",    type=float, default=1.0,  help="Interaction strength λ")

    # Options
    parser.add_argument("--base",      type=str, default=None, help="Base name for output files")
    parser.add_argument("--output_dir",type=str, default="results", help="Output directory")

    return parser.parse_args()

# --- Main ------------------------------------------------------------------

@profile
def main():
    args   = parse_args()
    params = vars(args)

    # Default packet width
    if params['sigmax'] is None:
       params['sigmax'] = float(1.0 / np.sqrt(2.0 * params['mx']))
    #print(params['sigmax'] )
    
    out_dir = Path(params['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    base    = args.base
    metadata = {}

    modes = [args.mode] if args.mode != 'all' else ['quantum','cq','classical']
    
    # inside RunSimulation… before you create the Quantum_Bipartite_System
   # import QuantumSimulationModules_mx_my_memlog as qmod
   # print("USING:", qmod.__file__)
   # print("contains renorm?", 'vec /= nrm' in open(qmod.__file__).read())

    # Full quantum
    if 'qq' in modes:
        (T_q, qdata_q, ext_q), rt_q = profile_and_dump(run_quantum, 'qq', params)
        ext_q['runtime_qq'] = format_hms_str(rt_q)
        metadata['qq']      = ext_q
        np.savez(out_dir/f"{base}.npz", t=T_q, **qdata_q)

    # Classical-Quantum mean-field
    if 'cq' in modes:
        (T_cq, qdata_cq, ext_cq), rt_cq = profile_and_dump(run_cq, 'cq', params)
        ext_cq['runtime_cq'] = format_hms_str(rt_cq)
        metadata['cq']       = ext_cq
        np.savez(out_dir/f"{base}.npz", t=T_cq, **qdata_cq)

    # Purely classical
    if 'cc' in modes:
        (T_cl, cdata, ext_c), rt_cl = profile_and_dump(run_classical, 'cc', params)
        ext_c['runtime_cc'] = format_hms_str(rt_cl)
        metadata['cc']      = ext_c
        np.savez(out_dir/f"{base}.npz", t=T_cl, **cdata)

    # Write metadata JSON
    with open(out_dir/f"{base}_params.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[DONE] mode={args.mode}, outputs → {out_dir}")

if __name__=='__main__':
    main()
