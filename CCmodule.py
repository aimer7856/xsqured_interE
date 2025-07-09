"""
Updated Classical Simulation Modules

This module defines the ClassicalBipartiteSystemSimulator class with several
integration methods. Here we implement the refined RK45 method that returns:
  - time, oscillator position (x), oscillator momentum (px)
  - projectile position (y), projectile momentum (py)
  - oscillator energy, projectile energy, interaction energy, and total energy.
"""

import numpy as np
from scipy.integrate import solve_ivp
import os, time, psutil

START_TIME = time.time()

def log_peak_memory(note="", filename=None):
    if filename is None:
        filename = os.environ.get("MEM_LOG_FILE", "func_mem_log.txt")
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # MB
    timestamp = time.time()
    elapsed = timestamp - START_TIME
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        with open(filename, "a") as f:
            f.write("timestamp,elapsed_seconds,note,memory_MB\n")
    with open(filename, "a") as f:
        f.write(f"{timestamp:.2f},{elapsed:.2f},{note},{mem:.2f}\n")
        
class ClassicalBipartiteSystem:
    def __init__(self, x0=0, y0=0, vx0=0, vy0=0, mx=1, my=1, lambda_=1, 
                 total_time=40, timesteps=401):
        self.mx = mx
        self.my = my
        self.px = mx*vx0
        self.py = my*vy0
        self.lambda_ = lambda_
        self.total_time = total_time
        self.timesteps = timesteps
        self.frames = np.linspace(0, total_time, timesteps)
        # Initial state: [x, y, momentum_x, momentum_y]
        self.state0 = np.array([x0, y0, self.px, self.py])
    
    def update_func(self, t, vals):
        x, y, px, py = vals
        dxdt = px / self.mx
        dydt = py / self.my
        dpxdt = - (self.mx * x + self.lambda_ * 2 * x * np.exp(-y))
        dpydt = self.lambda_ * x**2 * np.exp(-y)
        return [dxdt, dydt, dpxdt, dpydt]
    
    def simulate_rk45_refined(self, rtol=1e-10, atol=1e-12):
        """
        Runs RK45 with refined tolerances and returns:
          t, x, px, y, py, H_osc, H_proj, H_int, H_total
        """
        
        log_peak_memory(note="start simulate_rk45_refined")
        
        sol = solve_ivp(self.update_func, [0, self.total_time], self.state0,
                        t_eval=self.frames, method='RK45', rtol=rtol, atol=atol)
        
        log_peak_memory(note="eafter CC  solve_ivp")
        
        x, y, px, py = sol.y
        H_osc = (px**2) / (2 * self.mx) + self.mx * (x**2) / 2
        H_proj = (py**2) / (2 * self.my)
        H_int = self.lambda_ * x**2 * np.exp(-y)
        H_total = H_osc + H_proj + H_int
        
        log_peak_memory(note="end simulate_rk45_refined")
        return sol.t, x, px, y, py, H_osc, H_proj, H_int, H_total

