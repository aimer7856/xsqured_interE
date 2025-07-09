"""
classical_quantum_ehrenfest.py

Module defining a single class CQEhrenfestSystem that models
an eigenbasis quantum harmonic oscillator coupled to a classical projectile
via the Ehrenfest mean-field approximation:

    i dC/dt = λ e^{-y(t)} B(t) C
      ẏ    = p_y/m_y
      ṗ_y  = λ e^{-y(t)} C† M C

It provides:
 - analytic overlap M matrix
 - operator matrices X, P, Hx, Hx2
 - initial-state generators (ground, Gaussian)
 - ODE solver for coupled dynamics
 - expectation & stddev computation for ⟨x⟩, ⟨p⟩, ⟨Hx⟩
 - classical energy Hy
"""
import numpy as np
import os
import time
import psutil
from scipy.integrate import solve_ivp
from scipy.special import factorial, genlaguerre



START_TIME = time.time()

def log_peak_memory(note="", filename=None):
    if filename is None:
        filename = os.environ.get("MEM_LOG_FILE", "func_mem_log.txt")
        
     # Ensure header is written once if file is new or empty
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        with open(filename, "a") as f:
            f.write("timestamp,elapsed_seconds,note,memory_MB\n")

    
    # Get memory in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    
    # Current time and elapsed time
    timestamp = time.time()
    elapsed = timestamp - START_TIME  # seconds since script start
    
    with open(filename, "a") as f:
        f.write(f"{timestamp:.2f},{elapsed:.2f},{note},{mem:.2f}\n")

class CQEhrenfestSystem:
    def __init__(self, N_eig: int, xmin,xmax,mx: float, x0: float,vx0: float,ymin:float, ymax:float, my: float, y0: float,  vy0:float, sigmax:float, total_time:float, timesteps:int, lambda_: float, omega: float = 1.0):
        """
        Initialize masses, coupling, and basis dimension.
        Precompute energy levels, overlap matrix M, and operators.
        """
        if N_eig < 1 or not isinstance(N_eig, int):
            raise ValueError("N must be a positive integer")
        self.N_eig = N_eig
        self.xmin = xmin
        self.xmax = xmax
        self.mx = mx
        self.x0 = x0
        self.vx0 = vx0
        self.ymin =ymin
        self.ymax = ymax
        self.my = my
        self.y0 = y0
        self.vy0 = vy0
        self.sigmax = sigmax
        self.total_time = total_time
        self.timesteps = timesteps
        self.lambda_ = lambda_
        self.omega = omega
        
        self.C0 = self.get_gaussian_state()
        # harmonic oscillator energies Em = ω(m + 1/2)
        self.E = omega * (np.arange(N_eig, dtype=float) + 0.5)
        # analytic overlap matrix M_{mn} = <φ_m|e^x|φ_n>
        self.M = self._analytic_M(mx, N_eig)
        # operators in energy basis
        self.X  = self._X_matrix()
        self.P  = self._P_matrix()
        self.Hx = np.diag(self.E)
        self.Hx2 = np.diag(self.E**2)

    @staticmethod
    def _analytic_M(mx: float, N: int) -> np.ndarray:
        """
        Overlap matrix M_{mn} = ⟨φ_m|x²|φ_n⟩ in harmonic oscillator eigenbasis.
        This replaces e^x with x² as the interaction operator.
        """
        M = np.zeros((N, N), dtype=float)
        pref_diag = (2*np.arange(N) + 1) / (2*mx)
        M += np.diag(pref_diag)

        for n in range(N - 2):
            off_diag = np.sqrt((n + 1) * (n + 2)) / (2*mx)
            M[n, n + 2] = M[n + 2, n] = off_diag

        return M

    def _X_matrix(self) -> np.ndarray:
        pref = 1.0/np.sqrt(2.0*self.mx*self.omega)
        off = pref * np.sqrt(np.arange(1, self.N_eig))
        return np.diag(off,1) + np.diag(off,-1)

    def _P_matrix(self) -> np.ndarray:
        pref = 1j * np.sqrt(self.mx*self.omega/2.0)
        off = pref * np.sqrt(np.arange(1, self.N_eig))
        return np.diag(-off,1) + np.diag(off,-1)


    def get_gaussian_state(self) -> np.ndarray:
        """
        Gaussian wavepacket in eigenbasis with center x0 and velocity v0.
        """
        p0 = self.mx * self.vx0
        alpha = (np.sqrt(self.mx*self.omega/2.0)*self.x0 + 1j*p0*self.sigmax)
        pref = np.exp(-abs(alpha)**2/2.0)
        n = np.arange(self.N_eig)
        return pref * alpha**n / np.sqrt(factorial(n))

        # =========================
    # UPDATED METHOD STARTS HERE
    # Time-dependent expectation value and variance using Hadamard phase product
    # Implements:
    #   ⟨A⟩(τ) = C*ᵗ · (e^{i(Eₘ - Eₙ)τ} ∘ A) · C
    #   σ²(τ) = ⟨A²⟩(τ) - ⟨A⟩²(τ)
    # =========================
    def _mean_std(self, t: float, op: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the expectation value and standard deviation of an operator in energy eigenbasis.

        Parameters
        ----------
        t : float
            Time τ at which to evaluate.
        op : np.ndarray
            Operator matrix A in the energy eigenbasis.
        C : np.ndarray
            Coefficients C_n(τ) of the state in the eigenbasis.

        Returns
        -------
        mean : float
            Expectation value ⟨A⟩(τ).
        std : float
            Standard deviation σ_A(τ).
        """
        # Compute the phase matrix: B_{mn} = exp(i(E_m - E_n) * t)
        phase = np.exp(1j * (self.E[:, None] - self.E[None, :]) * t)

        # Hadamard product B_A = phase ∘ A
        B = phase * op

        # Expectation value: C† (B ∘ A) C
        mean = np.vdot(C, B @ C)

        # Repeat for A^2
        A2 = op @ op
        B2 = phase * A2
        mean2 = np.vdot(C, B2 @ C)

        # Variance = ⟨A²⟩ - ⟨A⟩²
        var = mean2 - mean**2
        std = np.sqrt(np.clip(var.real, 0, None))

        return mean.real, std
    # =========================
    # END OF UPDATED METHOD
    # =========================

    def _rhs(self, t: float, S: np.ndarray) -> np.ndarray:
        """
        RHS for combined ODE: real/imag parts of C, plus y, py.
        """
        N = self.N_eig
        Cr = S[:N]; Ci = S[N:2*N]
        y, py = S[2*N], S[2*N+1]
        C = Cr + 1j*Ci
        # time-dependent phase matrix B
        phase = np.exp(1j*(self.E[:,None]-self.E[None,:])*t)
        B = phase * self.M
        factor = self.lambda_ * np.exp(-y)
        #print(f"[debug] time={t}, y = {y}")
        dC = -1j * factor * (B @ C)
        dy = py / self.my
        dpy = factor * np.vdot(C, B @ C).real
        return np.concatenate([dC.real, dC.imag, [dy, dpy]])

    def solve(self, atol: float = 1e-9, rtol: float = 1e-9) -> dict:
        """
        Integrate the Ehrenfest ODE up to t_final.
        Returns dict with time series and observables.
        """
        
        log_peak_memory(note="start solve")
        
        N= self.N_eig
        tn = self.timesteps
        py0 = self.my * self.vy0
        S0 = np.concatenate([self.C0.real, self.C0.imag, [self.y0, py0]])
        t_eval = np.linspace(0, self.total_time, tn)
        
        # Use BDF if system is expected to be stiff
        #method = 'Radau' if self.mx < 1.0 else 'RK45'
    
        sol = solve_ivp(self._rhs, [0, self.total_time], S0,
                        t_eval=t_eval, method='RK45', atol=atol, rtol=rtol)
        #print(f"[diag] steps={sol.nfev},  LU factorizations={sol.nlu},  message=‘{sol.message}’")
        #print(f"[diag] steps={sol.nfev},  message=‘{sol.message}’")
        t = sol.t
        Cr = sol.y[:N]
        Ci = sol.y[N:2*N]
        C = Cr + 1j * Ci
        C = C.T  # shape (steps, N)
    
        y = sol.y[2*N]
        py = sol.y[2*N+1]
       # print(f"[diag] CQ solver completed {len(t)} steps out of requested {self.timesteps}")
       # print(f"[diag] min(y): {np.min(y)}, max(y): {np.max(y)}")  # assuming y is at index 4
        # Allocate arrays for observables
        x_exp = np.empty(tn)
        std_x = np.empty(tn)
        px_exp = np.empty(tn)
        std_px = np.empty(tn)
        Hx_exp = np.empty(tn)
        std_Hx = np.empty(tn)
        
        # Adaptive memory logging checkpoints
        checkpoints = set(np.linspace(0, tn - 1, num=10, dtype=int))
    
        for i in range(tn):
            x_exp[i], std_x[i] = self._mean_std(t[i], self.X, C[i])
            px_exp[i], std_px[i] = self._mean_std(t[i], self.P, C[i])
            Hx_exp[i], std_Hx[i] = self._mean_std(t[i], self.Hx, C[i])
            Hx_exp[i]-=0.5 # subtract zero-point energy
            
            if i in checkpoints:
               log_peak_memory(note=f"step {i} in solve")

               
               
        Hy = py**2 / (2 * self.my)
        
        log_peak_memory(note="end solve")
        
        return {
            't': t,
            'C': C,
            'x': x_exp,
            'std_x': std_x,
            'px': px_exp,
            'std_px': std_px,
            'Hx': Hx_exp,
            'std_Hx': std_Hx,
            'y': y,
            'py': py,
            'Hy': Hy
        }
        