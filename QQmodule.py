"""
Created on Wed Mar 26 16:28:42 2025

@author: doyeonkim

Revised: Full implementation with initial wavefunction and time evolution
"""

import numpy as np
import os
import time
import psutil
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from collections import OrderedDict, namedtuple

START_TIME = time.time()

def log_peak_memory(note="", filename=None):
    if filename is None:
        filename = os.environ.get("MEM_LOG_FILE")
        
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
        
        
# -----------------------------------------------------------------------------
# GLOBAL NUMPY SETTINGS – stop on the FIRST floating‑point error
# -----------------------------------------------------------------------------
#np.seterr(over="raise", divide="raise", invalid="raise",under='warn')

# -----------------------------------------------------------------------------
# HELPER
# -----------------------------------------------------------------------------

#def _safe_exp(arr, clip=100.0):
    #"""Exponentiate with overflow protection:  exp(clip(arr,‑clip,+clip))."""
    #return np.exp(np.clip(arr, -clip, +clip))

# =============================================================================
# Quantum Particle Base Class
# =============================================================================
class Quantum_Particle:
    """
    Base class for quantum particles: holds grid, mass, wavefunction routines.
    """

    Obs = namedtuple("Obs",
        ["trace","mean_pos","mean_mom","std_pos","std_mom","mean_H","std_H"]
    )

    
    def __init__(self, n, xmin, xmax, mx, x0, px):
        self.n = n
        self.xmin = xmin
        self.xmax = xmax
        self.mass = mx
        self.x0 = x0
        self.px= px
        self.grid_points = np.linspace(xmin, xmax, n)
        self.ds = self.grid_points[1] - self.grid_points[0]
        self.rho = None

    def get_P(self):
        # first-order central difference momentum operator
        return -1j / (2 * self.ds) * sps.diags([-1, 0, 1], [-1, 0, 1], shape=(self.n, self.n))

    def get_X(self):
        return sps.diags(self.grid_points, 0)

    def get_P_squared(self):
        return -1 / (self.ds**2) * sps.diags([1, -2, 1], [-1, 0, 1], shape=(self.n, self.n))

    def get_X_squared(self):
        return sps.diags(self.grid_points**2, 0)

    def get_H(self):
        return
    
    def get_U(self):
        return
    
    def set_rho(self, psi, ds):
        return 

    def get_expected_position(self):
        X = self.get_X().toarray()
        
       # try:
           #val = self.ds * np.real(np.trace(self.rho @ X))
        #except FloatingPointError as e:
              #raise FloatingPointError("matmul failed in get_expected_position") from e
        #return val
        return self.ds * np.real(np.trace(self.rho @ X))

    def get_expected_momentum(self):
        P = self.get_P().toarray()
        return self.ds * np.real(np.trace(self.rho @ P))
    
    def get_expected_H(self):
        H = self.get_H().toarray()
        return self.ds * np.real(np.trace(self.rho @ H))
    
    def get_std_position(self, mean):
        # ⟨X²⟩ = ds * trace(ρ·X²)
        X2 = self.get_X_squared().toarray()
        expectation_X2 = self.ds * np.real(np.trace(self.rho @ X2))
        # variance = ⟨X²⟩ − ⟨X⟩²
        var = expectation_X2 - mean**2
        # guard against tiny negative rounding errors
        return np.sqrt(max(var, 0.0))
    
    def get_std_momentum(self, mean):
        # ⟨P²⟩ = ds · tr(ρ P²)
        P2 = self.get_P_squared().toarray()
        expectation_P2 = self.ds * np.real(np.trace(self.rho @ P2))
        # variance = ⟨P²⟩ − ⟨P⟩²
        var = expectation_P2 - mean**2
        # clamp any tiny negative rounding error to zero
        return np.sqrt(max(var, 0.0))

    def get_std_H(self, mean):
        # ⟨H²⟩ = ds · tr(ρ H²)
        H = self.get_H().toarray()
        H2 = np.dot(H , H)
        expectation_H2 = self.ds * np.real(np.trace(self.rho @ H2))
        # variance = ⟨H²⟩ − ⟨H⟩²
        var = expectation_H2 - mean**2
        return np.sqrt(max(var, 0.0))
    
  
    def get_expected_obs(self, psi, ds):
        self.set_rho(psi, ds)
        mean_pos = self.get_expected_position()
        mean_mom = self.get_expected_momentum()
        mean_H = self.get_expected_H()
        
        if isinstance(self, Quantum_Oscillator):
            mean_H -= 0.5  # Subtract zero-point energy (ħ = ω = 1)
        
        return [self.ds*np.real(np.trace(self.rho)), 
                mean_pos, 
                mean_mom, 
                self.get_std_position(mean_pos), 
                self.get_std_momentum(mean_mom),
                mean_H,
                self.get_std_H(mean_H)]
    
# =============================================================================
# Quantum Oscillator Class
# =============================================================================
class Quantum_Oscillator(Quantum_Particle):
    def get_H(self):
        return (1/(2*self.mass))*self.get_P_squared() + (self.mass/2)*self.get_X_squared()
    # def get_U(self):
    #     return sps.diags(np.exp(self.grid_points), 0)
    def get_U(self):
        return sps.diags(self.grid_points**2, 0)
    def set_rho(self, psi, ds):
        self.rho = ds * np.matmul(psi, psi.conj().T)
        
# =============================================================================
# Quantum Projectile Class
# =============================================================================
class Quantum_Projectile(Quantum_Particle):
    def get_H(self):
        return (1/(2*self.mass))*self.get_P_squared()
    def get_U(self):
        return sps.diags(np.exp(-self.grid_points), 0)
    
    def set_rho(self, psi, ds):
        self.rho = ds *  np.matmul(psi.T, psi.conj())
       
# =============================================================================
# Quantum Bipartite System Class
# =============================================================================
class Quantum_Bipartite_System:
    def __init__(self, nx, xmin, xmax, mx, x0, vx0,
                       ny, ymin, ymax, my, y0, vy0,
                       sigmax, sigmay, total_time, timesteps,
                       lambda_):
        self.nx, self.ny = nx, ny
        self.x0, self.y0 = x0, y0
        self.px, self.py = mx*vx0, my*vy0
        self.sigmax, self.sigmay = sigmax, sigmay
        self.lambda_ = lambda_
        self.timesteps = timesteps
        self.T = np.linspace(0, total_time, timesteps)
        self.dt = self.T[1] - self.T[0]
        self.oscillator = Quantum_Oscillator(nx, xmin, xmax, mx, x0, self.px)
        self.projectile = Quantum_Projectile(ny, ymin, ymax, my, y0, self.py)

    def get_meshgrid(self):
        x_grid = self.oscillator.grid_points
        y_grid = self.projectile.grid_points
        return np.meshgrid(x_grid, y_grid)

    # def init_wavefunction(self):
    #     """
    #     Ψ₀(x,y) = exp[-i(px·x - py·y)] · exp[-((x-x0)²)/(4σx²)] · exp[-((y-y0)²)/(4σy²)]
    #     """
    #     X, Y = self.get_meshgrid()
    #     psi = (1.0/(np.sqrt(2)*(np.pi**4))) * \
    #           np.exp(-1j*(self.px*X - self.py*Y)) * \
    #           np.exp(-((X-self.x0)**2)/(4*self.sigmax**2)) * \
    #           np.exp(-((Y-self.y0)**2)/(4*self.sigmay**2))
    #     return psi

    def init_wavefunction(self):
        """
        Initialize Ψ₀(x,y) with the oscillator in a true coherent state (x0, p0),
        and the projectile in a standard Gaussian wavepacket.
        """
        X, Y = self.get_meshgrid()

        # Constants
        hbar = 1.0
        omega = 1.0
        m = self.oscillator.mass

        # Coherent state parameters from given x0, px0
        re_alpha = np.sqrt(m * omega / (2 * hbar)) * self.x0
        im_alpha = self.px / np.sqrt(2 * hbar * m * omega)
        alpha0 = re_alpha + 1j * im_alpha
        abs_alpha = np.abs(alpha0)
        sigma = np.angle(alpha0)

        # Coherent state's displacement
        x_shift = np.sqrt(2 * hbar / (m * omega)) * re_alpha
        p_shift = np.sqrt(2 * m * omega * hbar) * im_alpha

        # Coherent state's global phase at t=0
        theta_0 = -0.5 * abs_alpha**2 * np.sin(-2 * sigma)

        # Width of ground state
        #sigmax = np.sqrt(hbar / (2 * m * omega))
        norm_x = (m * omega / (np.pi * hbar))**0.25

        # Oscillator's coherent wavefunction
        psi_x = norm_x * np.exp(
            - (m * omega / (2 * hbar)) * (X - x_shift)**2 +
            1j * p_shift * X / hbar +
            1j * theta_0
        )

        # Projectile's Gaussian wavefunction
        norm_y = (1.0 / (np.pi * self.sigmay**2))**0.25
        psi_y = norm_y * np.exp(-((Y - self.y0)**2) / (2 * self.sigmay**2)) * np.exp(1j * self.py * Y)

        psi = psi_x * psi_y
        return psi

    def normalized_init_wavefunction(self, psi_flat):
        norm = (self.oscillator.ds * self.projectile.ds * np.vdot(psi_flat, psi_flat))
        return psi_flat/np.sqrt(norm)

    def __CN_method(self, k, psi_evol):
        """
        Crank–Nicolson propagation: (I - dt/2 A) ψ^{n+1} = (I + dt/2 A) ψ^n
        where A = -i[H_osc ⊕ H_proj + λ U_int]
        """
        H_sum = sps.kronsum(self.oscillator.get_H(), self.projectile.get_H())
        U_int = sps.kron(self.projectile.get_U(), self.oscillator.get_U())
        A = -1j*(H_sum + self.lambda_*U_int)
        PNext = sps.identity(k) - (self.dt/2)*A
        PPrev = sps.identity(k) + (self.dt/2)*A

        vec = psi_evol[0,:]
        for t in range(1, self.timesteps):
            rhs = PPrev.dot(vec)
            vec = spla.spsolve(PNext, rhs)
           # --- new: rescale & guard ---
            #nrm = np.linalg.norm(vec)
            #if not np.isfinite(nrm) or nrm == 0.0:
                #raise FloatingPointError(f"norm blew up at step {t}: {nrm}")
            #vec /= nrm
            # ----------------------------
        
            psi_evol[t, :] = vec

    def time_evolution_wavefunction(self):
        log_peak_memory(note="start time_evolution_wavefunction")
        #print(f"[DEBUG] Calling log_peak_memory() with MEM_LOG_FILE = {os.environ.get('MEM_LOG_FILE')}")
        k = self.nx * self.ny
        psi0 = self.init_wavefunction()
       
        psi_evol = np.zeros((self.timesteps, k), dtype='complex')
        log_peak_memory(note="after allocating psi_evol array")
        flat0 = psi0.reshape(k, order='C')
        psi_evol[0,:] = self.normalized_init_wavefunction(flat0)
      
        self.__CN_method(k, psi_evol)
        log_peak_memory(note="after __CN_method propagation")
        log_peak_memory(note="end time_evolution_wavefunction")
        
        return psi_evol

    
    def get_VN_entanglement(self):
        def F(x):
            if abs(x) > 1e-14:
                return -np.real(x) * np.log(np.real(x))
            else:
                return 0
        eigenvalues = np.linalg.eig(self.oscillator.ds *self.oscillator.rho)[0]
        return np.sum(np.vectorize(F)(eigenvalues))
    
    def get_expected_InterEngergy(self, psi):
        U1 = self.oscillator.get_U().toarray()
        U2 = self.projectile.get_U().toarray()
        A = np.dot(psi.conj().T , U1)
        B = np.dot(psi, U2)
        return self.oscillator.ds*self.projectile.ds*np.real(np.trace(np.dot(A,B)))
    """
    def get_expected_InterEngergy(self, psi):
        
        #<H_int> = λ · (ds_x·ds_y) · Σ_{j,i} |ψ_{j,i}|² · U2[j] · U1[i],
        #implemented by extracting the diagonals of the sparse U operators.
        
        # get the diagonal entries of each U (shape (nx,) and (ny,))
        dU1 = self.oscillator.get_U().diagonal()    # length nx
        dU2 = self.projectile.get_U().diagonal()    # length ny
    
        # weight each row of psi by dU2[j]
        # psi has shape (ny, nx)
        weighted = psi * dU2[:, None]
    
        # now sum over i,j: dU1[i] * |weighted[j,i]|^2
        E_U = np.sum(dU1[None, :] * np.abs(weighted)**2)
    
        # include both grid spacings and the coupling λ
        return self.lambda_ * self.oscillator.ds * self.projectile.ds * E_U
    """
    def get_std_interaction_energy(self, psi, mean):
        """
        σ(H_int) = sqrt( ⟨H_int^2⟩ – ⟨H_int⟩^2 )
        computed in O(nx·ny) by reading the diagonals of U_osc and U_proj.
        """
        # extract diagonal entries of each U operator
        dU1 = self.oscillator.get_U().diagonal()   # shape (nx,)
        dU2 = self.projectile.get_U().diagonal()   # shape (ny,)
    
        # 2) compute ⟨H_int^2⟩: use squared diagonals
        weighted2 = psi * (dU2**2)[:, None]
        E_U2      = np.sum((dU1**2)[None, :] * np.abs(weighted2)**2)
        second    = (self.lambda_**2) * self.oscillator.ds * self.projectile.ds * E_U2
    
        # 3) form variance and take sqrt, guarding against tiny negative drift
        var = second - mean**2
        return np.sqrt(max(var, 0.0))

    def get_expected_data(self, include=None):
        """
        Compute any of:
          'oscillator'   : (t,7)
          'projectile'   : (t,7)
          'vn_entropy'   : (t,)
          'inter_energy' : (t,)
          'std_inter_energy' : (t,)
          'linear_entropy':(t,)
          'rho1_diag'    : (t,nx)
          
        Returns T and dict of requested arrays.
        """
        log_peak_memory(note="start get_expected_data")
        
        ALL = ('oscillator','projectile','vn_entropy',
               'inter_energy','linear_entropy','rho1_diag','rho1')
        if include is None:
            include = ALL
        elif isinstance(include, str):
            include = (include,)
        include = [k for k in include if k in ALL]

        results = {}
        if 'oscillator'     in include:
            results['oscillator']     = np.zeros((self.timesteps,7))
        if 'projectile'     in include:
            results['projectile']     = np.zeros((self.timesteps,7))
        if 'vn_entropy'     in include:
            results['vn_entropy']     = np.zeros(self.timesteps)
        if 'inter_energy'   in include:
            results['inter_energy']   = np.zeros(self.timesteps)
        if 'std_inter_energy'   in include:
            results['std_inter_energy']   = np.zeros(self.timesteps)
        if 'linear_entropy' in include:
            results['linear_entropy'] = np.zeros(self.timesteps)
        if 'rho1_diag'      in include:
            results['rho1_diag']      = np.zeros((self.timesteps,self.nx))
            
        log_peak_memory(note="after array allocations")
        
    
        psi_evol = self.time_evolution_wavefunction()
        
        # Generate 10 evenly spaced memory checkpoints
        checkpoints = np.linspace(0, self.timesteps - 1, num=10, dtype=int)
        
        for t in range(self.timesteps):
            psi = psi_evol[t].reshape((self.ny,self.nx),order='C')
            psiT = psi.T
            if 'oscillator' in include:
                results['oscillator'][t] = self.oscillator.get_expected_obs(psiT,self.projectile.ds)
            if 'projectile' in include:
                results['projectile'][t] = self.projectile.get_expected_obs(psiT,self.oscillator.ds)
            if 'vn_entropy' in include:
                results['vn_entropy'][t] = self.get_VN_entanglement()
            if 'inter_energy' in include:
                results['inter_energy'][t] = self.lambda_*self.get_expected_InterEngergy(psiT)
            if 'std_inter_energy' in include:
                mean = results['inter_energy'][t]
                results['std_inter_energy'][t] = self.get_std_interaction_energy(psi,mean)
            rho1 = self.oscillator.rho
            if 'rho1_diag' in include:
                results['rho1_diag'][t] = np.real(np.diag(rho1))
            if 'linear_entropy' in include:
                purity = self.oscillator.ds**2*np.real(np.trace(np.dot(rho1,rho1)))
                results['linear_entropy'][t] = 1.0-purity
                
            # Log memory at select checkpoints
            if t in checkpoints:
               log_peak_memory(note=f"step {t} in get_expected_data")

            
        log_peak_memory(note="end get_expected_data")    
        
        return self.T, results