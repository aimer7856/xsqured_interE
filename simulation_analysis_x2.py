import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re

def extract_mx_my(folder_name):
    """
    Extract mx and my values from folder names like mx1.0_my0.5
    """
    match = re.match(r"mx(\d+(?:\.\d+)?)_my(\d+(?:\.\d+)?)", folder_name)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def load_simulation_data(folder, mode):
    """
    Load .npz + params.json from a folder, according to simulation mode.
    mode must be one of: 'qq', 'cc', 'cq'
    """
    if mode == "qq":
        npz_files = glob.glob(os.path.join(folder, "*.npz"))
    elif mode == "cc":
        npz_files = glob.glob(os.path.join(folder, "*.npz"))
    elif mode == "cq":
        npz_files = glob.glob(os.path.join(folder, "*.npz"))
    else:
        raise ValueError(f"Unknown mode {mode}")

    json_files = glob.glob(os.path.join(folder, "*_params.json"))

    if not npz_files or not json_files:
        raise FileNotFoundError(f"Missing data for {mode} in {folder}")

    data = np.load(npz_files[0])
    with open(json_files[0], "r") as f:
        params = json.load(f)
    return data, params

def scan_all_data(root_dir="results_x2"):
    """
    Scan root_dir for subfolders: 'qq', 'cq', 'cc'
    Under each, look for mx*_my* folders and load data.
    Returns:
        results[(mx, my)] = {'qq':(...), 'cc':(...), 'cq':(...)}
    """
    folder_map = {
        "qq": "qq",
        "cq": "cq",
        "cc": "cc"
    }
    results = {}
    for folder_name, mode_key in folder_map.items():
        mode_dir = os.path.join(root_dir, folder_name)
        print(mode_dir)
        if not os.path.isdir(mode_dir):
            continue
        for sub in os.listdir(mode_dir):
            subpath = os.path.join(mode_dir, sub)
        
            if not os.path.isdir(subpath):
                continue
            mx, my = extract_mx_my(sub)
            print(mx, my)
            if mx is None or my is None:
                continue
            try:
                #print("Contents of", subpath, ":", os.listdir(subpath))
                #print("  .npz matches:", glob.glob(os.path.join(subpath, "*_*.npz")))
                #print("  .json matches:", glob.glob(os.path.join(subpath, "*params_*.json")))
                data, params = load_simulation_data(subpath, mode_key)
                results.setdefault((mx, my), {})[mode_key] = (data, params)
            except Exception as e:
                print(f"[WARN] {mode_key} load failed at {subpath}: {e}")
    return results

def plot_subplot(ax, title, label, t, data_c, data_q, data_cq=None, std_q=None):
    ax.plot(t, data_c, 'C0-', label='Classical-Classical')
    ax.plot(t, data_q, 'C1-', label='Quantum-Quantum')
    if data_cq is not None:
        ax.plot(t, data_cq, 'C2-', label='Classical-Quantum')
    if std_q is not None:
        ax.fill_between(t, data_q - std_q, data_q + std_q,color='C1', alpha=0.3, label='Quantum ±1σ')
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(label)
    ax.legend()
    ax.grid(True)

def process_folder(mx, my, data_dict, out_dir):
    fig, axs = plt.subplots(3,3,figsize=(18,18))
    plt.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.88, wspace=0.15, hspace=0.2)

    # Initialize variables
    rho1_init = x_grid = None
    int_c = int_q = std_int_q = None
    nx=ny= None
    t = None  # initialize safely
    xmin = xmax = ymin = ymax = x0 = y0 = None
    sigmax = sigmay = None
    lambda_ = total_time = timesteps = None
    cruntime = qruntime = cqruntime = None
    cc_x = cc_px = cc_Hx = None
    cc_y = cc_py = cc_Hy = None
    cq_x = cq_px = cq_Hx = None
    cq_y = cq_py = cq_Hy = None
    exp_x = exp_px = Hx = std_x = std_px = None
    exp_y = exp_py = Hy = std_y = std_py = None
    vn = lin = None

    # Quantum data
    if "qq" in data_dict:
        qdata, qparams = data_dict["qq"]
        t = qdata["t"]
        osc = qdata["oscillator"]
        proj = qdata["projectile"]
        exp_x, std_x   = osc[:,1], osc[:,3]
        exp_px, std_px = osc[:,2], osc[:,4]
        Hx, std_Hx     = osc[:,5], osc[:,6]
        exp_y, std_y   = proj[:,1], proj[:,3]
        exp_py, std_py = proj[:,2], proj[:,4]
        Hy, std_Hy     = proj[:,5], proj[:,6]
        vn = qdata["vn_entropy"]
        lin = qdata.get("linear_entropy", None)
        rho1_init = qdata["rho1_diag"][0]
        int_q = qdata["inter_energy"]
        std_int_q = qdata.get("std_inter_energy", None)

        qmeta = qparams.get("qq", {})
        xmin = qmeta.get('xmin', -10)
        xmax = qmeta.get('xmax', 10)
        nx   = qmeta.get('nx', 256)
        x0   = qmeta.get('x0', 0.0)
        sigmax = qmeta.get('sigmax', 1.0)
        my   = qmeta.get('my', 1)
        ymin = qmeta.get('ymin', -10)
        ymax = qmeta.get('ymax', 100)
        ny   = qmeta.get('ny', 2048)
        y0   = qmeta.get('y0', 0.0)
        sigmay = qmeta.get('sigmay', 3.0)
        
        lambda_ = qmeta.get('lambda_')
        total_time = qmeta.get('total_time')
        timesteps = qmeta.get('timesteps')
        
        qruntime = qmeta.get('runtime_qq')
        
        x_grid = np.linspace(xmin, xmax, nx)

    # Classical data (folder 'cc')
    if "cc" in data_dict:
        cdata, cparams = data_dict["cc"]
        cc_x, cc_px, cc_Hx = cdata["x"], cdata["px"], cdata["Hx"]
        cc_y, cc_py, cc_Hy = cdata["y"], cdata["py"], cdata["Hy"]
        int_c = cdata["Hint"]

        if t is None:
           t = cdata["t"]
      
        
        cmeta = cparams.get("cc", {})
        x0 = cmeta.get("x0")
        y0 = cmeta.get("y0")
        vx0 = cmeta.get("vx0")
        vy0 = cmeta.get("vy0")
        mx = cmeta.get("mx")
        my = cmeta.get("my")
        lambda_ = cmeta.get("lambda_")
        total_time = cmeta.get("total_time")
        timesteps = cmeta.get("timesteps")
        cruntime = cmeta.get("runtime_cc")
        
    # CQ data
    if "cq" in data_dict:
        cqdata, cqparams = data_dict["cq"]
        cq_x, cq_px, cq_Hx = cqdata["x"], cqdata["px"], cqdata["Hx"]
        cq_y, cq_py, cq_Hy = cqdata["y"], cqdata["py"], cqdata["Hy"]
        
        if t is None:
           t = cqdata["t"]

        cqmeta = cqparams.get("cq", {})
        N_eig = cqmeta.get("N_eig")
        xmin = cqmeta.get("xmin")
        xmax = cqmeta.get("xmax")
        mx = cqmeta.get("mx")
        x0 = cqmeta.get("x0")
        vx0 = cqmeta.get("vx0")
        ymin = cqmeta.get("ymin")
        ymax = cqmeta.get("ymax")
        my = cqmeta.get("my")
        y0 = cqmeta.get("y0")
        vy0 = cqmeta.get("vy0")
        sigmax = cqmeta.get("sigmax")
        sigmay = cqmeta.get("sigmay")
        lambda_ = cqmeta.get("lambda_")
        total_time = cqmeta.get("total_time")
        timesteps = cqmeta.get("timesteps")
        cqruntime = cqmeta.get("runtime_cq")
       # print(cqruntime)
        
    # Row 1: Oscillator
    osc_items = [
        ('Oscillator Position ⟨x⟩', '⟨x⟩', 
        data_dict.get("cc", (None,))[0]["x"] if "cc" in data_dict else None,
        data_dict.get("qq", (None,))[0]["oscillator"][:,1] if "qq" in data_dict else None,
        data_dict.get("cq", (None,))[0]["x"] if "cq" in data_dict else None,
        data_dict.get("qq", (None,))[0]["oscillator"][:,3] if "qq" in data_dict else None),

        ('Oscillator Momentum ⟨px⟩', '⟨px⟩',
        data_dict.get("cc", (None,))[0]["px"] if "cc" in data_dict else None,
        data_dict.get("qq", (None,))[0]["oscillator"][:,2] if "qq" in data_dict else None,
        data_dict.get("cq", (None,))[0]["px"] if "cq" in data_dict else None,
        data_dict.get("qq", (None,))[0]["oscillator"][:,4] if "qq" in data_dict else None),

        ('Oscillator Energy ⟨Hx⟩', '⟨Hx⟩',
        data_dict.get("cc", (None,))[0]["Hx"] if "cc" in data_dict else None,
        data_dict.get("qq", (None,))[0]["oscillator"][:,5] if "qq" in data_dict else None,
        data_dict.get("cq", (None,))[0]["Hx"] if "cq" in data_dict else None,
        data_dict.get("qq", (None,))[0]["oscillator"][:,6] if "qq" in data_dict else None)
    ]

    for i, (ttl, lbl, cd, qd, cqd, std_q) in enumerate(osc_items):
        ax = axs[0, i]
        
        # Check what is available
        has_q = qd is not None
        has_c = cd is not None
        has_cq = cqd is not None
        
        if has_q:
            plot_subplot(ax, ttl, lbl, t, cd if has_c else np.zeros_like(t), qd, cqd if has_cq else None, std_q)
        elif has_c or has_cq:
            # Use available data to plot at least something
            if has_c:
                ax.plot(t, cd, 'C0-', label='Classical')
            if has_cq:
                ax.plot(t, cqd, 'C2-', label='Classical-Quantum')
            ax.set_title(ttl + " (no quantum)")
            ax.set_xlabel("Time")
            ax.set_ylabel(lbl)
            ax.legend()
            ax.grid(True)
        else:
            # No data at all
            ax.set_title(ttl + " (no data)")
            ax.axis("off")
    # Row 2: Projectile (general case)
    proj_items = [
        ('Projectile Position ⟨y⟩', '⟨y⟩', exp_y if "qq" in data_dict else None,
        std_y if "qq" in data_dict else None,
        cc_y if "cc" in data_dict else None,
        cq_y if "cq" in data_dict else None),

        ('Projectile Momentum ⟨py⟩', '⟨py⟩', exp_py if "qq" in data_dict else None,
        std_py if "qq" in data_dict else None,
        cc_py if "cc" in data_dict else None,
        cq_py if "cq" in data_dict else None),

        ('Projectile Energy ⟨Hy⟩', '⟨Hy⟩', Hy if "qq" in data_dict else None,
        std_Hy if "qq" in data_dict else None,
        cc_Hy if "cc" in data_dict else None,
        cq_Hy if "cq" in data_dict else None)
    ]

    for i, (ttl, lbl, qd, std_q, cd, cqd) in enumerate(proj_items):
        ax = axs[1, i]

        has_q = qd is not None
        has_c = cd is not None
        has_cq = cqd is not None

        if has_q:
            plot_subplot(ax, ttl, lbl, t, cd if has_c else np.zeros_like(t), qd, cqd if has_cq else None, std_q)
        elif has_c or has_cq:
            # fallback plot without quantum
            if has_c:
                ax.plot(t, cd, 'C0-', label='Classical')
            if has_cq:
                ax.plot(t, cqd, 'C2-', label='Classical-Quantum')
            ax.set_title(ttl + " (no quantum)")
            ax.set_xlabel("Time")
            ax.set_ylabel(lbl)
            ax.legend()
            ax.grid(True)
        else:
            ax.set_title(ttl + " (no data)")
            ax.axis("off")
    # Row 3, Col 0: Initial profile
    if rho1_init is not None and x_grid is not None:
        ax = axs[2,0]
        analytic = (1/(np.sqrt(2*np.pi)*sigmax)) * np.exp(-(x_grid-x0)**2/(2*sigmax**2))
        ax.plot(x_grid, rho1_init, 'C1--', marker='D',  markevery=1, label='Initial Data')
        ax.plot(x_grid, analytic, 'C2-', linewidth =2, label='Analytic Gaussian')
        ax.set_title('Initial x-profile'); ax.set_xlabel('x'); ax.set_ylabel('Probability density')
        ax.legend(); ax.grid(True)

    # Row 3, Col 1: Interaction Energy (general case)
    ax = axs[2, 1]

    has_q = int_q is not None
    has_c = int_c is not None

    if has_q:
        plot_subplot(ax, "Interaction Energy", "Energy", t,
                    int_c if has_c else np.zeros_like(t),
                    int_q, None, std_int_q)
    elif has_c:
        ax.plot(t, int_c, 'C0-', label='Classical-Classical')
        ax.set_title("Interaction Energy (no quantum)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Energy")
        ax.legend()
        ax.grid(True)
    else:
        ax.set_title("Interaction Energy (no data)")
        ax.axis("off")

    # Row 3, Col 2: Entropies
    ax = axs[2,2]
    if "quantum" in data_dict:
        ax.plot(t, vn, 'C2-', label='Von Neumann')
        if lin is not None:
            ax.plot(t, lin, 'C3-', label='Linear')
        ax.set_title('Entropies'); ax.set_xlabel('Time'); ax.set_ylabel('Entropy')
        ax.legend(); ax.grid(True)

    fig = ax.figure
    fig.suptitle(f"QQ vs CC vs CQ Observables (mx,my)= ({mx},{my})", fontsize=18)
    subtitle = "\n".join([
        f"(x0,y0)=({x0},{y0}), (sigmax,sigmay)=({sigmax:.2f}, {sigmay})",
        f"(xmin,xmax)=({xmin},{xmax}), (ymin,ymax)=({ymin},{ymax}), (nx,ny)=({nx},{ny})",
        f"lambda = {lambda_}, N_eig = {N_eig}, total_time = {total_time}, timesteps = {timesteps}",
        f"runtime: QQruntime={qruntime}, CCruntime={cruntime}, CQruntime={cqruntime}"
    ])
    fig.text(0.5, 0.91, subtitle, ha='center', fontsize=14)
    fname = os.path.join(out_dir, f"panel_mx{mx}_my{my}.png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)

def plot_entropy_energy_by_mx(grouped, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for mx, items in grouped.items():
       
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        
        # Energy
        ax_h = axs[0]
        for my,(t,vn,energy) in sorted(items, key=lambda x: x[0]):
            ax_h.plot(t, energy, label=f"my={my}")
        ax_h.set_title(f"Oscillator Energy⟨Hx⟩ vs Time (mx={mx})")
        ax_h.set_xlabel("Time"); ax_h.set_ylabel("⟨Hx⟩"); ax_h.legend(); ax_h.grid(True)
        
        # Entropy
        ax_s = axs[1]
        for my,(t,vn,energy) in sorted(items, key=lambda x: x[0]):
            ax_s.plot(t, vn, label=f"my={my}")
        ax_s.set_title(f"Von Neumann Entory vs Time (mx={mx})")
        ax_s.set_xlabel("Time"); ax_s.set_ylabel("VN Entropy"); ax_s.legend(); ax_s.grid(True)
        
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"energy_entropy_vs_my_mx{mx}.png"), dpi =300)
        plt.close(fig)

def main():
    root = "/Users/doyeonkim/OneDrive/Documents/Project1_Sanjeev/xsqured_interE/results_x2"
    panel_dir = os.path.join(root, "panels_all_modes")
    summary_dir = os.path.join(root, "entropy_energy_by_mx")
    os.makedirs(panel_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    all_data = scan_all_data(root_dir=root)
    grouped = {}
   
    for (mx, my), data_dict in all_data.items():
        try:
            process_folder(mx, my, data_dict, panel_dir)

            # Only include quantum entries in entropy-energy summary
            if "quantum" in data_dict:
                qdata, _ = data_dict["quantum"]
                t = qdata["t"]
                vn = qdata.get("vn_entropy", None)
                energy = qdata["oscillator"][:,5]

                if vn is not None:
                    grouped.setdefault(mx, []).append((my, (t, vn, energy)))

        except Exception as e:
            print(f"[ERROR] mx={mx}, my={my} failed: {e}")

    # Only call plotting if there is quantum data grouped
    if grouped:
        plot_entropy_energy_by_mx(grouped, summary_dir)

if __name__ == "__main__":
    main()
