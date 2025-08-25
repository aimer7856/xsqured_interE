import itertools

#import os

#os.makedirs("test",exist_ok=True)

# Parameters to sweep
modes       = ["qq", "cq", "cc"]
mx_vals     = [0.25,0.5,1.0,2.0,4.0]
my_vals     = [0.25,0.5,1.0,2.0,4.0]
x0_vals     = [5.0]
vx0_vals    = [0.0]

# Fixed parameters
nx_vals     = [256]
xmin_vals   = [-10.0]
xmax_vals   = [10.0]
ny_vals     = [4096]
ymin_vals   = [-15.0]
ymax_vals   = [130.0]
y0_vals     = [10.0]
vy0_vals    = [-1.0]
sigmay_vals = [3.0]
total_time_vals = [30.0]
timesteps_vals  = [4096]
lambda_vals = [1.0]
n_eig_vals  = [192]

# Header
header = [
    "mode", "mx", "my", "x0", "vx0",
    "nx", "xmin", "xmax", "ny", "ymin", "ymax",
    "y0", "vy0", "sigmay", "total_time", "timesteps",
    "lambda", "n_eig", "filename"
]

#with open(os.path.join("test", "coherent_param_list.txt"), "w") as f:
with open("x2_param_list_x0_5.txt", "w") as f:
    f.write(','.join(header) + '\n')
    for combo in itertools.product(
        modes, mx_vals, my_vals, x0_vals, vx0_vals,
        nx_vals, xmin_vals, xmax_vals, ny_vals, ymin_vals, ymax_vals,
        y0_vals, vy0_vals, sigmay_vals, total_time_vals, timesteps_vals,
        lambda_vals, n_eig_vals
    ):
        (
            mode, mx, my, x0, vx0,
            nx, xmin, xmax, ny, ymin, ymax,
            y0, vy0, sigmay, total_time, timesteps,
            lambda_, n_eig
        ) = combo

        # Only include varying parameters in filename
        filename_parts = []
        # Add x0 or vx0 to filename if they are varied
        if len(x0_vals) > 1:
            filename_parts.append(f"x0{x0}")
        if len(vx0_vals) > 1:
            filename_parts.append(f"vx0{vx0}")
        if len(mx_vals) > 1:
            filename_parts.append(f"mx{mx}")
        if len(my_vals) > 1:
            filename_parts.append(f"my{my}")
        if len(modes) > 1:
            filename_parts.insert(0, mode)  # put mode first
        print(filename_parts)
        filename = "_".join(filename_parts)
        
        row = list(map(str, combo)) + [filename]
        f.write(','.join(row) + '\n')