# File: split_param_list_by_mode.py

with open("x2_param_list_x0_2.txt") as f:
    lines = f.readlines()

header = lines[0]
entries = lines[1:]

# Buckets by mode
buckets = {
    "qq": [],
    "cq": [],
    "cc": []
}

# Distribute entries
for line in entries:
    mode = line.split(',')[0].strip()
    if mode in buckets:
        buckets[mode].append(line)

# Write files
for mode, lines in buckets.items():
    with open(f"param_list_{mode}_x0_2.txt", "w") as f:
        f.write(header)
        f.writelines(lines)