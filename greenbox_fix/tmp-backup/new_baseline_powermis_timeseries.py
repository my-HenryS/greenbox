import matplotlib.pyplot as plt
import numpy as np
import matplotlib.backends.backend_pdf
import matplotlib
import pandas as pd

TEXT_ONLY = False

plt.rcParams.update({'figure.max_open_warning': 0})
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'font.family': 'serif'})
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'
matplotlib.rcParams['hatch.linewidth'] = 0.5


line_styles = ['-', '--', '-.', ':']
hatches = ["", "\\", "//", "||"]
colors = ['#ff796c', 'plum', '#95d0fc', 'gray']
line_colors = ['#ff796c', 'plum', '#23a8eb', 'gray', 'black']
markers = ['o', '*', 'v', '^']
plt.figure(figsize=(7,3.5), dpi=300)

results = []
policies = []
with open("data/new_baseline_powermis_6_timeseries_-20_20.dat", "r") as f:
    for i, line in enumerate(f.readlines()):
        if i == 0:
            continue
        parsed_line = line.strip().split()
        raw = list(map(lambda x: float(x), parsed_line[1:]))
        results.append(raw)
        policies.append(parsed_line[0])

x = np.arange(0,len(results[0]))
plt.xlabel('Power Supply Misprediction Ratio')
plt.ylabel('Power Grid Energy Usage (Wh)')
# plt.xticks(x, labels=["-40%", "-30%", "-20%", "-10%", "0%", "10%", "20%", "30%", "40%"]) 
plt.xticks(x, labels=["-20%", "-15%", "-10%", "-5%", "0%", "5%", "10%", "15%","20%"]) 

for i in range(len(results)):
    plt.plot(x, results[i], line_styles[0], label=policies[i], linewidth=2, marker=markers[0])
# plt.plot(x, results[0], line_styles[0], label=policies[0], color=colors[0], linewidth=2, marker=markers[0])
    
# plt.legend(ncol=3, bbox_to_anchor=(0.5, 1.2), loc='upper center', frameon=False, fontsize=11)
plt.grid(which='major', axis='y', linestyle='--',linewidth=1)
plt.tight_layout()
plt.savefig("power_misprediction.png")
plt.clf()