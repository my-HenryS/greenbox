import enum
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict
import pickle
import sys
from glob import glob
import networkx as nx
from matplotlib.animation import FuncAnimation
import matplotlib.backends.backend_pdf
import matplotlib
from networkx.drawing.nx_agraph import to_agraph 
import img2pdf
import os
import pandas as pd
from PyPDF2 import PdfMerger
import glob
import math

TEXT_ONLY = False

plt.rcParams.update({'figure.max_open_warning': 0})
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'font.family': 'serif'})
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'
matplotlib.rcParams['hatch.linewidth'] = 0.5


# line_styles = ['-', '--', '-.', ':']
line_styles = ['-', '--', '-.', ':']
hatches = ["", "\\", "//", "||"]
# colors = ['#ff796c', 'plum', '#95d0fc', 'grey']
colors = ['#ff796c', 'plum', '#95d0fc', '#23a8eb', '#3d8c40', 'grey']
line_colors = ['#ff796c', 'plum', '#23a8eb', 'gray', 'black']
# markers = ['P', '*', 'v', '^']
markers = ['o', 'P', '*', 'v', '^']
plt.figure(figsize=(6,3.5), dpi=300)

results = []
policies = []
policies = ["0%", "10%", "20%", "30%", "40%", "50%"]
with open("data/new_baseline_lookahead.dat", "r") as f:
    for i, line in enumerate(f.readlines()):
        if i == 0 or i > len(policies):
            continue
        parsed_line = line.strip().split()
        raw = list(map(lambda x: float(x), parsed_line[1:]))
        results.append(raw)
        # policies.append(parsed_line[0])
# policies = ["0%", "20%", "40%", "60%"]

x = np.arange(len(results[0]))
xticks = x.astype(str)
xticks[0] = '0 (greedy)'
plt.xlabel('Power Prediction Hour(s)')
plt.ylabel('Carbon Footprint (kgCO2eq)')
plt.xticks(x, labels=xticks) 

for i in range(len(results)):
    plt.plot(x, results[i], '-', label=policies[i], color=colors[i % len(colors)], linewidth=2, marker=markers[i % len(markers)])

# size = 4
# plt.plot(x, results[0], '-', label=policies[0], color='black', linewidth=2, marker='o')
# for i in range(1, size+1):
#     plt.plot(x, results[i], '--', label=policies[i], color=colors[i % len(colors)], linewidth=2, marker=markers[i % len(markers)])
# for i in range(1, size+1):
#     idx = i + size
#     plt.plot(x, results[idx], ':', label=policies[idx], color=colors[i % len(colors)], linewidth=2, marker=markers[i % len(markers)])

# plt.legend(ncol=2, bbox_to_anchor=(0.5, 1.21), loc='upper center', frameon=False, fontsize=11)
plt.legend(ncol=2, loc='upper right', frameon=False, fontsize=11)
plt.grid(which='major', axis='y', linestyle='--',linewidth=1)
plt.tight_layout()
plt.savefig("figure/new_baseline_lookahead.png")
plt.clf()