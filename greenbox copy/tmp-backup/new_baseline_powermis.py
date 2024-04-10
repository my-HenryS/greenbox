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
hatches = ["", "\\\\", "//", "--", "xx"]
# colors = ['#ff796c', 'plum', '#95d0fc', 'grey']
colors = ['#ff796c', 'plum', '#95d0fc', '#23a8eb', '#3d8c40', 'grey']
line_colors = ['#ff796c', 'plum', '#23a8eb', 'gray', 'black']
markers = ['P', '*', 'v', '^']
plt.figure(figsize=(6,3.5), dpi=300)

results = []
policies = []
zero = []
five = []
ten = []
fifteen = []
twenty = []
with open("data/new_baseline_powermis_6_week2.dat", "r") as f:
    for i, line in enumerate(f.readlines()):
        if i == 0 or i > 6:
            continue
        parsed_line = line.strip().split()
        raw = list(map(lambda x: float(x), parsed_line[1:]))
        results.append(raw)
        policies.append(parsed_line[0])
zero = [i[0] for i in results]
five = [i[1] for i in results]
ten = [i[2] for i in results]
fifteen = [i[3] for i in results]
twenty = [i[4] for i in results]

ll = len(zero)
x = np.arange(ll)
xticks = ['0%', '10%', '20%', '30%', '40%', '50%']
# xticks = policies
plt.xlabel('Low-priority VM Percentage')
plt.ylabel('Carbon Footprint (kgCO2eq)')
plt.xticks(x, labels=xticks) 

size = 4
width = 0.16
plt.bar(x+(-1.5*width), zero, width=width, color=colors[0], edgecolor="black", hatch=hatches[0], zorder=3)
plt.bar(x+(-0.5*width), five, width=width, color=colors[1], edgecolor="black", hatch=hatches[1], zorder=3)
plt.bar(x+(0.5*width), ten, width=width, color=colors[2], edgecolor="black", hatch=hatches[2], zorder=3)
plt.bar(x+(1.5*width), fifteen, width=width, color=colors[3], edgecolor="black", hatch=hatches[3], zorder=3)
# plt.bar(x+(2*width), twenty, width=width, color=colors[4], edgecolor="black", hatch=hatches[4], zorder=3)
# plt.plot(x, results[0], '-', label=policies[0], color='black', linewidth=2, marker='o')
# for i in range(1, size+1):
#     plt.plot(x, results[i], '--', label=policies[i], color=colors[i % len(colors)], linewidth=2, marker=markers[i % len(markers)])
# for i in range(1, size+1):
#     idx = i + size  
#     plt.plot(x, results[idx], ':', label=policies[idx], color=colors[i % len(colors)], linewidth=2, marker=markers[i % len(markers)])

# plt.legend(['0%','5%','10%','15%','20%'], ncol=5, bbox_to_anchor=(0.5, 1.21), loc='upper center', frameon=False, fontsize=11)
plt.legend(['0%','5%','10%','15%'], ncol=4, bbox_to_anchor=(0.5, 1.21), loc='upper center', frameon=False, fontsize=11)
plt.grid(which='major', axis='y', linestyle='--',linewidth=1)
plt.tight_layout()
plt.savefig("figure/new_baseline_powermis.png")
plt.clf()