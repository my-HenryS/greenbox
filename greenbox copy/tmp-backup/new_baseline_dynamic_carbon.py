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


line_styles = ['-', '--', '-.', ':']
hatches = ["", "\\", "//", "||"]
colors = ['#ff796c', 'plum', '#95d0fc', 'gray']
line_colors = ['#ff796c', 'plum', '#23a8eb', 'gray', 'black']
markers = ['o', '*', 'v', '^']
plt.figure(figsize=(7,4), dpi=300)

results = []
policies = []
with open("data/dynamic_carbon.dat", "r") as f:
    for i, line in enumerate(f.readlines()):
        if i == 0:
            continue
        parsed_line = line.strip().split()
        raw = list(map(lambda x: float(x), parsed_line[1:]))
        results.append(raw)
        policies.append(parsed_line[0])

l = ["week1", "week2", "week3", "week4", "week5", "week6", "week7", "week8"]
x = np.arange(0,len(l))
# plt.xlabel('Time')
plt.ylabel('Carbon Footprint (kgCO2eq)')
plt.xticks(x, labels=l)

for i in range(2):
    plt.plot(x, results[i], line_styles[i % len(policies[i])], label=policies[i], color=colors[i], linewidth=2, marker=markers[i])
    
# plt.legend(ncol=2, bbox_to_anchor=(0.5, 1.21), loc='upper center', frameon=False, fontsize=11)
plt.legend(ncol=1, loc='upper left', frameon=False, fontsize=11)
plt.grid(which='major', axis='y', linestyle='--',linewidth=1)
plt.tight_layout()
plt.savefig("figure/new_baseline_dynamic_carbon.png")
plt.clf()