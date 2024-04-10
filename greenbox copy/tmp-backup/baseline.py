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
plt.figure(figsize=(4,5), dpi=300)

results = []
policies = []
with open("baseline.dat", "r") as f:
    for i, line in enumerate(f.readlines()):
        if i == 0:
            continue
        parsed_line = line.strip().split()
        raw = list(map(lambda x: float(x), parsed_line[1:]))
        results.append(raw)
        policies.append(parsed_line[0])
       

width = 0.2
gap = 0.25
for i in range(3):
    plt.bar((2+i)*gap, results[i], width=width, label=policies[i], color=colors[i], edgecolor="black", zorder=3)
    
x = np.arange(0,3)
plt.xticks([]) 
plt.xlabel('Policies')
plt.ylabel('Total Operational Carbon Footprint (gCO2e)')
    
plt.legend(ncol=2, bbox_to_anchor=(0.5, 1.2), loc='upper center', frameon=False, fontsize=11, columnspacing = 0.2)
plt.grid(which='major', axis='y', linestyle='--',linewidth=1)
plt.tight_layout()
plt.savefig("baseline.png")
