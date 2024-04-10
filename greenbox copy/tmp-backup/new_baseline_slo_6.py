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


line_styles = ['-', '--', '-.', ':', '-']
hatches = ["", "\\\\", "//", "--", "xx"]
# colors = ['#ff796c', 'plum', '#95d0fc', 'gray']
colors = ['#ff796c', 'plum', '#95d0fc', '#23a8eb', '#3d8c40', 'grey']
line_colors = ['#ff796c', 'plum', '#23a8eb', 'gray', 'black']
markers = ['o', '*', 'v', '^', 's']
plt.figure(figsize=(7,3.5), dpi=300)

results = []
policies = []
expt = "week2"
with open(f"data/new_baseline_slo_6_{expt}.dat", "r") as f:
# with open("data/new_baseline_slo_6_week7.dat", "r") as f:
    for i, line in enumerate(f.readlines()):
        if i == 0:
            continue
        parsed_line = line.strip().split()
        raw = list(map(lambda x: float(x), parsed_line[1:]))
        results.append(raw)
        policies.append(parsed_line[0])
policies = ["Centr-Global", "Centr-Graph", "Distr-Grid", "Distr-Battery", "GreenBox"]

l = ["100%", "95%", "90%", "80%", "70%"]
x = np.arange(0,len(l))
plt.xlabel('Low-Priority VM Uptime')
plt.ylabel('Carbon Footprint (kgCO2eq)')
plt.xticks(x, labels=l) 

width = 0.16
for i in range(len(policies)):
    plt.bar(x+(-2+i)*width, results[i], width=width, color=colors[i], edgecolor="black", hatch=hatches[i], zorder=3)

plt.legend(policies, ncol=3, bbox_to_anchor=(0.5, 1.25), loc='upper center', frameon=False, fontsize=10)
plt.grid(which='major', axis='y', linestyle='--',linewidth=1)
plt.tight_layout()
plt.savefig(f"figure/new_baseline_slo_bar_6_{expt}.png")
plt.savefig(f"figure/new_baseline_slo_bar_6_{expt}.pdf")
plt.clf()

x = [0,0.5,1,2,3]
plt.xticks([0,0.5,1,2,3], labels=l) 
for i in range(len(policies)):
    plt.plot(x, results[i], line_styles[i], label=policies[i], color=colors[i], linewidth=2, marker=markers[i])
    
# plt.legend(ncol=2, bbox_to_anchor=(0.5, 1.21), loc='upper center', frameon=False, fontsize=11)
plt.legend(ncol=2, loc='upper right', frameon=False, fontsize=10)
plt.grid(which='major', axis='y', linestyle='--',linewidth=1)
plt.tight_layout()
plt.savefig(f"figure/new_baseline_slo_6_{expt}.png")
plt.savefig(f"figure/new_baseline_slo_6_{expt}.pdf")
plt.clf()