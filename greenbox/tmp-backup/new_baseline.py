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
colors = ['gold', '#ff796c', 'plum', '#95d0fc', 'royalblue']
line_colors = ['#ff796c', 'plum', '#23a8eb', 'gray', 'black']
markers = ['o', '*', 'v', '^']
plt.figure(figsize=(6,5), dpi=1200)

results = []
policies = []
with open("new_baseline_24.dat", "r") as f:
    for i, line in enumerate(f.readlines()):
        if i == 0:
            continue
        parsed_line = line.strip().split()
        raw = list(map(lambda x: float(x), parsed_line[1:]))
        results.append(raw)
        policies.append(parsed_line[0])
embodied = np.array([i[0] for i in results])
clean_op = np.array([i[1] for i in results])
brown_op = np.array([i[2] for i in results])

x = np.arange(0,4)
width = 0.4
gap = 0.25
plt.bar(policies, embodied, width=width, color="#DFDFD4", edgecolor="black", hatch="\\\\", zorder=3)
plt.bar(policies, brown_op, bottom=embodied, width=width, color="#846954", edgecolor="black", hatch="//", zorder=3)
plt.bar(policies, clean_op, bottom=embodied+brown_op, width=width, color="#a5d74d", edgecolor="black", zorder=3)
    
plt.xticks(x, labels=["Centralized", "Distributed-PowerGrid", "Distributed-Battery", "GreenBox"], rotation = 15, fontsize=11) 
plt.ylabel("Total Carbon Footprint (kgCO2eq)")

# plt.legend(ncol=3, bbox_to_anchor=(0.5, 1.2), loc='upper right', frameon=False, fontsize=11)
plt.legend(["Embodied Carbon", "Brown Operational Carbon", "Green Operational Carbon"], ncol=1, loc='upper right', frameon=False, fontsize=10)
plt.grid(which='major', axis='y', linestyle='--',linewidth=1)
plt.tight_layout()
plt.savefig("new_baseline.png")
plt.clf()
plt.cla()

results = []
policies = []
with open("new_baseline_cost_24.dat", "r") as f:
    for i, line in enumerate(f.readlines()):
        if i == 0:
            continue
        parsed_line = line.strip().split()
        raw = list(map(lambda x: float(x), parsed_line[1:]))
        results.append(raw)
        policies.append(parsed_line[0])
Million = 10**6
server = np.array([i[0] for i in results]) / Million
battery = np.array([i[1] for i in results]) / Million
transmission = np.array([i[2] for i in results]) / Million
network = np.array([i[3] for i in results]) / Million
manufacturing = np.array([i[4] for i in results]) / Million

x = np.arange(0,4)
width = 0.4
gap = 0.25
plt.bar(policies, battery, width=width, color=colors[2], edgecolor="black", hatch="oo", zorder=3)
plt.bar(policies, network, bottom=battery, width=width, color=colors[3], edgecolor="black", hatch="||", zorder=3)
plt.bar(policies, manufacturing, bottom=network+battery, width=width, color=colors[0], edgecolor="black", hatch="\\\\", zorder=3)
plt.bar(policies, server, bottom=network+battery+manufacturing, width=width, color=colors[1], edgecolor="black", hatch="//", zorder=3)
plt.bar(policies, transmission, bottom=manufacturing+server+battery+network, width=width, color=colors[4], edgecolor="black", zorder=3)

plt.xticks(x, labels=["Centralized", "Distributed-PowerGrid", "Distributed-Battery", "GreenBox"], rotation = 15, fontsize=11) 
plt.ylabel("Cost (M$)")
# plt.yscale("log")
top=8
plt.ylim(bottom=0, top=top)
plt.text(0-0.25, top +0.03*top, "$152.9M")

plt.legend(["Battery",  "Network", "Manufacturing", "Server", "Transmission"], ncol=1, loc='upper right', frameon=False, fontsize=10)
plt.grid(which='major', axis='y', linestyle='--',linewidth=1)
plt.tight_layout()
plt.savefig("new_baseline_cost.png")
plt.clf()
plt.cla()


# results = []
# policies = []
# with open("new_baseline_noamertize.dat", "r") as f:
#     for i, line in enumerate(f.readlines()):
#         if i == 0:
#             continue
#         parsed_line = line.strip().split()
#         raw = list(map(lambda x: float(x), parsed_line[1:]))
#         results.append(raw)
#         policies.append(parsed_line[0])
# server = np.array([i[0] for i in results]) / 1000000
# battery = np.array([i[1] for i in results]) / 1000000
# transmission = np.array([i[2] for i in results]) / 1000000
# network = np.array([i[3] for i in results]) / 1000000
# manufacturing = np.array([i[4] for i in results]) / 1000000

# x = np.arange(0,4)
# width = 0.4
# gap = 0.25
# plt.bar(policies, battery, width=width, color=colors[2], edgecolor="black", hatch="oo", zorder=3)
# plt.bar(policies, network, bottom=battery, width=width, color=colors[3], edgecolor="black", hatch="||", zorder=3)
# plt.bar(policies, manufacturing, bottom=network+battery, width=width, color=colors[0], edgecolor="black", hatch="\\\\", zorder=3)
# plt.bar(policies, server, bottom=network+battery+manufacturing, width=width, color=colors[1], edgecolor="black", hatch="//", zorder=3)
# plt.bar(policies, transmission, bottom=manufacturing+server+battery+network, width=width, color=colors[4], edgecolor="black", zorder=3)

# plt.xticks(x, labels=["Centralized", "Distributed-PowerGrid", "Distributed-Battery", "GreenBox"], rotation = 15, fontsize=11) 
# plt.ylabel("Cost (M$)")
# # plt.yscale("log")
# top=5
# plt.ylim(bottom=0, top=top)
# plt.text(0-0.25, top +0.03*top, "$37.8M")

# plt.legend(["Battery",  "Network", "Manufacturing", "Server", "Transmission"], ncol=1, loc='upper right', frameon=False, fontsize=10)
# plt.grid(which='major', axis='y', linestyle='--',linewidth=1)
# plt.tight_layout()
# plt.savefig("new_baseline_cost_noamertize.png")