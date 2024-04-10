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


def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

brown_color = ["#745961", "#eab768"]
green_color = ["#2EB62C", "#C5E8B7"]
embodied_color = ["#454545", "#707070"]
embodied_colors = get_color_gradient(embodied_color[0], embodied_color[1], 4)
brown_colors = get_color_gradient(brown_color[0], brown_color[1], 4)
green_colors = get_color_gradient(green_color[0], green_color[1], 4)

plt.rcParams.update({'figure.max_open_warning': 0})
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'font.family': 'serif'})
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'
matplotlib.rcParams['hatch.linewidth'] = 0.5


line_styles = ['-', '--', '-.', ':']
hatches = ["", "\\\\\\\\", "////", "oooo"]
colors = ['gold', '#ff796c', 'plum', '#95d0fc', 'royalblue']
line_colors = ['#ff796c', 'plum', '#23a8eb', 'gray', 'black']
markers = ['o', '*', 'v', '^']
fig, ax = plt.subplots(figsize=(10,3.5), dpi=1200)

# data_dir = "data/new_baseline_separate_timestart_6"
data_dir = "data/new_baseline_separate_timestart_new_6"
times = 8
all_embodied = []
all_clean = []
all_brown = []
np.random.seed(1)
time_ids = np.arange(1, times+1)
# np.random.shuffle(time_ids)
for time in range(times):
    time_id = time_ids[time]
    results = []
    policies = []
    with open(f"{data_dir}/time{time_id}.dat", "r") as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            parsed_line = line.strip().split()
            raw = list(map(lambda x: float(x), parsed_line[1:]))
            results.append(raw)
            policies.append(parsed_line[0])
    embodied = [i[0] for i in results]
    brown_op = [i[1] for i in results]
    clean_op = [i[2] for i in results]
    all_embodied.append(embodied)
    all_brown.append(brown_op)
    all_clean.append(clean_op)
all_embodied = np.array(all_embodied)
all_brown = np.array(all_brown)
all_clean = np.array(all_clean)

timelabels = ["Centr-Global", "Centr-Graph", "Distr-Grid", "Distr-Battery", "GreenBox"]
x = np.arange(0,times)
all_x = np.zeros(times*len(timelabels))
width = 0.14
gap = 0.4
for i in range(5):
    # normal one
    ax.bar(x+(i-2)*width, all_embodied[:,i], width=width, color="#DFDFD4", edgecolor="black", hatch="\\\\", zorder=3)
    ax.bar(x+(i-2)*width, all_brown[:,i], bottom=all_embodied[:,i], width=width, color="#846954", edgecolor="black", hatch="//", zorder=3)
    ax.bar(x+(i-2)*width, all_clean[:,i], bottom=all_embodied[:,i]+all_brown[:,i], width=width, color="#a5d74d", edgecolor="black", zorder=3)

    # no hatch one
    # ax.bar(x+(i-1.5)*width, all_embodied[:,i], width=width, color="#DFDFD4", edgecolor="black", hatch=hatches[i], zorder=3)
    # ax.bar(x+(i-1.5)*width, all_brown[:,i], bottom=all_embodied[:,i], width=width, color="#846954", edgecolor="black", hatch=hatches[i], zorder=3)
    # ax.bar(x+(i-1.5)*width, all_clean[:,i], bottom=all_embodied[:,i]+all_brown[:,i], width=width, color="#a5d74d", edgecolor="black", hatch=hatches[i], zorder=3)

    # gradient one
    # ax.bar(x+(i-1.5)*width, all_embodied[:,i], width=width, color=embodied_colors[i] ,edgecolor="black", hatch="\\\\", zorder=3)
    # ax.bar(x+(i-1.5)*width, all_brown[:,i], bottom=all_embodied[:,i], width=width, color=brown_colors[i], edgecolor="black", hatch="//", zorder=3)
    # ax.bar(x+(i-1.5)*width, all_clean[:,i], bottom=all_embodied[:,i]+all_brown[:,i], width=width, color=green_colors[i], edgecolor="black", zorder=3)

    if i == 0:
        all_x[i*times:(i+1)*times] = x+(i-3.4)*width
    elif i == 1:
        all_x[i*times:(i+1)*times] = x+(i-2.9)*width
    elif i == 2:
        all_x[i*times:(i+1)*times] = x+(i-2.7)*width
    elif i == 3:
        all_x[i*times:(i+1)*times] = x+(i-2.9)*width
    elif i == 4:
        all_x[i*times:(i+1)*times] = x+(i-2.2)*width

labels = []
for l in timelabels:
    labels += [l]*times
ax.tick_params(axis=u'x', which=u'both',length=0)
ax.set_xticks(all_x, labels=labels, fontsize=8, rotation=30) 
# ax.set_xticks(x, labels=["Subgraph1", "Subgraph2", "Subgragh3", "Subgraph4", "Subgraph5", "Subgraph6", "Subgragh7", "Subgraph8"], fontsize=11) 
# ax.set_xticks(x, labels=["Centralized", "Distributed-PowerGrid", "Distributed-Battery", "GreenBox"], rotation = 15, fontsize=11) 
ax.set_ylabel("Total Carbon Footprint (kgCO2eq)", fontsize=11)
for i in range(times):
    ax.text(x[i]-1.8*width, -2400, f"Week{i+1}", weight='bold', fontsize=10.5)

# plt.legend(ncol=3, bbox_to_anchor=(0.5, 1.2), loc='upper right', frameon=False, fontsize=11)
# plt.legend(["Centralized-Embodied", "Centralized-Brown", "Centralized-Green", 
#             "Distributed Powergrid-Embodied", "Distributed Powergrid-Brown", "Distributed Powergrid-Green", 
#             "Distributed Battery-Embodied", "Distributed Battery-Brown", "Distributed Battery-Green", 
#             "Greenbox-Embodied", "Greenbox-Brown", "Greenbox-Green"], ncol=2, loc='upper right', frameon=False, fontsize=10)
ax.legend(["Embodied Carbon", "Brown Operational Carbon", "Green Operational Carbon"], ncol=1, loc='upper left', frameon=False, fontsize=11)
ax.grid(which='major', axis='y', linestyle='--',linewidth=1)
fig.tight_layout()
plt.savefig("./figure/new_baseline_separate_starttime_new_6.png")
plt.savefig("./figure/new_baseline_separate_starttime_new_6.pdf")
plt.clf()
plt.cla()


# results = []
# policies = []
# with open("new_baseline_cost_24.dat", "r") as f:
#     for i, line in enumerate(f.readlines()):
#         if i == 0:
#             continue
#         parsed_line = line.strip().split()
#         raw = list(map(lambda x: float(x), parsed_line[1:]))
#         results.append(raw)
#         policies.append(parsed_line[0])
# Million = 10**6
# server = np.array([i[0] for i in results]) / Million
# battery = np.array([i[1] for i in results]) / Million
# transmission = np.array([i[2] for i in results]) / Million
# network = np.array([i[3] for i in results]) / Million
# manufacturing = np.array([i[4] for i in results]) / Million

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
# top=8
# plt.ylim(bottom=0, top=top)
# plt.text(0-0.25, top +0.03*top, "$152.9M")

# plt.legend(["Battery",  "Network", "Manufacturing", "Server", "Transmission"], ncol=1, loc='upper right', frameon=False, fontsize=10)
# plt.grid(which='major', axis='y', linestyle='--',linewidth=1)
# plt.tight_layout()
# plt.savefig("new_baseline_cost.png")
# plt.clf()
# plt.cla()


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