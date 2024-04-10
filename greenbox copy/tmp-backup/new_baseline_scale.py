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
hatches = ["", "\\\\", "//", "--", "xx"]
colors = ['#ff796c', 'plum', '#95d0fc', '#23a8eb', '#3d8c40', 'grey']
line_colors = ['#ff796c', 'plum', '#23a8eb', 'gray', 'black']
markers = ['o', '*', 'v', '^']
fig, ax = plt.subplots(figsize=(6,3.5), dpi=1200)

data_dir = "data/new_baseline_separate_scale"
# num_sites = [6, 12, 24, 36, 48, 60, 72]
num_sites = [6, 12, 24, 36, 48]
all_carbon = []
for site in num_sites:
    results = []
    policies = []
    with open(f"{data_dir}/site{site}.dat", "r") as f:
        for i, line in enumerate(f.readlines()):
            if i == 0 or i > len(num_sites):
                continue
            parsed_line = line.strip().split()
            raw = list(map(lambda x: float(x), parsed_line[1:]))
            results.append(raw)
            policies.append(parsed_line[0])
    total_carbon = [sum(i) for i in results]
    all_carbon.append(total_carbon)
all_carbon = np.array(all_carbon)

timelabels = ["Centr-Global", "Centr-Graph", "Distr-Grid", "Distr-Battery", "GreenBox"]
x = np.arange(len(num_sites))
width = 0.14
gap = 0.4
for i in range(len(timelabels)):
    # normal one
    ax.bar(x+(i-2)*width, all_carbon[:,i], width=width, color=colors[i], edgecolor="black", hatch=hatches[i], zorder=3)

    # if i == 0:
    #     all_x[i*times:(i+1)*times] = x+(i-3.4)*width
    # elif i == 1:
    #     all_x[i*times:(i+1)*times] = x+(i-2.9)*width
    # elif i == 2:
    #     all_x[i*times:(i+1)*times] = x+(i-2.7)*width
    # elif i == 3:
    #     all_x[i*times:(i+1)*times] = x+(i-2.9)*width
    # elif i == 4:
    #     all_x[i*times:(i+1)*times] = x+(i-2.2)*width


ax.tick_params(axis=u'x', which=u'both',length=0)
ax.set_xticks(x, labels=num_sites, fontsize=11) 
ax.set_ylabel("Carbon Footprint (kgCO2eq)", fontsize=11)
ax.set_xlabel("Number of sites")
# for i in range(len(num_sites)):
#     ax.text(x[i]-1.8*width, -2400, f"Week{i+1}", weight='bold', fontsize=10.5)

ax.legend(timelabels, ncol=3, bbox_to_anchor=(0.5, 1.25), loc='upper center', frameon=False, fontsize=11)
ax.grid(which='major', axis='y', linestyle='--',linewidth=1)
fig.tight_layout()
plt.savefig("./figure/new_baseline_scale.png")
plt.savefig("./figure/new_baseline_scale.pdf")
plt.clf()
plt.cla()