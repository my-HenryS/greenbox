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


fig, axs = plt.subplots(1, 2, figsize=(10,3.5), dpi=1200)

# site24_slo90_powermis0_lifemis0_dist40_centralized
weeks = [1,2]
for count in range(len(weeks)):
    week = weeks[count]
    # data_dir = f"data/new_baseline_separate_slo90dist10_6/week{week}"
    data_dir = f"data/new_baseline_separate_slo90dist10_new_6/week{week}"
    subgraphs = 2
    ax = axs[count]
    all_embodied = []
    all_clean = []
    all_brown = []
    for sub in range(subgraphs):
        sub_id = sub+1
        results = []
        policies = []
        with open(f"{data_dir}/subgraph{sub_id}.dat", "r") as f:
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

    x = np.arange(0,subgraphs)
    width = 0.18
    gap = 0.4
    # sublabels = ["Centralized", "Distributed-PowerGrid", "Distributed-Battery", "GreenBox"]
    sublabels = ["Distr-Grid", "Distr-Battery", "GreenBox"]
    all_x = np.zeros(subgraphs*len(sublabels))
    for i in range(len(sublabels)):
        # normal one
        ax.bar(x+(i-1.5)*width, all_embodied[:,i], width=width, color="#DFDFD4", edgecolor="black", hatch="\\", zorder=3)
        ax.bar(x+(i-1.5)*width, all_brown[:,i], bottom=all_embodied[:,i], width=width, color="#846954", edgecolor="black", hatch="/", zorder=3)
        ax.bar(x+(i-1.5)*width, all_clean[:,i], bottom=all_embodied[:,i]+all_brown[:,i], width=width, color="#a5d74d", edgecolor="black", zorder=3)
        # ax.bar(x+(i-1.5)*width, all_embodied[:,i], width=width, color="#DFDFD4", label = "Embodied Carbon", edgecolor="black", hatch="\\\\", zorder=3)
        # ax.bar(x+(i-1.5)*width, all_brown[:,i], bottom=all_embodied[:,i], width=width, label = "Brown Operational Carbon", color="#846954", edgecolor="black", hatch="//", zorder=3)
        # ax.bar(x+(i-1.5)*width, all_clean[:,i], bottom=all_embodied[:,i]+all_brown[:,i], width=width, label="Green Operational Carbon", color="#a5d74d", edgecolor="black", zorder=3)

        if i == 0:
            all_x[i*subgraphs:(i+1)*subgraphs] = x+(i-1.8)*width
        elif i == 1:
            all_x[i*subgraphs:(i+1)*subgraphs] = x+(i-1.8)*width
        else:
            all_x[i*subgraphs:(i+1)*subgraphs] = x+(i-1.5)*width

    sublabels = [sublabels[0]]*subgraphs+[sublabels[1]]*subgraphs+[sublabels[2]]*subgraphs
    ax.tick_params(axis=u'x', which=u'both',length=0)
    # ax.set_xticks(all_x, labels=sublabels, fontsize=12, rotation=15) 
    ax.set_xticks(x-0.5*width, labels=["Subgraph1", "Subgraph2"], fontsize=16) 
    ax.set_ylabel("Carbon Footprint (kgCO2eq)", fontsize=14)
    # for i in range(subgraphs):
    loc = -90
    if week == weeks[1]:
        loc = -270
    ax.text(0+width, loc, f"Week{week}", weight='bold', fontsize=18)

    # plt.legend(ncol=3, bbox_to_anchor=(0.5, 1.2), loc='upper right', frameon=False, fontsize=11)
    # plt.legend(["Centralized-Embodied", "Centralized-Brown", "Centralized-Green", 
    #             "Distributed Powergrid-Embodied", "Distributed Powergrid-Brown", "Distributed Powergrid-Green", 
    #             "Distributed Battery-Embodied", "Distributed Battery-Brown", "Distributed Battery-Green", 
    #             "Greenbox-Embodied", "Greenbox-Brown", "Greenbox-Green"], ncol=2, loc='upper right', frameon=False, fontsize=10)
    # ax.legend(["Embodied Carbon", "Brown Operational Carbon", "Green Operational Carbon"], ncol=1, loc='upper left', frameon=False, fontsize=10)
    # ax.legend(ncol=1, loc='upper left', frameon=False, fontsize=10)
    ax.grid(which='major', axis='y', linestyle='--',linewidth=1)
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
axs[0].annotate("Distr-Grid", xy=(-width*3/2,220), xytext=(-width*3/2-0.15, 320), arrowprops=dict(arrowstyle= '->',color='black', lw=2), fontsize=16)
axs[0].annotate("Distr-Battery", xy=(-0.1,210), xytext=(-0.2, 260), arrowprops=dict(arrowstyle= '->',color='black', lw=2), fontsize=16)
axs[0].annotate("Greenbox", xy=(0.12,110), xytext=(0.05, 160), arrowprops=dict(arrowstyle= '->',color='black', lw=2), fontsize=16)
fig.legend(["Embodied Carbon", "Brown Operational Carbon", "Green Operational Carbon"],ncol=3, bbox_to_anchor=(0.5, 1.15), loc='upper center', fontsize=14)
fig.tight_layout()
plt.savefig(f"./figure/new_baseline_separate_week{week}_6.png",bbox_inches='tight')
plt.savefig(f"./figure/new_baseline_separate_week{week}_6.pdf",bbox_inches='tight')
plt.clf()
plt.cla()
