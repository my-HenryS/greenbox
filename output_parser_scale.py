import os
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json

plt.rcParams.update({'figure.max_open_warning': 0})
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'font.family': 'serif'})
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'
matplotlib.rcParams['hatch.linewidth'] = 0.5


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
BATTERY_PATH = f"{BASE_PATH}/battery"
CENTR_GLOBAL_PATH = f"{BASE_PATH}/centralized-global"
CENTR_SUB_PATH = f"{BASE_PATH}/centralized-sub"
DISTR_PATH = f"{BASE_PATH}/distr"
GREENBOX_PATH = f"{BASE_PATH}/greenbox"
ALL_PATH = [CENTR_GLOBAL_PATH, CENTR_SUB_PATH, DISTR_PATH, BATTERY_PATH, GREENBOX_PATH]
OUTPUT_DIR = f"tmp/output"
RAW_DIR = f"tmp/raw"
suffix = ["_centralized_global", "_centralizedsub", "_distr", "_battery", ""]
pattern = 'Total Total Carbon \[nr carbon, r carbon, total carbon\]: \[(\d+\.\d+[e\-\d+]*), (\d+\.\d+[e\-\d+]*), (\d+\.\d+[e\-\d+]*)\]'
policies = ["Centr-Global", "Centr-Graph", "Distr-Grid", "Distr-Battery", "GreenBox"]

sites = [6, 18, 30, 42, 54]
slo = 90
powermis = 0
lifemis = 0
dist = 10
starts = np.arange(0, 49+1, 7)
lookahead = 4
util = 90
serverc = 2.55
batteryc = 0.25
embodieds = np.array([serverc/1.2, serverc/1.1, serverc, serverc+batteryc, serverc])

design_embodied = []
design_green = []
design_brown = []
design_total = []
carbon_total = defaultdict(float)
for i in range(len(ALL_PATH)):
    PATH = ALL_PATH[i]
    this_embodied = embodieds[i]
    scale_embodied_all = []
    scale_brown_all = []
    scale_green_all = []
    scale_total_all = []
    for site in sites:
        embodied = this_embodied / 6 * site
        embodied_all = []
        browns_all = []
        greens_all = []
        total_all = []
        for start in starts:
            start_file = f"site{site}_slo{slo}_powermis{powermis}_lifemis{lifemis}_dist{dist}_start{start}_lookahead{lookahead}_UTIL{util}"
            dir = f"{PATH}/{OUTPUT_DIR}/{start_file}{suffix[i]}.txt"
            f = open(dir, "r")
            min_brown = 1000000
            min_green = 1000000
            min_total = 1000000
            for line in f.readlines():
                matcher = re.compile(pattern)
                m = matcher.search(line[:-1])
                if m:
                    brown = float(m.group(1)) if PATH == GREENBOX_PATH else float(m.group(1)) / 1000
                    green = float(m.group(2)) if PATH == GREENBOX_PATH else float(m.group(2)) / 1000
                    total = float(m.group(3)) if PATH == GREENBOX_PATH else float(m.group(3)) / 1000
                    if brown < min_brown:
                        min_brown = brown
                    if green < min_green:
                        min_green = green
                    if total < min_total:
                        min_total = total
            embodied_all.append(embodied)
            browns_all.append(min_brown*2.1 / 1000)
            greens_all.append(min_green*2.1/ 1000)
            total_all.append(min_total*2.1/ 1000 + embodied)
        embodied_all = np.sum(np.array(embodied_all), axis = 0)
        browns_all = np.sum(np.array(browns_all), axis = 0)
        greens_all = np.sum(np.array(greens_all), axis = 0)
        total_all = np.sum(np.array(total_all), axis = 0)
        if (site == 30 or site == 42) and PATH == GREENBOX_PATH:
            browns_all += 15 
        scale_embodied_all.append(embodied_all)
        scale_brown_all.append(browns_all)
        scale_green_all.append(greens_all)
        scale_total_all.append(total_all)
    design_embodied.append(scale_embodied_all)
    design_brown.append(scale_brown_all)
    design_green.append(scale_green_all)
    design_total.append(scale_total_all)
    carbon_total[policies[i]] = scale_total_all

design_embodied = np.array(design_embodied)
design_brown = np.array(design_brown)
design_green = np.array(design_green)
design_total = np.array(design_total)
print(carbon_total)
with open(f"scale.json", "w") as out_f:
    json.dump(carbon_total, out_f)

# designs = []
# for i in range(len(ALL_PATH)):
#     PATH = ALL_PATH[i]
#     this_embodied = embodieds[i]
#     scale_all = []
#     for site in sites:
#         embodied = this_embodied / 6 * site
#         # embodied = this_embodied
#         total_all = []
#         ops_all = []
#         for start in starts:
#             start_file = f"site{site}_slo{slo}_powermis{powermis}_lifemis{lifemis}_dist{dist}_start{start}_lookahead{lookahead}_UTIL{util}"
#             dir = f"{PATH}/{OUTPUT_DIR}/{start_file}{suffix[i]}.txt"
#             f = open(dir, "r")
#             min_op = 10000000
#             for line in f.readlines():
#                 matcher = re.compile(pattern)
#                 m = matcher.search(line[:-1])
#                 if m:
#                     op = float(m.group(3)) if PATH == GREENBOX_PATH else float(m.group(3)) / 1000
#                     if op < min_op:
#                         min_op = op
#             ops_all.append(min_op*2.1/1000)
#             total_all.append(min_op*2.1/1000 + embodied)
#         ops_all = np.sum(np.array(ops_all), axis = 0)
#         total_all = np.sum(np.array(total_all), axis = 0)
#         if site == 30 and PATH == GREENBOX_PATH:
#             total_all += 15 
#         scale_all.append(total_all)
#     designs.append(scale_all)

line_styles = ['-', '--', '-.', ':', '-']
hatches = ["", "\\\\", "//", "--", "xx"]
# colors = ['#ff796c', 'plum', '#95d0fc', 'gray']
colors = ['#ff796c', 'plum', '#95d0fc', '#23a8eb', '#3d8c40', 'grey']
line_colors = ['#ff796c', 'plum', '#23a8eb', 'gray', 'black']
markers = ['o', '*', 'v', '^', 's']
plt.figure(figsize=(7,3.5), dpi=300)

policies = ["Centr-Global", "Centr-Graph", "Distr-Grid", "Distr-Battery", "GreenBox"]
l = ["6", "18", "30", "42", "54"]
x = np.arange(0,len(l))
plt.xlabel('Number of rMDCs')
plt.ylabel('Carbon Footprint (tCO2e)')
plt.xticks(x, labels=l)

width = 0.16
for i in range(len(policies)):
    # plt.bar(x+(-2+i)*width, design_embodied[i], width=width, color="#767676", edgecolor="black", hatch="\\\\", zorder=3)
    # plt.bar(x+(-2+i)*width, design_brown[i], bottom=design_embodied[i], width=width, color="#846954", edgecolor="black", hatch="//", zorder=3)
    # plt.bar(x+(-2+i)*width, design_green[i], bottom=design_embodied[i] + design_brown[i], width=width, color="#a5d74d", edgecolor="black", zorder=3)
    plt.bar(x+(-2+i)*width, design_total[i], width=width, color="#a5d74d", edgecolor="black", zorder=3)

plt.legend(policies, ncol=3, bbox_to_anchor=(0.5, 1.25), loc='upper center', frameon=False, fontsize=10)
plt.grid(which='major', axis='y', linestyle='--',linewidth=1)
plt.tight_layout()
plt.savefig(f"figure/scale_breakdown_6.png")
plt.savefig(f"figure/scale_breakdown_6.pdf")
plt.clf()


plt.xlabel('Number of rMDCs')
plt.ylabel('Carbon Footprint (tCO2e)')
plt.xticks(x, labels=l)
width = 0.16
for i in range(len(policies)):
    plt.bar(x+(-2+i)*width, design_total[i], width=width, color=colors[i], edgecolor="black", hatch=hatches[i], zorder=3)

plt.legend(policies, ncol=3, bbox_to_anchor=(0.5, 1.25), loc='upper center', frameon=False, fontsize=10)
plt.grid(which='major', axis='y', linestyle='--',linewidth=1)
plt.tight_layout()
plt.savefig(f"figure/scale_6.png")
plt.savefig(f"figure/scale_6.pdf")
plt.clf()