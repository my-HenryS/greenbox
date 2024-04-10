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
policies = ["GreenBox-noPrunning", "GreenBox-noSubgraph", "GreenBox-noMIP", "GreenBox"]

site = 6
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
embodied = embodieds[-1]

no_site_all = []
no_sub_all = []
no_mip_all = []
greenbox_all = []
carbon_total = defaultdict(float)
for start in starts:
    start_file = f"site{site}_slo{slo}_powermis{powermis}_lifemis{lifemis}_dist{dist}_start{start}_lookahead{lookahead}_UTIL{util}"
    dir = f"{GREENBOX_PATH}/{OUTPUT_DIR}/{start_file}_step.txt"
    count = 0
    f = open(dir, "r")
    for line in f.readlines():
        matcher = re.compile(pattern)
        m = matcher.search(line[:-1])
        if m:
            count += 1                
            total = float(m.group(3))
            if count == 1:
                no_site = total
            elif count == 2:
                no_sub = total
            elif count == 3:
                no_mip = total
            elif count == 4:
                greenbox = total
    # no_site_all.append(no_site * 2.1 / 1000 )
    # no_sub_all.append(no_sub * 2.1 / 1000)
    # no_mip_all.append(no_mip * 2.1 / 1000)
    # greenbox_all.append(greenbox * 2.1 / 1000)
    no_site_all.append(no_site * 2.1 / 1000 + embodied)
    no_sub_all.append(no_sub * 2.1 / 1000 + embodied)
    no_mip_all.append(no_mip * 2.1 / 1000 + embodied)
    greenbox_all.append(greenbox * 2.1 / 1000 + embodied)
    
no_site_all = np.sum(np.array(no_site_all), axis = 0)
no_sub_all = np.sum(np.array(no_sub_all), axis = 0)
no_mip_all = np.sum(np.array(no_mip_all), axis = 0)
greenbox_all = np.sum(np.array(greenbox_all), axis = 0)
step_all = [no_site_all, no_sub_all, no_mip_all, greenbox_all]
for i in range(len(step_all)):
    carbon_total[policies[i]] = step_all[i]

with open(f"step.json", "w") as out_f:
    json.dump(carbon_total, out_f)

line_styles = ['-', '--', '-.', ':', '-']
hatches = ["", "\\\\", "//", "--", "xx"]
# colors = ['#ff796c', 'plum', '#95d0fc', 'gray']
colors = ['#ff796c', 'plum', '#95d0fc', '#23a8eb', '#3d8c40', 'grey']
line_colors = ['#ff796c', 'plum', '#23a8eb', 'gray', 'black']
markers = ['o', '*', 'v', '^', 's']
plt.figure(figsize=(7,3.5), dpi=300)

x = np.arange(0,len(policies))
plt.xlabel('Policies')
plt.ylabel('Carbon Footprint (tCO2e)')
plt.xticks(x, labels=policies)

width = 0.5
plt.bar(x, step_all, width=width, color="#a5d74d", edgecolor="black", zorder=3)

# plt.legend(policies, ncol=3, bbox_to_anchor=(0.5, 1.25), loc='upper center', frameon=False, fontsize=10)
plt.grid(which='major', axis='y', linestyle='--',linewidth=1)
plt.tight_layout()
plt.savefig(f"figure/step.png")
plt.savefig(f"figure/step.pdf")
plt.clf()