import os
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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

site = 6
slo = 90
powermis = 0
lifemis = 0
dist = 10
starts = np.arange(0, 49+1, 7)
lookahead = 4
utils = [40,50,60,70,80,90,100]
serverc = 2.55
batteryc = 0.25
embodieds = np.array([serverc/1.2, serverc/1.1, serverc, serverc+batteryc, serverc])
designs = []
for i in range(len(ALL_PATH)):
    PATH = ALL_PATH[i]
    embodied = embodieds[i]
    util_all = []
    for util in utils:
        total_all = []
        ops_all = []
        for start in starts:
            start_file = f"site{site}_slo{slo}_powermis{powermis}_lifemis{lifemis}_dist{dist}_start{start}_lookahead{lookahead}_UTIL{util}"
            dir = f"{PATH}/{OUTPUT_DIR}/{start_file}{suffix[i]}.txt"
            f = open(dir, "r")
            min_op = 10000000
            for line in f.readlines():
                matcher = re.compile(pattern)
                m = matcher.search(line[:-1])
                if m:
                    op = float(m.group(3)) if PATH == GREENBOX_PATH else float(m.group(3)) / 1000
                    if op < min_op:
                        min_op = op
            ops_all.append(min_op*2.1/1000)
            total_all.append(min_op*2.1/1000 + embodied)
        ops_all = np.sum(np.array(ops_all), axis = 0)
        total_all = np.sum(np.array(total_all), axis = 0)
        util_all.append(ops_all)
    designs.append(util_all)

for i in range(len(designs)-1):
    baseline = sum(designs[i])
    greenbox = sum(designs[-1])
    print((baseline - greenbox) / max(baseline, greenbox))
line_styles = ['-', '--', '-.', ':', '-']
hatches = ["", "\\\\", "//", "--", "xx"]
# colors = ['#ff796c', 'plum', '#95d0fc', 'gray']
colors = ['#ff796c', 'plum', '#95d0fc', '#23a8eb', '#3d8c40', 'grey']
line_colors = ['#ff796c', 'plum', '#23a8eb', 'gray', 'black']
markers = ['o', '*', 'v', '^', 's']
plt.figure(figsize=(7,3.5), dpi=300)

policies = ["Centr-Global", "Centr-Graph", "Distr-Grid", "Distr-Battery", "GreenBox"]
l = ["40%", "50%", "60%", "70%", "80%", "90%", "100%"]
x = np.arange(0,len(l))
plt.xlabel('Util')
plt.ylabel('Carbon Footprint (tCO2e)')
plt.xticks(x, labels=l)
# plt.yticks([0,25,50,75,100])

width = 0.16
for i in range(len(policies)):
    plt.bar(x+(-2+i)*width, designs[i], width=width, color=colors[i], edgecolor="black", hatch=hatches[i], zorder=3)

plt.legend(policies, ncol=3, bbox_to_anchor=(0.5, 1.25), loc='upper center', frameon=False, fontsize=10)
plt.grid(which='major', axis='y', linestyle='--',linewidth=1)
max_value = max(max(r) for r in designs)
plt.ylim(top=max_value*1.2)
plt.tight_layout()
plt.savefig(f"figure/util_6_op.png")
plt.savefig(f"figure/util_6_op.pdf")
plt.clf()


# plt.xlabel('Low-Priority VM Percentage')
# plt.ylabel('Carbon Footprint (kgCO2eq)')
# plt.xticks(x, labels=l)
# for i in range(len(policies)):
#     plt.plot(x, results[i], line_styles[i], label=policies[i], color=colors[i], linewidth=2, marker=markers[i])
    
# # plt.legend(ncol=2, bbox_to_anchor=(0.5, 1.21), loc='upper center', frameon=False, fontsize=11)
# plt.legend(ncol=2, loc='upper right', frameon=False, fontsize=10)
# plt.grid(which='major', axis='y', linestyle='--',linewidth=1)
# plt.tight_layout()
# plt.savefig(f"figure/new_baseline_low_priority_ratio_6_{expt}.png")
# plt.savefig(f"figure/new_baseline_low_priority_ratio_6_{expt}.pdf")
# plt.clf()
