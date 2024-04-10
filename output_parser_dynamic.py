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
util = 90
serverc = 2.55
batteryc = 0.25
embodied = serverc
dynamic = []
for i in range(2):
    ops_all = []
    total_all = []
    for start in starts:
        if i ==0:
            start_file = f"site{site}_slo{slo}_powermis{powermis}_lifemis{lifemis}_dist{dist}_start{start}_lookahead{lookahead}_UTIL{util}"
        else:
            start_file = f"site{site}_slo{slo}_powermis{powermis}_lifemis{lifemis}_dist{dist}_start{start}_lookahead{lookahead}_UTIL{util}_static"
        dir = f"{GREENBOX_PATH}/{OUTPUT_DIR}/{start_file}.txt"
        f = open(dir, "r")
        min_op = 10000000
        for line in f.readlines():
            matcher = re.compile(pattern)
            m = matcher.search(line[:-1])
            if m:
                op = float(m.group(3))
                if op < min_op:
                    min_op = op
        ops_all.append(min_op*2.1/1000)
        total_all.append(min_op*2.1/1000 + embodied)
    dynamic.append(ops_all)

print(dynamic)
line_styles = ['-', '--', '-.', ':', '-']
hatches = ["", "\\\\", "//", "--", "xx"]
# colors = ['#ff796c', 'plum', '#95d0fc', 'gray']
colors = ['#ff796c', 'plum', '#95d0fc', '#23a8eb', '#3d8c40', 'grey']
line_colors = ['#ff796c', 'plum', '#23a8eb', 'gray', 'black']
markers = ['o', '*', 'v', '^', 's']
plt.figure(figsize=(7,3.5), dpi=300)

policies = ["Dynamic", "Static"]
# l = ["week1", "week2", "week3", "week4", "week5", "week6", "week7", "week8"]
l = ["1", "2", "3", "4", "5", "6", "7", "8"]
x = np.arange(0,len(l))
plt.xlabel('Index of week')
plt.ylabel('Carbon Footprint (tCO2e)')
plt.xticks(x, labels=l)
for i in range(len(policies)):
    plt.plot(x, dynamic[i], line_styles[i], label=policies[i], color=colors[i], linewidth=2, marker=markers[i])

plt.legend(policies, ncol=3, bbox_to_anchor=(0.5, 1.25), loc='upper center', frameon=False, fontsize=10)
plt.grid(which='major', axis='y', linestyle='--',linewidth=1)
plt.tight_layout()
plt.savefig(f"figure/dynamic.png")
plt.savefig(f"figure/dynamic.pdf")
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
