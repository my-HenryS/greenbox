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
GREENBOX_PATH = f"{BASE_PATH}/greenbox"
OUTPUT_DIR = f"tmp/output"
pattern = 'Total Total Carbon \[nr carbon, r carbon, total carbon\]: \[(\d+\.\d+[e\-\d+]*), (\d+\.\d+[e\-\d+]*), (\d+\.\d+[e\-\d+]*)\]'

site = 6
slo = 90
powermis = 0
lifemis = 0
# dists = [10, 20, 30, 40, 50]
dists = [10, 30, 50]
# starts = np.arange(0, 49+1, 7)
starts = [35]
# lookaheads = [1, 2, 3, 4, 5, 6]
lookaheads = [1, 2, 3, 4, 5, 6, 7]
util = 90
serverc = 2.55
batteryc = 0.25
PATH = GREENBOX_PATH
embodied = serverc

dist_all = []
for dist in dists:
    lahead_all = []
    for lookahead in lookaheads:
        total_all = []
        ops_all = []
        if lookahead == 1:
            lookahead = 2
            for start in starts:
                start_file = f"site{site}_slo{slo}_powermis{powermis}_lifemis{lifemis}_dist{dist}_start{start}_lookahead{lookahead}_UTIL{util}"
                dir = f"{PATH}/{OUTPUT_DIR}/{start_file}.txt"
                f = open(dir, "r")
                count = 0
                for line in f.readlines():
                    matcher = re.compile(pattern)
                    m = matcher.search(line[:-1])
                    if m:
                        count += 1
                        if count == 2:
                            op = float(m.group(1)) if PATH == GREENBOX_PATH else float(m.group(1)) / 1000
                            break
                ops_all.append(op*2.1/1000)
                total_all.append(op*2.1/1000 + embodied)
        else:
            for start in starts:
                start_file = f"site{site}_slo{slo}_powermis{powermis}_lifemis{lifemis}_dist{dist}_start{start}_lookahead{lookahead}_UTIL{util}"
                dir = f"{PATH}/{OUTPUT_DIR}/{start_file}.txt"
                f = open(dir, "r")
                count = 0
                for line in f.readlines():
                    matcher = re.compile(pattern)
                    m = matcher.search(line[:-1])
                    if m:
                        count += 1
                        if count == 6:
                            op = float(m.group(1))
                            break
                ops_all.append(op*2.1/1000)
                total_all.append(op*2.1/1000 + embodied)
        ops_all = np.sum(np.array(ops_all), axis = 0)
        total_all = np.sum(np.array(total_all), axis = 0)
        lahead_all.append(ops_all)
    dist_all.append(lahead_all)

line_styles = ['-', '--', '-.', ':', '-']
hatches = ["", "\\\\", "//", "--", "xx"]
# colors = ['#ff796c', 'plum', '#95d0fc', 'gray']
colors = ['#ff796c', 'plum', '#95d0fc', '#23a8eb', '#3d8c40', 'grey']
line_colors = ['#ff796c', 'plum', '#23a8eb', 'gray', 'black']
markers = ['o', '*', 'v', '^', 's']
plt.figure(figsize=(7,3.5), dpi=300)

labels = ["10%", "50%", "90%"]
l = ["greedy", "1", "2", "3", "4", "5", "6"]
# l = ["1", "2", "3", "4", "5", "6"]
x = np.arange(0,len(l))
plt.xlabel('Power Supply Prediction Time (hours)')
plt.ylabel('Power Grid Usage (Watts)')
plt.xticks(x, labels=l)
# plt.ylim(top = 1)

for i in range(len(dist_all)):
    plt.plot(x, dist_all[i], line_styles[i], color=colors[i], linewidth=2, marker=markers[i])

# plt.legend(labels, ncol=3, bbox_to_anchor=(0.5, 1.05), loc='upper center', frameon=False, fontsize=10)
plt.legend(labels)
plt.grid(which='major', axis='y', linestyle='--',linewidth=1)
plt.tight_layout()
plt.savefig(f"figure/lahead_6.png", bbox_inches='tight')
plt.savefig(f"figure/lahead_6.pdf", bbox_inches='tight')
plt.clf()

