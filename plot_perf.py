import os
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import matplotlib

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
BATTERY_PATH = f"{BASE_PATH}/battery"
CENTR_GLOBAL_PATH = f"{BASE_PATH}/centralized-global"
CENTR_SUB_PATH = f"{BASE_PATH}/centralized-sub"
DISTR_PATH = f"{BASE_PATH}/distr"
GREENBOX_PATH = f"{BASE_PATH}/greenbox"
ALL_PATH = [CENTR_GLOBAL_PATH, CENTR_SUB_PATH, DISTR_PATH, BATTERY_PATH, GREENBOX_PATH]
# ALL_PATH = [CENTR_GLOBAL_PATH, CENTR_SUB_PATH, DISTR_PATH, GREENBOX_PATH]
OUTPUT_DIR = f"tmp/output"
BACKUP_OUTPUT_DIR = f"tmp-backup/output"
RAW_DIR = f"tmp/raw"

pattern = 'Total Total Carbon \[nr carbon, r carbon, total carbon\]: \[(\d+\.\d+[e\-\d+]*), (\d+\.\d+[e\-\d+]*), (\d+\.\d+[e\-\d+]*)\]'

site = 6
# slos = np.append(np.arange(0, 50, 10), np.arange(50, 80, 5))
# slos = np.append(slos, np.arange(80, 100+1, 2))
# slos = np.arange(0, 50, 10)
# slos = np.append(slos, [99])
# slos = np.sort(slos)
powermis = 0
lifemis = 0
dists = [10, 30, 50, 90]
dist = dists[2]
starts = np.arange(0, 50, 7)
slos = np.append(np.arange(0, 50, 10), np.arange(50, 80, 5))
slos = np.append(slos, np.arange(80, 100+1, 2))
lookahead = 4
util = 90
dirs = []
for slo in slos:
    for start in starts:
        _dir = f"site{site}_slo{slo}_powermis{powermis}_lifemis{lifemis}_dist{dist}_start{start}_lookahead{lookahead}_UTIL{util}_SUFFIX.txt"
        dirs.append(_dir)

total_results = defaultdict(lambda : defaultdict(int))
all_suffix = ["_centralized_global", "_centralizedsub", "_distr", "_battery", ""]
methods = ["Centr-Global", "Centr-Graph", "Distr-Grid", "Distr-Battery", "GreenBox"]

finished, running = 0, 0
for i, path in enumerate(ALL_PATH):
    parent_dir = f"{path}/{OUTPUT_DIR}"
    suffix = all_suffix[i]
    for _dir in dirs:
        args = [int(re.findall(r'-?\d+', arg)[0]) for arg in _dir.split("_") if len(re.findall(r'-?\d+', arg)) > 0]
        slo, date = args[1], args[5]
        # print(slo)
        filename = f"{parent_dir}/{_dir}".replace("_SUFFIX", suffix)
        print(filename)
        if not os.path.exists(filename):
            running += 1
            continue
        else:
            finished += 1
        f = open(filename, "r")
        count = 0
        for line in f.readlines():
            matcher = re.compile(pattern)
            m = matcher.search(line[:-1])
            if m:
                # print(m.group())
                count += 1
                if count == 6:
                    total_carbon = float(m.group(3))
                    if path != GREENBOX_PATH:
                        total_carbon = float(m.group(3)) / 1000 

                    method = path.split("/")[-1]
                    total_results[methods[i]][slo] += total_carbon * 2.1
                    
print(finished, running) 
def lookup(key, carbon):
    sorted_items = sorted(total_results[key].items())
    for i in range(len(sorted_items)):
        if sorted_items[i][1] < carbon:
            continue
        if i == 0:
            return 0
        x0, y0 = sorted_items[i-1]
        x1, y1 = sorted_items[i]
        uptime = (x1 - x0) / (y1 - y0) * (carbon - y0) + x0
        # print(x0, y0, x1, y1, carbon, uptime)
        return uptime
    return 100
    
print(total_results)
with open(f"performance_lookup_table_low_priority_{dist}.json", "w") as out_f:
    json.dump(total_results, out_f)
    
proj_results = defaultdict(list)
if dist == 10:
    thresholds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1300, 1500] #%10
elif dist == 50:
    thresholds = [1000, 1100, 1500, 1700, 1900, 2100, 2300, 2500, 2700, 2900, 3100, 3200] #%50
elif dist == 90:
    thresholds = [2500, 2700, 2900, 3100, 3200, 3400, 3600, 3800, 4000, 4200] #%90
else:
    thresholds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1300, 1500] #%10
# thresholds = [2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 13000, 15000]
# thresholds = [3000, 3500, 5000, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 10000, 16000, 17000, 18000, 21000, 22000, 23000, 23500, 24000, 24500, 26000, 30000]
methods = total_results.keys()
for method in methods:
    for thr in thresholds:
        proj_results[method] += [lookup(method, thr)]
        # exit()

with open(f"vm_performance_low_priority_test_{dist}.json", "w") as out_f:
    json.dump(proj_results, out_f)

print(proj_results)
plt.rcParams.update({'figure.max_open_warning': 0})
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'font.family': 'serif'})
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'
matplotlib.rcParams['hatch.linewidth'] = 0.5
plt.figure(figsize=(7,5), dpi=300)

markers = ['o', '*', 'v', '^', '+']
line_styles = ['-', '-.']
colors = ['#ff796c', 'plum', '#95d0fc', '#23a8eb', '#3d8c40', 'grey']
line_colors = ['#ff796c', 'plum', '#23a8eb', 'gray']

for i, method in enumerate(methods):
    plt.plot(thresholds, proj_results[method], marker=markers[i], markersize=8, label=method, linewidth=2, linestyle=line_styles[i % len(line_styles)], color=colors[i%len(colors)])
plt.ylim([0, None])
plt.xlabel("Carbon Footprint")
plt.ylabel("VM Uptime Percentage")
plt.grid(which='major', axis='y', linestyle='--',linewidth=1)
plt.legend(ncol=3, bbox_to_anchor=(0.5, 1.2), loc='upper center', frameon=False, fontsize=10)
plt.tight_layout()
plt.savefig(f"vm_performance_low_priority_test_{dist}.png")