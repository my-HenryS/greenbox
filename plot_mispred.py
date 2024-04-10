import os
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import json

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
BATTERY_PATH = f"{BASE_PATH}/battery"
CENTR_GLOBAL_PATH = f"{BASE_PATH}/centralized-global"
CENTR_SUB_PATH = f"{BASE_PATH}/centralized-sub"
DISTR_PATH = f"{BASE_PATH}/distr"
GREENBOX_PATH = f"{BASE_PATH}/greenbox"
ALL_PATH = [CENTR_GLOBAL_PATH, CENTR_SUB_PATH, DISTR_PATH, BATTERY_PATH, GREENBOX_PATH]
OUTPUT_DIR = f"tmp/output"
BACKUP_OUTPUT_DIR = f"tmp-backup/output"
RAW_DIR = f"tmp/raw"

greedy_brown = []
greedy_green = []
greedy_total = []
mip_app_brown = []
mip_app_green = []
mip_app_total = []
mip_brown = []
mip_green = []
mip_total = []
browns = [greedy_brown, mip_app_brown, mip_brown]
greens = [greedy_green, mip_app_green, mip_green]
totals = [greedy_total, mip_app_total, mip_total]
pattern = 'Total Total Carbon \[nr carbon, r carbon, total carbon\]: \[(\d+\.\d+[e\-\d+]*), (\d+\.\d+[e\-\d+]*), (\d+\.\d+[e\-\d+]*)\]'

site = 6
slo = 90
powermiss = np.arange(-30, 40+1, 2)
lifetimemiss = np.arange(-30, 40+1, 2)
lifemis = 0
dist = 10
starts = np.arange(0, 49+1, 7)
lookahead = 4
util = 90
power_mispred_dirs = []
for powermis in powermiss:
    for start in starts:
        power_mispred = f"site{site}_slo{slo}_powermis{powermis}_lifemis{lifemis}_dist{dist}_start{start}_lookahead{lookahead}_UTIL{util}.txt"
        power_mispred_dirs.append(power_mispred)

power_mispred_total = defaultdict(lambda : [0]*len(starts))
dir = f"{GREENBOX_PATH}/{OUTPUT_DIR}"
for power_mispred in power_mispred_dirs:
    args = [int(re.findall(r'-?\d+', arg)[0]) for arg in power_mispred.split("_")]
    mispred_ratio, date = args[2], args[5]
    filename = f"{dir}/{power_mispred}"
    f = open(filename, "r")
    count = 0
    for line in f.readlines():
        matcher = re.compile(pattern)
        m = matcher.search(line[:-1])
        if m:
            browns[count].append(float(m.group(1)))
            greens[count].append(float(m.group(2)))
            totals[count].append(float(m.group(3)))
            count += 1
            if count == 3:
                #mispred_brown[mispred_ratio][date] = float(m.group(1))
                power_mispred_total[mispred_ratio][date//7] = float(m.group(3))
                
lifetime_mispred_dirs = []
powermis = 0
for lifetimemis in lifetimemiss:
    for start in starts:
        lifetime_dir = f"site{site}_slo{slo}_powermis{powermis}_lifemis{lifetimemis}_dist{dist}_start{start}_lookahead{lookahead}_UTIL{util}.txt"
        lifetime_mispred_dirs.append(lifetime_dir)

lifetime_mispred_total = defaultdict(lambda : [0]*len(starts))
dir = f"{GREENBOX_PATH}/{OUTPUT_DIR}"
for lifetime_mispred in lifetime_mispred_dirs:
    args = [int(re.findall(r'-?\d+', arg)[0]) for arg in lifetime_mispred.split("_")]
    mispred_ratio, date = args[3], args[5]
    filename = f"{dir}/{lifetime_mispred}"
    f = open(filename, "r")
    count = 0
    for line in f.readlines():
        matcher = re.compile(pattern)
        m = matcher.search(line[:-1])
        if m:
            browns[count].append(float(m.group(1)))
            greens[count].append(float(m.group(2)))
            totals[count].append(float(m.group(3)))
            count += 1
            if count == 2:
                #mispred_brown[mispred_ratio][date] = float(m.group(1))
                lifetime_mispred_total[mispred_ratio][date//7] = float(m.group(3))
                
                       
print(power_mispred_total)
print(lifetime_mispred_total)

baselines = defaultdict(lambda : [0]*len(starts))
baseline_dirs = []
site, slo, powermis, lifetimemis, dist, start, lookahead, util = 6, 90, 0, 0, 10, 0, 4, 90
for start in starts:
    battery_dir = f"{BATTERY_PATH}/{OUTPUT_DIR}/site{site}_slo{slo}_powermis{powermis}_lifemis{lifetimemis}_dist{dist}_start{start}_lookahead{lookahead}_UTIL{util}_battery.txt"
    dist_dir = f"{DISTR_PATH}/{OUTPUT_DIR}/site{site}_slo{slo}_powermis{powermis}_lifemis{lifetimemis}_dist{dist}_start{start}_lookahead{lookahead}_UTIL{util}_distr.txt"
    baseline_dirs.extend([battery_dir, dist_dir])
    
for baseline in baseline_dirs:
    args = [int(re.findall(r'-?\d+', arg)[0]) for arg in baseline.split("_") if len(re.findall(r'-?\d+', arg)) > 0]
    date = args[5]
    if "battery" in baseline:
        mode = "battery"
    if "distr" in baseline:
        mode = "distr"
    filename = baseline
    f = open(filename, "r")
    count = 0
    for line in f.readlines():
        matcher = re.compile(pattern)
        m = matcher.search(line[:-1])
        if m:
            browns[count].append(float(m.group(1)))
            greens[count].append(float(m.group(2)))
            totals[count].append(float(m.group(3)))
            count += 1
            if count == 3:
                #mispred_brown[mispred_ratio][date] = float(m.group(1))
                baselines[mode][date//7] = float(m.group(3)) / 1000

print(baselines)

# power_mispred_total = {k:v[:4]+v[5:] for k,v in power_mispred_total.items()}
# lifetime_mispred_total = {k:v[:4]+v[5:] for k,v in lifetime_mispred_total.items()}

aggr_power_mispred = {k:sum(v) for k,v in power_mispred_total.items()}
aggr_lifetimemispred = {k:sum(v) for k,v in lifetime_mispred_total.items()}

total_results = dict()
total_results["power mispred"] = aggr_power_mispred
total_results["lifetime mispred"] = aggr_lifetimemispred
total_results["battery"] = baselines["battery"]
total_results["distr"] = baselines["distr"]
with open("misprediction.json", "w") as out_f:
    json.dump(total_results, out_f)

x,power_y = zip(*sorted(aggr_power_mispred.items()))
plt.plot(x,np.array(power_y), marker="*", label="power")

x,lifetime_y = zip(*sorted(aggr_lifetimemispred.items()))
plt.plot(x, np.array(lifetime_y), marker='o', label="lifetime")

print(np.array(lifetime_y) / np.array(aggr_lifetimemispred[0]))

linestyles = ['--', '-.']
colors = ['black', 'orange']
for i, (method, data) in enumerate(baselines.items()):
    plt.axhline(sum(data), label=method, color = colors[i], linestyle = linestyles[i])
plt.ylim([0, None])
plt.legend()
plt.savefig("misprediction.png")
plt.clf()

# aggr_power_mispred = {k:np.average(np.array(v)/np.array(power_mispred_total[0])) for k,v in power_mispred_total.items()}
# aggr_lifetimemispred = {k:np.average(np.array(v)/np.array(lifetime_mispred_total[0])) for k,v in lifetime_mispred_total.items()}

# # print(aggr_power_mispred)
# # print({k:np.array(v)/np.array(lifetime_mispred_total[0]) for k,v in lifetime_mispred_total.items()})

# x,power_y = zip(*sorted(aggr_power_mispred.items()))
# plt.plot(x,(np.array(power_y)-1)*100, marker="*", label="power")

# x,lifetime_y = zip(*sorted(aggr_lifetimemispred.items()))
# # print(lifetime_y)
# plt.plot(x, (np.array(lifetime_y)-1)*100, marker='o', label="lifetime")
# plt.legend()
# plt.savefig("misprediction_avg.png")
# plt.clf()