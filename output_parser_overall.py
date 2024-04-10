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

covs = [0.276, 0.508, 0.591, 0.775, 0.604, 0.589, 0.726, 0.286]

site = 6
slo = 90
powermis = 0
lifemis = 0
dist = 10
starts = np.arange(0, 49+1, 7)
lookahead = 4
util = 90
embodied_all = []
browns_all = []
greens_all = []
carbon_total = defaultdict(lambda: defaultdict(float))
cov_total = defaultdict(float)
for start in starts:
    start_file = f"site{site}_slo{slo}_powermis{powermis}_lifemis{lifemis}_dist{dist}_start{start}_lookahead{lookahead}_UTIL{util}"
    # serverc = 2927.34
    serverc = 2.55
    batteryc = 0.25
    embodied = np.array([serverc/1.2, serverc/1.1, serverc, serverc+batteryc, serverc])
    browns = []
    greens = []
    for i in range(len(ALL_PATH)):
        PATH = ALL_PATH[i]
        dir = f"{PATH}/{OUTPUT_DIR}/{start_file}{suffix[i]}.txt"
        f = open(dir, "r")
        # min_brown = 1000000
        # min_green = 1000000
        count = 0
        for line in f.readlines():
            matcher = re.compile(pattern)
            m = matcher.search(line[:-1])
            if m:
                count += 1
                if count == 5:
                    brown = float(m.group(1)) if PATH == GREENBOX_PATH else float(m.group(1)) / 1000
                    green = float(m.group(2)) if PATH == GREENBOX_PATH else float(m.group(2)) / 1000
                    break
                # if brown < min_brown:
                #     min_brown = brown
                # if green < min_green:
                #     min_green = green
        browns.append(brown*2.1 / 1000)
        greens.append(green*2.1/ 1000)
    embodied_all.append(embodied)
    browns_all.append(browns)
    greens_all.append(greens)
    carbon_total['embodied'][int(start)] = list(embodied)
    carbon_total['brown'][int(start)] = browns 
    carbon_total['green'][int(start)] = greens
    cov_total[int(start)] = covs[start // 7]
embodied_all = np.array(embodied_all)
browns_all = np.array(browns_all)
greens_all = np.array(greens_all)
print(carbon_total)
with open(f"carbon.json", "w") as out_f:
    json.dump(carbon_total, out_f)
with open(f"cov.json", "w") as out_f:
    json.dump(cov_total, out_f)

fig, ax = plt.subplots(figsize=(12,3.5), dpi=1200)
timelabels = ["Centr-Global", "Centr-Gragh", "Distr-Grid", "Distr-Battery", "GreenBox"]
x = np.arange(0,len(starts))
all_x = np.zeros(len(starts)*len(timelabels))
width = 0.14
gap = 0.4
for i in range(len(timelabels)):
    # normal one
    ax.bar(x+(i-2)*width, embodied_all[:,i], width=width, color="#767676", edgecolor="black", hatch="\\\\", zorder=3)
    ax.bar(x+(i-2)*width, browns_all[:,i], bottom=embodied_all[:,i], width=width, color="#846954", edgecolor="black", hatch="//", zorder=3)
    ax.bar(x+(i-2)*width, greens_all[:,i], bottom=embodied_all[:,i]+browns_all[:,i], width=width, color="#a5d74d", edgecolor="black", zorder=3)

    times = len(starts)
    if i == 0:
        all_x[i*times:(i+1)*times] = x+(i-3.4)*width
    elif i == 1:
        all_x[i*times:(i+1)*times] = x+(i-3.2)*width
    elif i == 2:
        all_x[i*times:(i+1)*times] = x+(i-2.7)*width
    elif i == 3:                                   
        all_x[i*times:(i+1)*times] = x+(i-2.9)*width
    elif i == 4:
        all_x[i*times:(i+1)*times] = x+(i-2.2)*width
ax2 = ax.twinx()
ax2.plot(x, covs, linewidth=2, color= '#3d8c40', marker= 'o', linestyle="--", markersize=7)
ax2.set_ylabel("Average CoV of Power Supply", fontsize=11)
ax2.set_ylim(-0.1, 0.95)
ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax2.legend(["CoV of Power Supply"], loc="upper right", frameon=False, fontsize=11)

labels = []
text_offset = -6.5
for l in timelabels:
    labels += [l]*times
ax.tick_params(axis=u'x', which=u'both',length=0)
ax.set_xticks(all_x, labels=labels, fontsize=8, rotation=30) 
ax.set_ylabel("Carbon Footprint (tCO2e)", fontsize=11)
for i in range(times):
    ax.text(x[i]-1.8*width, text_offset, f"Week{i+1}", weight='bold', fontsize=10.5)
ax.legend(["Amortized Embodied Carbon", "Brown Operational Carbon", "Green Operational Carbon"], ncol=1, loc='upper left', frameon=False, fontsize=11)
ax.set_ylim(top=20)
ax.grid(which='major', axis='y', linestyle='--',linewidth=1)
fig.tight_layout()
plt.savefig("./figure/separate_starttime_new_6.png")
plt.savefig("./figure/separate_starttime_new_6.pdf")
plt.clf()
plt.cla()

fig, ax = plt.subplots(figsize=(10,4), dpi=1200)
width = 0.5
x = np.arange(0, len(timelabels))
e = np.array([sum(embodied_all[:,i]) for i in range(len(timelabels))])
b = np.array([sum(browns_all[:,i]) for i in range(len(timelabels))])
g = np.array([sum(greens_all[:,i]) for i in range(len(timelabels))])
ax.set_ylabel("Carbon Footprint (tCO2e)", fontsize=14)
ax.set_xticks(x, labels=timelabels, fontsize=14)
ax.set_xlabel("Designs for rMDC placement", fontsize=14)
ax.bar(x, e, width=width, color="#767676", edgecolor="black", hatch="\\", zorder=3)
ax.bar(x, b, bottom=e, width=width, color="#846954", edgecolor="black", hatch="/", zorder=3)
ax.bar(x, g, bottom=e+b, width=width, color="#a5d74d", edgecolor="black", zorder=3)
ax.legend(["Embodied Carbon", "Brown Operational Carbon", "Green Operational Carbon"], ncol=3, bbox_to_anchor=(0.5, 1.2), loc='upper center', frameon=False, fontsize=13.5)
ax.grid(which='major', axis='y', linestyle='--',linewidth=1)
max_value = np.max(e+b+g)
ax.set_ylim(top=max_value*5)
fig.tight_layout()
plt.savefig("./figure/separate_starttime_total_6.png",bbox_inches='tight')
plt.savefig("./figure/separate_starttime_total_6.pdf",bbox_inches='tight')
plt.clf()
plt.cla()