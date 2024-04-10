import os
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
BATTERY_PATH = f"{BASE_PATH}/battery"
CENTR_GLOBAL_PATH = f"{BASE_PATH}/centralized-global"
CENTR_SUB_PATH = f"{BASE_PATH}/centralized-sub"
DISTR_PATH = f"{BASE_PATH}/distr"
GREENBOX_PATH = f"{BASE_PATH}/greenbox"
ALL_PATH = [CENTR_GLOBAL_PATH, CENTR_SUB_PATH, DISTR_PATH, BATTERY_PATH, GREENBOX_PATH]
OUTPUT_DIR = f"tmp/output"
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
powermiss = np.arange(-20, 20+1, 2)
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

mispred_total = defaultdict(lambda : [0]*len(powermiss))
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
                mispred_total[date][(mispred_ratio+20)//2] = float(m.group(3))
mispred_avg = []
for start in starts:
    if start != 42 and start != 21:
        mispred_avg.append(mispred_total[start])
mispred_avg = np.array(mispred_avg)
mispred_avg = np.sum(mispred_avg, axis = 0)
plt.plot(mispred_avg)
plt.savefig("powermis_v2.png")
plt.clf()

for start in starts:
    plt.plot(mispred_total[start], label = start)
plt.legend()
plt.savefig("powermis_breakdown.png")

life_mispred_dirs = []
powermis = 0
lifemiss = np.arange(-20,20+1,2)
for lifemis in lifemiss:
    for start in starts:
        life_mispred = f"site{site}_slo{slo}_powermis{powermis}_lifemis{lifemis}_dist{dist}_start{start}_lookahead{lookahead}_UTIL{util}.txt"
        life_mispred_dirs.append(life_mispred)
lifemispred_total = defaultdict(lambda : [0]*len(lifemiss))
for life_mispred in life_mispred_dirs:
    args = [int(re.findall(r'-?\d+', arg)[0]) for arg in life_mispred.split("_")]
    mispred_ratio, date = args[3], args[5]
    filename = f"{dir}/{life_mispred}"
    f = open(filename, "r")
    count = 0
    for line in f.readlines():
        matcher = re.compile(pattern)
        m = matcher.search(line[:-1])
        if m:
            count += 1
            lifemispred_total[date][(mispred_ratio+20)//2] = float(m.group(3))
mispred_avg = []
for start in starts:
    mispred_avg.append(lifemispred_total[start])
mispred_avg = np.array(mispred_avg)
mispred_avg = np.sum(mispred_avg, axis = 0)
plt.plot(mispred_avg)
plt.savefig("lifemis_v2.png")
plt.clf()

for start in starts:
    plt.plot(lifemispred_total[start], label = start)
plt.legend()
plt.savefig("lifemis_breakdown.png")


#for powermis in powermiss:
#    for i in range(1, len(mispred_brown[powermis])):
#        mispred_brown[powermis][i] += mispred_brown[powermis][i-1]
#    plt.plot(mispred_brown[powermis], label = powermis)
#plt.legend()
#plt.savefig("powermis2.png")


#for path in ALL_PATH:
#    dir = f"{path}/{OUTPUT_DIR}"
#    for folder, subs, files in os.walk(dir):
#        for filename in files:
#            dir = os.path.abspath(os.path.join(folder, filename))
#            args = [int(re.findall(r'-?\d+', arg)[0]) for arg in filename.split("_")]
#            mispred_ratio, date = args[3], args[5]
#            f = open(dir, "r")
#            count = 0
#            for line in f.readlines():
#                matcher = re.compile(pattern)
#                m = matcher.search(line[:-1])
#                if m:
#                    browns[count].append(float(m.group(1)))
#                    greens[count].append(float(m.group(2)))
#                    totals[count].append(float(m.group(3)))
#                    count += 1
#                    mispred_brown[mispred_ratio][date//2] = float(m.group(1))
#print(mispred_brown)
#print({k:sum(v) for k,v in mispred_brown.items()})

