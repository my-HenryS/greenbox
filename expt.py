import subprocess
import itertools
import os
import time
import datetime as dt
import sys
import shutil
import copy
import psutil
from collections import defaultdict
import json
import numpy as np
import math

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
BATTERY_PATH = f"{BASE_PATH}/battery"
CENTR_GLOBAL_PATH = f"{BASE_PATH}/centralized-global"
CENTR_SUB_PATH = f"{BASE_PATH}/centralized-sub"
DISTR_PATH = f"{BASE_PATH}/distr"
GREENBOX_PATH = f"{BASE_PATH}/greenbox"
ALL_PATH = [CENTR_GLOBAL_PATH, CENTR_SUB_PATH, DISTR_PATH, BATTERY_PATH, GREENBOX_PATH]
ALL_PATH = [CENTR_GLOBAL_PATH, BATTERY_PATH, GREENBOX_PATH]
# ALL_PATH = [CENTR_GLOBAL_PATH, CENTR_SUB_PATH, DISTR_PATH, GREENBOX_PATH]
# ALL_PATH = [BATTERY_PATH]
# ALL_PATH = [GREENBOX_PATH]
RAW_DIR = f"tmp/raw"

#default setting
class default_config:
    def __init__(self):
        self.paths = ALL_PATH
        self.sites = [6]
        self.slos = [90]
        self.powermiss = [0]
        self.lifetimemiss = [0]
        self.distrs = [10]
        # self.starttimes = [42]
        self.starttimes = np.arange(0, 49+1, 7)
        self.lookaheads = [4]
        self.cloudutils = [90]

    def expand_path(self):
        return (self.paths, self.sites, self.slos, self.powermiss, self.lifetimemiss, self.distrs, self.starttimes, self.lookaheads, self.cloudutils)

    def expand(self):
        return (self.sites, self.slos, self.powermiss, self.lifetimemiss, self.distrs, self.starttimes, self.lookaheads, self.cloudutils)


def overall_path():
    setup = default_config()
    paths, sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils = setup.expand_path()
    parameters = [paths, sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils]
    args = list(itertools.product(*parameters))
    cmds = []
    for arg in args:
        path, site, slo, powermis, lifetimemis, distr, starttime, lookahead, cloudutil = arg
        RAW_PATH =  f"{path}/{RAW_DIR}/site{site}_slo{slo}_lifetimemis{lifetimemis}_powermis{powermis}_dist{distr}_starttime{starttime}_lookahead{lookahead}_util{cloudutil}.txt"
        cmd = f"python3 simulator.py {site} {slo} {powermis} {lifetimemis} {distr} {starttime} {lookahead} {cloudutil}"
        cmds.append((cmd, path, RAW_PATH))
    for site in sites:
        for starttime in starttimes:
            RAW_PATH =  f"{path}/{RAW_DIR}/site{site}_slo100_lifetimemis0_powermis0_dist0_starttime{starttime}_lookahead5_util90.txt"
            cmd = f"python3 simulator.py {site} 100 0 0 0 {starttime} 5 90"
            cmds.append((cmd, path, RAW_PATH))
    return cmds

def util_path():
    setup = default_config()
    paths, sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils = setup.expand_path()    
    cloudutils = [40,50, 60, 70, 80, 100]
    parameters = [paths, sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils]
    args = list(itertools.product(*parameters))
    cmds = []
    for arg in args:
        path, site, slo, powermis, lifetimemis, distr, starttime, lookahead, cloudutil = arg
        RAW_PATH =  f"{path}/{RAW_DIR}/site{site}_slo{slo}_lifetimemis{lifetimemis}_powermis{powermis}_dist{distr}_starttime{starttime}_lookahead{lookahead}_util{cloudutil}.txt"
        cmd = f"python3 simulator.py {site} {slo} {powermis} {lifetimemis} {distr} {starttime} {lookahead} {cloudutil}"
        cmds.append((cmd, path, RAW_PATH))
    return cmds

def performance_path():
    setup = default_config()
    paths, sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils = setup.expand_path()    
    distrs = [50]
    starttimes = [0,7,14,21,28,35,42,49]
    slos = np.append(np.arange(0, 50, 10), np.arange(50, 80, 5))
    slos = np.append(slos, np.arange(80, 100+1, 2))
    slos = np.append(slos, [99])
    parameters = [paths, sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils]
    args = list(itertools.product(*parameters))
    args.sort(key=lambda x: x[6])
    cmds = []
    for arg in args:
        path, site, slo, powermis, lifetimemis, distr, starttime, lookahead, cloudutil = arg
        RAW_PATH =  f"{path}/{RAW_DIR}/site{site}_slo{slo}_lifetimemis{lifetimemis}_powermis{powermis}_dist{distr}_starttime{starttime}_lookahead{lookahead}_util{cloudutil}.txt"
        cmd = f"python3 simulator.py {site} {slo} {powermis} {lifetimemis} {distr} {starttime} {lookahead} {cloudutil}"
        cmds.append((cmd, path, RAW_PATH))
    return cmds

def slo_path():
    setup = default_config()
    paths, sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils = setup.expand_path()    
    slos = [70, 80, 90, 95]
    parameters = [paths, sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils]
    args = list(itertools.product(*parameters))
    cmds = []
    for arg in args:
        path, site, slo, powermis, lifetimemis, distr, starttime, lookahead, cloudutil = arg
        RAW_PATH =  f"{path}/{RAW_DIR}/site{site}_slo{slo}_lifetimemis{lifetimemis}_powermis{powermis}_dist{distr}_starttime{starttime}_lookahead{lookahead}_util{cloudutil}.txt"
        cmd = f"python3 simulator.py {site} {slo} {powermis} {lifetimemis} {distr} {starttime} {lookahead} {cloudutil}"
        cmds.append((cmd, path, RAW_PATH))
    return cmds

def distr_path():
    setup = default_config()
    paths, sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils = setup.expand_path()
    distrs = [10, 20, 30, 40]
    parameters = [paths, sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils]
    args = list(itertools.product(*parameters))
    cmds = []
    for arg in args:
        path, site, slo, powermis, lifetimemis, distr, starttime, lookahead, cloudutil = arg
        RAW_PATH =  f"{path}/{RAW_DIR}/site{site}_slo{slo}_lifetimemis{lifetimemis}_powermis{powermis}_dist{distr}_starttime{starttime}_lookahead{lookahead}_util{cloudutil}.txt"
        cmd = f"python3 simulator.py {site} {slo} {powermis} {lifetimemis} {distr} {starttime} {lookahead} {cloudutil}"
        cmds.append((cmd, path, RAW_PATH))
    return cmds

def lahead_path():
    setup = default_config()
    paths, sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils = setup.expand_path()
    paths = [GREENBOX_PATH]
    distrs = [10, 20, 30, 40, 50]
    lookaheads = [2, 3, 4, 5, 6]
    parameters = [paths, sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils]
    args = list(itertools.product(*parameters))
    cmds = []
    for arg in args:
        path, site, slo, powermis, lifetimemis, distr, starttime, lookahead, cloudutil = arg
        RAW_PATH =  f"{path}/{RAW_DIR}/site{site}_slo{slo}_lifetimemis{lifetimemis}_powermis{powermis}_dist{distr}_starttime{starttime}_lookahead{lookahead}_util{cloudutil}.txt"
        cmd = f"python3 simulator.py {site} {slo} {powermis} {lifetimemis} {distr} {starttime} {lookahead} {cloudutil}"
        cmds.append((cmd, path, RAW_PATH))
    return cmds

def power_mispred_path():
    setup = default_config()
    paths, sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils = setup.expand_path()    
    paths = [GREENBOX_PATH]
    powermiss = np.concatenate((np.arange(-40, -20-1, 2), np.arange(20, 40+1, 2)))
    parameters = [paths, sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils]
    args = list(itertools.product(*parameters))
    cmds = []
    for arg in args:
        path, site, slo, powermis, lifetimemis, distr, starttime, lookahead, cloudutil = arg
        RAW_PATH =  f"{path}/{RAW_DIR}/site{site}_slo{slo}_lifetimemis{lifetimemis}_powermis{powermis}_dist{distr}_starttime{starttime}_lookahead{lookahead}_util{cloudutil}.txt"
        cmd = f"python3 simulator.py {site} {slo} {powermis} {lifetimemis} {distr} {starttime} {lookahead} {cloudutil}"
        cmds.append((cmd, path, RAW_PATH))
    return cmds

def vmlifetime_mispred_path():
    setup = default_config()
    paths, sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils = setup.expand_path()    
    paths = [GREENBOX_PATH]
    lifetimemiss = np.concatenate((np.arange(-40, -20-1, 2), np.arange(20, 40+1, 2))) #np.arange(-20, 20+1, 2)
    parameters = [paths, sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils]
    args = list(itertools.product(*parameters))
    cmds = []
    for arg in args:
        path, site, slo, powermis, lifetimemis, distr, starttime, lookahead, cloudutil = arg
        RAW_PATH =  f"{path}/{RAW_DIR}/site{site}_slo{slo}_lifetimemis{lifetimemis}_powermis{powermis}_dist{distr}_starttime{starttime}_lookahead{lookahead}_util{cloudutil}.txt"
        cmd = f"python3 simulator.py {site} {slo} {powermis} {lifetimemis} {distr} {starttime} {lookahead} {cloudutil}"
        cmds.append((cmd, path, RAW_PATH))
    return cmds

def scale_path():
    setup = default_config()
    paths, sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils = setup.expand_path()    
    # sites = [6, 12, 24, 36, 48]
    sites = [6, 18, 30, 42, 54]
    parameters = [paths, sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils]
    args = list(itertools.product(*parameters))
    cmds = []
    for arg in args:
        path, site, slo, powermis, lifetimemis, distr, starttime, lookahead, cloudutil = arg
        RAW_PATH =  f"{path}/{RAW_DIR}/site{site}_slo{slo}_lifetimemis{lifetimemis}_powermis{powermis}_dist{distr}_starttime{starttime}_lookahead{lookahead}_util{cloudutil}.txt"
        cmd = f"python3 simulator.py {site} {slo} {powermis} {lifetimemis} {distr} {starttime} {lookahead} {cloudutil}"
        cmds.append((cmd, path, RAW_PATH))
    return cmds

def dump_trace(PATH):
    setup = default_config()
    sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils = setup.expand()
    parameters = [sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils]
    args = list(itertools.product(*parameters))
    cmds = []
    for arg in args:
        site, slo, powermis, lifetimemis, distr, starttime, lookahead, cloudutil = arg
        cmd = f"python3 simulator.py {site} {slo} {powermis} {lifetimemis} {distr} {starttime} {lookahead} {cloudutil}"
        RAW_PATH =  f"{PATH}/{RAW_DIR}/site{site}_slo{slo}_lifetimemis{lifetimemis}_powermis{powermis}_dist{distr}_starttime{starttime}_lookahead{lookahead}_util{cloudutil}.txt"
        cmds.append((cmd, PATH, RAW_PATH))
    return cmds
        
STATIC = False
STEP_BY_STEP = False
def overall(PATH):
    setup = default_config()
    sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils = setup.expand()
    parameters = [sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils]
    args = list(itertools.product(*parameters))
    cmds = []
    for arg in args:
        site, slo, powermis, lifetimemis, distr, starttime, lookahead, cloudutil = arg
        if STATIC:
            RAW_PATH =  f"{PATH}/{RAW_DIR}/site{site}_slo{slo}_lifetimemis{lifetimemis}_powermis{powermis}_dist{distr}_starttime{starttime}_lookahead{lookahead}_util{cloudutil}_static.txt"
        elif STEP_BY_STEP:
            RAW_PATH =  f"{PATH}/{RAW_DIR}/site{site}_slo{slo}_lifetimemis{lifetimemis}_powermis{powermis}_dist{distr}_starttime{starttime}_lookahead{lookahead}_util{cloudutil}_step.txt"
        else:
            RAW_PATH =  f"{PATH}/{RAW_DIR}/site{site}_slo{slo}_lifetimemis{lifetimemis}_powermis{powermis}_dist{distr}_starttime{starttime}_lookahead{lookahead}_util{cloudutil}.txt"
        cmd = f"python3 simulator.py {site} {slo} {powermis} {lifetimemis} {distr} {starttime} {lookahead} {cloudutil}"
        cmds.append((cmd, PATH, RAW_PATH))
    for site in sites:
        for starttime in starttimes:
            if STATIC:
                RAW_PATH =  f"{PATH}/{RAW_DIR}/site{site}_slo100_lifetimemis0_powermis0_dist0_starttime{starttime}_lookahead5_util90_static.txt"
            elif STEP_BY_STEP:
                RAW_PATH =  f"{PATH}/{RAW_DIR}/site{site}_slo100_lifetimemis0_powermis0_dist0_starttime{starttime}_lookahead5_util90_step.txt"
            else:
                RAW_PATH =  f"{PATH}/{RAW_DIR}/site{site}_slo100_lifetimemis0_powermis0_dist0_starttime{starttime}_lookahead5_util90.txt"
            cmd = f"python3 simulator.py {site} 100 0 0 0 {starttime} 5 90"
            cmds.append((cmd, PATH, RAW_PATH))
    return cmds

def performance(PATH):
    setup = default_config()
    sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils = setup.expand()    
    distrs = [10, 30, 50, 90]
    # distrs = [30]
    starttimes = [0,7,14,21,28,35,42,49]
    # slos = np.arange(0, 50, 10) # only for test new objective of total carbon
    slos = np.append(np.arange(0, 50, 10), np.arange(50, 80, 5))
    # slos = np.arange(50, 80, 5)
    slos = np.append(slos, np.arange(80, 100+1, 2))
    # slos = np.append(slos, [99])
    # slos = [99]
    # slos = [90, 92, 94, 96, 98, 100]
    # slos = np.arange(80, 100, 2)
    parameters = [sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils]
    args = list(itertools.product(*parameters))
    cmds = []
    for arg in args:
        site, slo, powermis, lifetimemis, distr, starttime, lookahead, cloudutil = arg
        RAW_PATH =  f"{PATH}/{RAW_DIR}/site{site}_slo{slo}_lifetimemis{lifetimemis}_powermis{powermis}_dist{distr}_starttime{starttime}_lookahead{lookahead}_util{cloudutil}.txt"
        cmd = f"python3 simulator.py {site} {slo} {powermis} {lifetimemis} {distr} {starttime} {lookahead} {cloudutil}"
        cmds.append((cmd, PATH, RAW_PATH))
    return cmds

def util(PATH):
    setup = default_config()
    sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils = setup.expand()    
    cloudutils = [40, 50, 60, 70, 80, 100]
    parameters = [sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils]
    args = list(itertools.product(*parameters))
    cmds = []
    for arg in args:
        site, slo, powermis, lifetimemis, distr, starttime, lookahead, cloudutil = arg
        RAW_PATH =  f"{PATH}/{RAW_DIR}/site{site}_slo{slo}_lifetimemis{lifetimemis}_powermis{powermis}_dist{distr}_starttime{starttime}_lookahead{lookahead}_util{cloudutil}_test.txt"
        cmd = f"python3 simulator.py {site} {slo} {powermis} {lifetimemis} {distr} {starttime} {lookahead} {cloudutil}"
        cmds.append((cmd, PATH, RAW_PATH))
    return cmds

def slo(PATH):
    setup = default_config()
    sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils = setup.expand()    
    slos = [70, 80, 90, 95]
    parameters = [sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils]
    args = list(itertools.product(*parameters))
    cmds = []
    for arg in args:
        site, slo, powermis, lifetimemis, distr, starttime, lookahead, cloudutil = arg
        RAW_PATH =  f"{PATH}/{RAW_DIR}/site{site}_slo{slo}_lifetimemis{lifetimemis}_powermis{powermis}_dist{distr}_starttime{starttime}_lookahead{lookahead}_util{cloudutil}.txt"
        cmd = f"python3 simulator.py {site} {slo} {powermis} {lifetimemis} {distr} {starttime} {lookahead} {cloudutil}"
        cmds.append((cmd, PATH, RAW_PATH))
    return cmds

def distr (PATH):
    setup = default_config()
    sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils = setup.expand()
    distrs = [0, 20, 40]
    parameters = [sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils]
    args = list(itertools.product(*parameters))
    cmds = []
    for arg in args:
        site, slo, powermis, lifetimemis, distr, starttime, lookahead, cloudutil = arg
        RAW_PATH =  f"{PATH}/{RAW_DIR}/site{site}_slo{slo}_lifetimemis{lifetimemis}_powermis{powermis}_dist{distr}_starttime{starttime}_lookahead{lookahead}_util{cloudutil}.txt"
        cmd = f"python3 simulator.py {site} {slo} {powermis} {lifetimemis} {distr} {starttime} {lookahead} {cloudutil}"
        cmds.append((cmd, PATH, RAW_PATH))
    return cmds

def lahead(PATH):
    setup = default_config()
    sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils = setup.expand()
    # distrs = [10, 40, 50]
    distrs = [10, 30, 50]
    lookaheads = [2, 3, 4, 5, 6, 7]
    sites = [6]
    parameters = [sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils]
    args = list(itertools.product(*parameters))
    cmds = []
    for arg in args:
        site, slo, powermis, lifetimemis, distr, starttime, lookahead, cloudutil = arg
        RAW_PATH =  f"{PATH}/{RAW_DIR}/site{site}_slo{slo}_lifetimemis{lifetimemis}_powermis{powermis}_dist{distr}_starttime{starttime}_lookahead{lookahead}_util{cloudutil}.txt"
        cmd = f"python3 simulator.py {site} {slo} {powermis} {lifetimemis} {distr} {starttime} {lookahead} {cloudutil}"
        cmds.append((cmd, PATH, RAW_PATH))
    return cmds


def power_mispred(PATH):
    setup = default_config()
    sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils = setup.expand()    
    powermiss = np.concatenate((np.arange(-40, -20-1, 2), np.arange(20, 40+1, 2)))
    parameters = [sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils]
    args = list(itertools.product(*parameters))
    cmds = []
    for arg in args:
        site, slo, powermis, lifetimemis, distr, starttime, lookahead, cloudutil = arg
        RAW_PATH =  f"{PATH}/{RAW_DIR}/site{site}_slo{slo}_lifetimemis{lifetimemis}_powermis{powermis}_dist{distr}_starttime{starttime}_lookahead{lookahead}_util{cloudutil}.txt"
        cmd = f"python3 simulator.py {site} {slo} {powermis} {lifetimemis} {distr} {starttime} {lookahead} {cloudutil}"
        cmds.append((cmd, PATH, RAW_PATH))
    return cmds

def vmlifetime_mispred(PATH):
    setup = default_config()
    sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils = setup.expand()    
    lifetimemiss = np.arange(-40, 40+1, 2)
    # lifetimemiss = np.concatenate((np.arange(-40, -20-1, 2), np.arange(20, 40+1, 2))) #np.arange(-20, 20+1, 2)
    parameters = [sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils]
    args = list(itertools.product(*parameters))
    cmds = []
    for arg in args:
        site, slo, powermis, lifetimemis, distr, starttime, lookahead, cloudutil = arg
        RAW_PATH =  f"{PATH}/{RAW_DIR}/site{site}_slo{slo}_lifetimemis{lifetimemis}_powermis{powermis}_dist{distr}_starttime{starttime}_lookahead{lookahead}_util{cloudutil}.txt"
        cmd = f"python3 simulator.py {site} {slo} {powermis} {lifetimemis} {distr} {starttime} {lookahead} {cloudutil}"
        cmds.append((cmd, PATH, RAW_PATH))
    return cmds

def scale(PATH):
    setup = default_config()
    sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils = setup.expand()    
    # sites = [6, 12, 24, 36, 48]
    sites = [18, 30, 42, 54]
    parameters = [sites, slos, powermiss, lifetimemiss, distrs, starttimes, lookaheads, cloudutils]
    args = list(itertools.product(*parameters))
    cmds = []
    for arg in args:
        site, slo, powermis, lifetimemis, distr, starttime, lookahead, cloudutil = arg
        RAW_PATH =  f"{PATH}/{RAW_DIR}/site{site}_slo{slo}_lifetimemis{lifetimemis}_powermis{powermis}_dist{distr}_starttime{starttime}_lookahead{lookahead}_util{cloudutil}.txt"
        cmd = f"python3 simulator.py {site} {slo} {powermis} {lifetimemis} {distr} {starttime} {lookahead} {cloudutil}"
        cmds.append((cmd, PATH, RAW_PATH))
    return cmds

def check_all_expts(cmds, arr):
    cmd_arr = []
    for cmd in cmds:
        cmd_arr.append(cmd[2])
    cmd_arr.sort()
    arr.sort()
    if cmd_arr == arr:
        return True
    else:
        return False


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        mode = sys.argv[2]
    else:
        mode = "normal"
        
    node_no = int(sys.argv[1])
    cluster = int(sys.argv[3])
    
    if mode == "clean":
        #subprocess.run("pkill -f 'python3 simulator.py'", shell=True)
        subprocess.run("killall -f 'python3 simulator.py'", shell=True)
        subprocess.run(f"sudo rm -rf /tmp/ray/*", shell=True)
        # subprocess.run("pkill -f 'python3 expt.py'", shell=True)
        subprocess.run("killall python3", shell=True)
        exit()
    
    # num_cmds = 115
    cmds = set()
    for path in ALL_PATH:
        # cmd = overall(path)
        # cmds.update(cmd)
        # cmd = distr(path)
        # cmds.update(cmd)
        # cmd = slo(path)
        # cmds.update(cmd)
        cmd = scale(path)
        cmds.update(cmd)
        # cmd = performance(path)
        # cmds.update(cmd)
    # cmd = power_mispred(GREENBOX_PATH)
    # cmds.update(cmd)
    # cmd = vmlifetime_mispred(GREENBOX_PATH)
    # cmds.update(cmd)
    # cmd = dump_trace(GREENBOX_PATH)
    # cmds.update(cmd)
    # cmd = lahead(GREENBOX_PATH)
    # cmds.update(cmd)
    # cmd = performance(GREENBOX_PATH)
    # cmds.update(cmd)

    # cmd = performance_path()
    # cmds.update(cmd)
    # cmd = util_path()
    # cmds.update(cmd)
    # cmd = lahead(GREENBOX_PATH)
    # cmds.update(cmd)

    cmds = list(sorted(list(cmds), key=lambda x:x[2]))
    if cluster == 1:
        num_cmds = math.ceil(len(cmds) / 22)
    elif cluster == 2:
        num_cmds = math.ceil(len(cmds) / 5)
    elif cluster == 3:
        num_cmds = math.ceil(len(cmds) / 27)
        
    if node_no == 0:
        print(len(cmds))

    if mode == "check":
        arrs = []
        for path in ALL_PATH:
            dir = f"{path}/{RAW_DIR}"
            for folder, subs, files in os.walk(dir):
                for filename in files:
                    arrs.append(os.path.abspath(os.path.join(folder, filename)))
        if check_all_expts(cmds, arrs):
            print("all output files exist")
        else:
            print("files do not match")
        exit()
        
    cmds = cmds[num_cmds*node_no:num_cmds*(node_no+1)]
    # exit()



    procs = dict()
    while len(cmds) > 0 or len(procs) > 0:
        exit_codes = {proc:proc.poll() for proc in procs}
        finished_procs = [proc for proc, ec in exit_codes.items() if ec is not None ]
        if len(finished_procs) > 0:
            for finished in finished_procs:
                print(f"Finished job {procs[finished][0]} at {node_no}")
                finished.wait()
                del procs[finished]
        else:
            time.sleep(5)
        
        while len(procs) < 20 and len(cmds) > 0:
            next_batch = [cmds.pop()]

            for batchArg in next_batch:
                with open(batchArg[2],"wb") as out:
                    proc = subprocess.Popen(batchArg[0], cwd=batchArg[1], stdout=out, shell=True)
                    # proc = subprocess.Popen(batchArg[0], cwd=batchArg[1], stdout=out, stderr=out, shell=True)
                procs[proc] = batchArg
                print(f"Running job {proc.pid}: {batchArg} at {node_no}")
                print(f"{batchArg}")
        
        if len(cmds) > 0:
            time.sleep(5)
