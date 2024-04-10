import warnings 
warnings.filterwarnings(action='ignore',module='.*paramiko.*')

import pandas as pd 
import json
import glob
import datetime as dt
from dateutil import tz
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import graphviz
import json
import h3
import random
import math
from matplotlib.collections import LineCollection
from collections import defaultdict
import os
import sys
import copy
import pickle
import time
import csv
import utm

import gurobipy as gp
from gurobipy import GRB

from joblib import Parallel, delayed

import ray
GRB_ENV = gp.Env(empty=True)

# num_jobs = 24
# pool = Parallel(n_jobs=num_jobs)

DATA_DIR = "datasets"
PLOT_DIR = "plots"

random.seed(10)

MINS = 1
HOURS = 60*MINS
DAYS = 24*HOURS
WEEKS = 7*DAYS

TIME_OFFSET = 260972
DATE_OFFSET = dt.datetime(2015, 3, 1)

DEBUG = True
THR = 10
# BATCH_MIP_OPT_TIME = None
# BATCH_MIP_OPT_GOAL = None


np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

def log_msg(*msg, replace=False, emphasize=False, warning=False, header=False):
    '''
    Log a message with the current time stamp.
    '''
    if DEBUG:
        msg = [str(_) for _ in msg]
        out_str = "[%s] %s" % ((dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), " ".join(msg))
        if replace:
            ending="\r"#"\x1b[1K\r"
        else:
            ending="\n"
            
        if emphasize:
            out_str = bcolors.OKGREEN + out_str + bcolors.ENDC
        if warning:
            out_str = bcolors.WARNING + out_str + bcolors.ENDC
        if header:
            out_str = bcolors.HEADER + out_str + bcolors.ENDC
            
        print(out_str, flush=True, end=ending)

def format_numpy(array):
    return " ".join(map(str, array))

# obtain nuts region geolocation 
def read_vb_coords():
    vb_coords = dict()
    with open(f'datasets/nuts2.json') as f:
        vb_coords = json.load(f)
    return vb_coords

# obtain coeff 
def read_vb_capacity():
    capacity = dict()
    with open("datasets/coeff_vb.json", "r") as power_f:
        capacity.update(json.load(power_f))
    for k, v in capacity.items():
        capacity[k] = 500
    return capacity

# read renewable traces
def read_traces(vb_coords, capacity, interp_factor = 1):
    all_types = ["solar", "wind"]
    all_traces = []

    for type in all_types:
        renewable_trace = pd.read_csv(f'{DATA_DIR}/emhires_{type}_2000.csv')
        renewable_trace.rename(columns = {"Time step":'Time'}, inplace = True)
        # renewable_trace = renewable_trace.set_index('Time')
        for col in renewable_trace.columns:
            if sum(renewable_trace[col]) == 0.0:
                renewable_trace.drop(columns=[col], inplace = True)
                continue

            vb_name = col+"-"+type
            if col in vb_coords and vb_name in capacity:
                renewable_trace.rename(columns = {col:vb_name}, inplace = True)
                # TODO: it should be the np.max between the start time and end time
                renewable_trace[vb_name] *= capacity[vb_name] / np.max(renewable_trace[vb_name])
                # print(vb_name, np.average(renewable_trace[vb_name]), np.max(renewable_trace[vb_name]))

            else:
                renewable_trace.drop(columns=[col], inplace = True)

        all_traces.append(renewable_trace)

    renewable_traces = pd.concat(all_traces, axis=1, join='inner')
    renewable_traces.replace(np.NaN, 0, inplace=True)
    vb_sites = renewable_traces.columns[1:].to_list()
    
    n_rows = len(renewable_traces.iloc[:, 0])
    x = np.arange(n_rows)
    x_interp = np.arange(0, n_rows, 1/interp_factor)
    interp_results = dict()

    for site in vb_sites:
        y = renewable_traces[site].to_numpy()
        y_interp = np.interp(x_interp, x, y)
        interp_results[site] = y_interp
        
    interp_renewable_traces = pd.DataFrame(interp_results)
    
    return interp_renewable_traces, vb_sites

def cal_embodied_carbon(trace):
    avg_server_power = 500 # watts/server
    avg_server_carbon = 1220.87 # kgco2/server
    per_site_power = trace.max().sum()
    num_server = per_site_power  / avg_server_power
    embodied_carbon = num_server * avg_server_carbon
    return embodied_carbon

def cal_operational_carbon(nr_power, clean_power):
    brown_ci = 0.7 #kgco2/kwh
    clean_ci = 0.011 #kgco2/kwh
    return [nr_power * brown_ci, clean_power * clean_ci, nr_power * brown_ci + clean_power * clean_ci]

def cal_server_cost(trace):
    avg_server_power = 500 # watts/server
    avg_server_cost = 5000 # $/server
    per_site_power = trace.max().sum()
    num_server = per_site_power  / avg_server_power
    server_cost = num_server * avg_server_cost
    return server_cost

def cal_transmission_cost():
    avg_transmission_cost = 37.5 # k$ / km

def cal_network_cost():
    avg_network_cost = 300 # k$ . km
    

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# import matplotlib
# # line_styles = ['-', '--', '-.', ':']
# line_styles = ['-', '-', '-']
# markers = ['o', '*', 'v', '^']
# plt.rcParams.update({'figure.max_open_warning': 0})
# matplotlib.rcParams.update({'font.size': 12})
# matplotlib.rcParams.update({'font.family': 'serif'})
# matplotlib.rcParams['xtick.major.pad'] = '8'
# matplotlib.rcParams['ytick.major.pad'] = '8'
# matplotlib.rcParams['hatch.linewidth'] = 0.5
# matplotlib.rcParams['lines.linewidth'] = 2

# vb_coords = read_vb_coords()
# capacity = read_vb_capacity()
# trace, vb_sites = read_traces(vb_coords, capacity, 1)
# start_time = 24
# end_time = 96
# interval = end_time - start_time
# fig, axs = plt.subplots(1, 1, figsize=(5,3), dpi=1200)

# num = (end_time - start_time) * 4
# site_names = np.array(trace.columns)
# indexes = np.random.choice(site_names, 5)
# indexes = ['EL51-wind', 'PL92-wind', 'BE25-solar']
# xticks = np.arange(0, end_time - start_time, (end_time - start_time) / num)
# for i, index in enumerate(indexes):
#     certain_trace = trace[index][start_time:end_time]
#     certain_trace = np.interp(xticks, np.arange(end_time-start_time), certain_trace)
#     axs.plot(xticks, certain_trace, linestyle=line_styles[i], marker="o", markersize=2)
# axs.set_xlabel("Hours")
# axs.set_ylabel("Power Production (kW)")
# axs.legend(['EL51-wind', 'PL92-wind', 'BE25-solar'], fontsize=11)
# plt.grid(which='major', axis='y', linestyle='--',linewidth=1)
# fig.tight_layout()
# fig.savefig("test.png")
# selected_sites = ['EL51-wind', 'PL92-wind', 'PT18-wind', 'UKK3-wind', 'FI1C-wind', 'UKK4-wind']
# for site in selected_sites:
#     power_total[site] = list(trace.loc[start_time:end_time][site])
# print(power_total)
# with open(f"prune.json", "w") as out_f:
#     json.dump(power_total, out_f)

# selected_sites = ['EL51-wind', 'PL92-wind', 'PT18-wind']
# x = np.arange(0, interval + 1,1)
# for i in range(len(selected_sites)):
#     site = selected_sites[i]
#     axs[0].plot(x, trace.loc[start_time:end_time][site], label=site, marker=markers[i], linestyle=linestyles[i]) 
# axs[0].set_xlabel("Hours\n\n(a) Farms pruned by Greenbox")
# axs[0].set_ylabel("Power (MW)")
# axs[0].set_ylim(top=125)
# axs[0].set_xticks([0, 10, 20])
# axs[0].legend(["EL51", "PL92", "PT18"],ncol=3, bbox_to_anchor=(0.5, 1.2), loc='upper center',frameon=False, fontsize=12)

# selected_sites = ['UKK3-wind', 'FI1C-wind', 'UKK4-wind']
# for i in range(len(selected_sites)):
#     site = selected_sites[i]
#     axs[1].plot(x, trace.loc[start_time:end_time][site], label=site, marker=markers[i], linestyle=linestyles[i]) 
# axs[1].set_xlabel("Hours\n\n(a) Farms selected by Greenbox")
# # axs[1].set_ylabel("Power (MW)")
# axs[1].set_ylim(top=125)
# axs[1].set_xticks([0, 10, 20])
# axs[1].legend(["UKK3", "FI1C", "UKK4"],ncol=3, bbox_to_anchor=(0.5, 1.2), loc='upper center',frameon=False, fontsize=12)
# plt.savefig(f"selected_sites.jpg", bbox_inches='tight')
# plt.savefig(f"selected_sites.pdf", bbox_inches='tight')
# plt.clf()



# fig, axs = plt.subplots(1, 2, figsize=(10,3), dpi=1200)
# selected_sites = ['EL51-wind', 'PL92-wind', 'PT18-wind']
# x = np.arange(0, len(selected_sites),1)
# extras = []
# width = 0.6
# color = "#767676"
# for i in range(len(selected_sites)):
#     site = selected_sites[i]
#     extra = max(trace.loc[start_time:end_time][site]) / sum(trace.loc[start_time:end_time][site]) * interval - 1
#     extras.append(extra)
# axs[0].bar(x, extras, width=width, color=color, label=site) 
# axs[0].set_xlabel("Farms pruned by Greenbox")
# axs[0].set_ylabel("Extra Embodied Carbon (%)")
# axs[0].set_xticks(x, selected_sites)
# # axs[0].legend(["EL51", "PL92", "PT18"],ncol=3, bbox_to_anchor=(0.5, 1.2), loc='upper center',frameon=False, fontsize=12)

# extras = []
# selected_sites = ['UKK3-wind', 'FI1C-wind', 'UKK4-wind']
# for i in range(len(selected_sites)):
#     site = selected_sites[i]
#     extra = max(trace.loc[start_time:end_time][site]) / sum(trace.loc[start_time:end_time][site]) * interval - 1
#     extras.append(extra)
# axs[1].bar(x, extras, width=width, color=color,label=site) 
# axs[1].set_xlabel("Farms Selected by Greenbox")
# # axs[1].set_ylabel("Extra Embodied Carbon (%)")
# axs[1].set_xticks(x , selected_sites)
# plt.savefig(f"embodied.jpg", bbox_inches='tight')
# plt.savefig(f"embodied.pdf", bbox_inches='tight')
# plt.clf()