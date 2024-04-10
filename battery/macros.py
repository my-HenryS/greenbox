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
    brown_ci = 700 #gco2/kwh
    clean_ci = 11 #gco2/kwh
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