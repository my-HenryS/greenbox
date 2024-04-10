import pandas as pd 
import glob
import datetime as dt
from dateutil import tz
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import graphviz
from nuts_finder import NutsFinder
import json
import h3
import random
import math
from matplotlib.collections import LineCollection
from renewable_simulation import App, GREEDY, BATCH_MIP, create_workloads

def plot_sites(coords, sites):
    x, y = zip(*[coords[site.split("-")[0]] for site in sites])
    lines = []
    plt.scatter(x,y)
    for site0 in sites:
        for site1 in sites:
            if site0 == site1:
                continue
            coords0 = coords[site0.split("-")[0]]
            coords1 = coords[site1.split("-")[0]]
            dist = h3.point_dist(coords0, coords1, unit='km')
            if 2 * dist / 200 <= 10:
                lines.append([(coords0[0], coords0[1]), (coords1[0], coords1[1])])
    lc = LineCollection(lines, color=["k","blue"], lw=0.05)
    plt.gca().add_collection(lc)
    plt.savefig(f"results/vb_sites.jpg")
    plt.clf()

max_cores = 320
max_power = 1 # (MV)
def run_sim(energy):
    num_sites = len(energy.columns)
    energy = energy.to_numpy().transpose()
    energy *= max_cores / max_power
    apps = create_workloads([20,20,20])
    policy = GREEDY("greedy", energy, apps, num_sites)
    t, avg_t = policy.migrate(step=24, init=False)
    return avg_t

random.seed(10)
coords = dict()
with open('datasets/nuts.json') as f:
    raw = f.readlines()[0]
coords = eval(raw)

all_types = ["solar", "wind"]

all_traces = []

for site in all_types:
    renewable_trace = pd.read_csv(f'emhires_{site}_2000.csv')
    renewable_trace.rename(columns = {"Time step":'Time'}, inplace = True)
    renewable_trace = renewable_trace.set_index('Time')
    for col in renewable_trace.columns[1:]:
        renewable_trace.rename(columns = {col:col+"-"+site}, inplace = True)

    all_traces.append(renewable_trace)

renewable_traces = pd.concat(all_traces, axis=1, join='inner')
renewable_traces.replace(np.NaN, 0, inplace=True)
renewable_traces *= 20
# log_msg(renewable_traces)
all_sites = renewable_traces.columns[1:].to_list()
# log_msg(len(all_sites))
all_sites = [site for site in all_sites if site.split('-')[0] in coords]
#plot_sites(coords, all_sites[:len(all_sites)//2])


# all_combinations = [v for max_sites in range(5) for v in itertools.combinations(all_sites, max_sites)]

def random_combination(iterable,r):
    i = 0
    pool = tuple(iterable)
    n = len(pool)
    rng = range(n)
    while True:
        yield [pool[j] for j in random.sample(rng, r)] 

all_combinations = []
# all_combinations += [[site] for site in all_sites]
# all_combinations += [list(v) for v in itertools.combinations(all_sites, 2) if v[0].split("-")[1] != v[1].split("-")[1]]
log_msg(f"Total {len(all_combinations)} combinations")
max_K = 3
max_subgraphs = 100000
for combination in random_combination(all_sites, max_K):
    dist_check = True
    for site0, site1 in itertools.combinations(combination, 2):
        coords0 = coords[site0.split("-")[0]]
        coords1 = coords[site1.split("-")[0]]
        dist = h3.point_dist(coords0, coords1, unit='km')
        if 2 * dist / 200 > 10:
            dist_check = False
            break

    all_types = set([site.split("-")[1] for site in combination])
    type_check = len(all_types) == 2

    if dist_check and type_check:
        all_combinations.append(combination)
    if len(all_combinations) >= max_subgraphs:
        break
log_msg(f"Total {len(all_combinations)} combinations")

interval = 24
based_time = 260972
start_date = dt.datetime(2015, 3, 1)
total_interval = 10*interval # len(renewable_trace)

previous_subgraphs = []
best_subgraph = None
for time in range(based_time, based_time+total_interval, interval):
    date = start_date+dt.timedelta(hours=time-based_time)
    current_renewable_traces = renewable_traces.loc[time:time+interval].reset_index()
    results = []

    for i, combination in enumerate(all_combinations):
        if i % 10000 == 0:
            log_msg(i)
        total_energy = current_renewable_traces[combination].values.sum(axis=1)
        avg = np.sum(total_energy) / len(total_energy)
        if avg == 0:
            continue
        cov = np.std(total_energy) / avg
        min_ = np.min(total_energy) / avg
        result = [combination, cov, avg, min_]
        if np.isnan(cov):
            log_msg(combination)
            for site in combination:
                log_msg(current_renewable_traces[site])
            log_msg(total_energy)
            exit(0)
        results += [result]
    
    sorted_results = list(sorted(results, key=lambda x: x[3], reverse=True))[:3000]
    for result in sorted_results:
        avg_t = run_sim(current_renewable_traces[result[0]])
        result.append(avg_t)

    sorted_results = list(sorted(sorted_results, key=lambda x: x[4]))

    sorted_results_match_last_day = list(sorted([result for result in results if result[0] in previous_subgraphs], key=lambda x: x[3], reverse=True))
    subgraphs = []
    nodes = set()
    total_stable_energy = 0
    migration_amount = 0
    for result in sorted_results_match_last_day:
        if result[3] == 0 or nodes.intersection(set(result[0])):
            continue
        subgraphs.append(result)
        nodes.update(result[0])
        total_stable_energy += result[3] * result[2]

    for result in sorted_results:
        if nodes.intersection(set(result[0])):
            continue
        subgraphs.append(result)
        nodes.update(result[0])
        if result[0] not in previous_subgraphs:
            migration_amount += len(result[0])
        total_stable_energy += result[3] * result[2]
        
    # log_msg(date, subgraphs)
    log_msg(f"Date {date}: Total stable energy of all VBs {total_stable_energy:.2f} MW; Total VB sites to migrate {migration_amount}")
    previous_subgraphs = [v[0] for v in subgraphs]

    if not best_subgraph:
        best_subgraph = subgraphs[0][0]
    for site in best_subgraph:
        plt.plot(current_renewable_traces[site], label=site)
    plt.plot(sum(current_renewable_traces[site] for site in best_subgraph), label="total", color="purple")
    plt.xlabel("Hours")
    plt.ylabel("Power (MW)")
    plt.legend()
    plt.savefig(f"results/{date}.jpg")
    plt.clf()

    dot = graphviz.Graph(comment='The Round Table')
    for i, site in enumerate(all_sites):
        dot.node(str(i), site)
    for i, subgraph in enumerate(subgraphs):
        with dot.subgraph(name=f'cluster_{i}') as c:
            for a,b in itertools.combinations(subgraph[0],2):
                c.edge(str(all_sites.index(a)), str(all_sites.index(b)))
            c.attr(label = f"VB_Cluster_{i}, \nAverage Power {subgraph[2]:.2f} MW, \nCoeff of Variation {subgraph[1]:.2f}, \nStable power {subgraph[3]*subgraph[2]:.2f} MW")
    dot.render(f'results/{date}.gv')  





