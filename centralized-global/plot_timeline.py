from email.policy import default
import enum
from sqlite3 import Timestamp
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict
import pickle
import sys
from glob import glob
import networkx as nx
from matplotlib.animation import FuncAnimation
import matplotlib.backends.backend_pdf
import matplotlib
from networkx.drawing.nx_agraph import to_agraph 
import img2pdf
import os
import pandas as pd
from PyPDF2 import PdfMerger
import glob

TEXT_ONLY = False

plt.rcParams.update({'figure.max_open_warning': 0})
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'font.family': 'serif'})
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'
matplotlib.rcParams['hatch.linewidth'] = 0.5


line_styles = ['-', '--', '-.', ':']
hatches = ["", "\\", "//", "||"]
colors = ['#ff796c', 'plum', '#95d0fc', 'gray', 'black']
markers = ['.', 'o', '*', 'v', '^']
    
def plot_cdf(results, filename, xlabel="CDF (%)", ylabel="CDF of Overhead (hrs)"):
    if TEXT_ONLY:
        return
    
    plt.figure(figsize=(8, 4), dpi=300)

    max_y_values = []
    for i, (policy, cdf_t) in enumerate(results.items()):
        cdf_t = sorted(cdf_t, reverse=False)
        cdf_t = np.array(cdf_t) 
        max_y_values.append(np.max(cdf_t))
        x_values = np.arange(1,len(cdf_t)+1)/len(cdf_t) * 100
        plt.plot(x_values, cdf_t, line_styles[i % len(line_styles)], label=policy, color=colors[i % len(colors)], linewidth=2)

    plt.legend(ncol=1, bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(max_y_values) > 0 and max(max_y_values) != 0:
        plt.ylim([0, max(max_y_values)*1.2])
    plt.grid(visible=True, which='major', axis='y', color='#000000', linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{filename}")
    plt.clf()


def plot_dist(results, filename, xlabel="Distribution", ylabel="Execution Time (hrs)"):
    if TEXT_ONLY:
        return
    
    plt.figure(figsize=(8, 4), dpi=300)

    max_y_values = []
    for i, (policy, cdf_t) in enumerate(results.items()):
        cdf_t = np.array(cdf_t) 
        max_y_values.append(np.max(cdf_t))
        x_values = np.arange(1,len(cdf_t)+1)
        plt.plot(x_values, cdf_t, line_styles[i % len(line_styles)], label=policy, color=colors[i % len(colors)+1], linewidth=2)

    # plt.legend(ncol=1, bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=8)
    plt.legend()
    plt.xticks(np.arange(5) * len(cdf_t) // 5, labels=["coffi", "db", "dnn", "db-large", "dnn-large"])
    plt.ylabel(ylabel)
    if len(max_y_values) > 0 and max(max_y_values) != 0:
        plt.ylim([0, max(max_y_values)*1.2])
    plt.grid(visible=True, which='major', axis='y', color='#000000', linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{filename}")
    plt.clf()
    
def plot_grouped_bar(results, filename, ylabel="Avg Overhead (hrs)", ylim=None, transpose=False):
    if TEXT_ONLY:
        return
    
    plt.figure(figsize=(8, 4), dpi=300)
    results = pd.DataFrame(results)
    row_labels = results.index.values
    column_labels = results.columns
    for i, column_label in enumerate(column_labels):
        column = results[column_label]
        plt.bar(np.arange(len(row_labels))*(len(column_labels)+1)+i, column, label=column_label, edgecolor='black', color=colors[i % len(colors)], zorder=3)

    plt.xticks(np.arange(len(row_labels))*(len(column_labels)+1)+(len(column_labels)-1)/2, labels=row_labels)
    plt.ylabel(ylabel)
    if ylim:
        plt.ylim(ylim)
    plt.legend(ncol=4, fontsize=12, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.2))
    plt.tight_layout()
    plt.grid(axis='y', zorder=0, linestyle='--')
    plt.savefig(f"{filename}")
    plt.clf()
    
def plot_stacked_bar(results, filename, ylabel="Avg Overhead (hrs)", overlay=None, overlay_ylabel=None):
    if TEXT_ONLY:
        return
    
    plt.figure(figsize=(8, 4), dpi=300)
    fig, ax = plt.subplots()
    if overlay:
        ax2 = ax.twinx()
    results = pd.DataFrame(results)
    row_labels = results.index.values
    column_labels = results.columns
    max_y_value = 0
    for i, row_label in enumerate(row_labels):
        row = results.loc[row_label]
        # second_group_values = np.array(list(top_group_values.values()))
        # second_group_values[second_group_values==0] = 0.001
        for j in range(len(row)):
            label = None
            if i == 0:
                label = column_labels[j]
            ax.bar(i, row[j], bottom=sum(row[:j]), label=label, edgecolor='black', color=colors[j], zorder=3)
        max_y_value = max(max_y_value, sum(row))

    if overlay:
        ax2.plot(np.arange(len(row_labels)), [overlay[k] for k in row_labels], color='black', linestyle='--', marker="*", zorder=4, linewidth=2)
        ax2.set_ylabel(overlay_ylabel)
        # print(max(list(overlay.values())))
        ax2.set_ylim([0, max(list(overlay.values()))*1.1])

    plt.xticks(np.arange(len(row_labels)), labels=row_labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim([0, max_y_value*1.1])
    ax.legend(ncol=len(column_labels), fontsize=12, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.2))
    ax.grid(visible=True, which='major', axis='y', color='#000000', linestyle='--', zorder=0)
    plt.tight_layout()
    plt.savefig(f"{filename}")
    plt.clf()
    
def plot_timeline(results, filename, xlabel="Hours", ylabel="Migrate Overhead (hrs)", cumulative=False, step=False, aggregate=False, scaling=1.0):
    if TEXT_ONLY:
        return
    
    plt.figure(figsize=(8, 4), dpi=300)
    max_y_value = - float('inf')
    if step:
        plot_func = plt.step
    else:
        plot_func = plt.plot
    for i, (plot_policy, series) in enumerate(results.items()):
        if cumulative:
            series = np.cumsum(series)
        series = np.array(series) * scaling
        plot_func(np.arange(len(series)), series, label=plot_policy, color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)], marker=markers[i % len(markers)])
        max_y_value = max(max_y_value, max(series))
    if aggregate:
        agg_series = np.sum(series for _, series in results.items())
        agg_series = np.array(agg_series) * scaling
        plot_func(np.arange(len(agg_series)), agg_series, label="total", color="purple", linewidth=2)
        max_y_value = max(max_y_value, max(agg_series))
    plt.legend(ncol=4, fontsize=12, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.2))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([0, TIMESTEPS])
    if max_y_value != 0 and max_y_value != - float('inf'):
        plt.ylim([0, 1.2*max_y_value])
    plt.grid(visible=True, which='major', axis='y', color='#000000', linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{filename}")
    plt.clf()
    
    
def plot_multi_timeline(multi_results, filename, xlabel="Hours", ylabel="Migrate Overhead (hrs)", cumulative=False, step=False, aggregate=False):
    if TEXT_ONLY:
        return
    
    num_subplots = len(multi_results)
    if num_subplots == 0:
        return
    fig = plt.figure(figsize=(10, 4*num_subplots), dpi=300)
    axes = fig.subplots(nrows=num_subplots, ncols=1)
    max_y_value = - float('inf')
    for i, (top_label, results) in enumerate(multi_results.items()):
        if num_subplots == 1:
            ax = axes
        else:
            ax = axes[i]
        if step:
            plot_func = ax.step
        else:
            plot_func = ax.plot
        for i, (second_label, series) in enumerate(results.items()):
            if cumulative:
                series = np.cumsum(series)
            plot_func(np.arange(len(series)), series, label=second_label, color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)], marker=markers[i % len(markers)])
            max_y_value = max(max_y_value, max(series))
        if aggregate:
            agg_series = np.sum(series for _, series in results.items())
            plot_func(np.arange(len(agg_series)), agg_series, label="total", color="purple", linewidth=2)
            max_y_value = max(max_y_value, max(agg_series))
        ax.legend(ncol=4, fontsize=12, loc="center")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(top_label)
        ax.set_xlim([0, TIMESTEPS])
        if max_y_value != 0 and max_y_value != - float('inf'):
            ax.set_ylim([0, 1.2*max_y_value])
        ax.grid(visible=True, which='major', axis='y', color='#000000', linestyle='--')
    
    plt.tight_layout()  
    plt.savefig(f"{filename}")
    plt.clf()

    
def plot_transition_graph(running_vms, app_profiles, migrations, evictions, completions, reschedule, power_traces, filename):
    if TEXT_ONLY:
        return
    
    vb_sites = power_traces.columns.tolist()
    power_per_vm = np.array([profile.app.cores_per_vm for profile in app_profiles])
    T, M, N = running_vms.shape
    frames = []
    frame_edge_labels = []
    frame_node_labels = []
    G= nx.DiGraph()
    for j in range(N):
        G.add_node(j)
    pos = nx.circular_layout(G)
    for t in range(T-1):
        Gt = G.copy()     
                        
        edge_labels = defaultdict(str)
        node_labels = defaultdict(str)
        for j0 in range(N):
            power = power_traces[vb_sites[j0]].iloc[t]
            next_power = power_traces[vb_sites[j0]].iloc[t+1]
            cur_util =  np.sum(np.matmul(power_per_vm, running_vms[t, :, j0]))
            
            node_labels[j0] = f"{vb_sites[j0]}\n{power : .2f} -> {next_power : .2f}W\n Utilized:{cur_util : .2f} W"
            
            if len(evictions[t][j0]) != 0 or len(completions[t][j0]) != 0 or len(reschedule[t][j0]) != 0:
                Gt.add_edge(j0, j0)
                
            if len(evictions[t][j0]) != 0:
                edge_labels[(j0, j0)] += f"Evictions:\n"
                for name, migrated_apps in evictions[t][j0].items():
                    edge_labels[(j0, j0)] += f"{name} x {migrated_apps:.1f}\n"
                    
            if len(completions[t][j0]) != 0:
                edge_labels[(j0, j0)] += f"Completion:\n"
                for name, migrated_apps in completions[t][j0].items():
                    edge_labels[(j0, j0)] += f"{name} x {migrated_apps:.1f}\n"
                    
            if len(reschedule[t][j0]) != 0:
                edge_labels[(j0, j0)] += f"Reschedule:\n"
                for name, migrated_apps in reschedule[t][j0].items():
                    edge_labels[(j0, j0)] += f"{name} x {migrated_apps:.1f}\n"
            
            edge_labels[(j0, j0)] = edge_labels[(j0, j0)][:-1]
            
            for j1 in range(N):
                if j0 == j1 or len(migrations[t][j0][j1]) == 0:
                    continue
                Gt.add_edge(j0, j1)
                for name, migrated_apps in migrations[t][j0][j1].items():
                    edge_labels[(j0, j1)] += f"{name} x {migrated_apps:.1f}\n"
                edge_labels[(j0, j1)] = edge_labels[(j0, j1)][:-1]
                
        frame_edge_labels.append(edge_labels)
        frame_node_labels.append(node_labels)
        frames.append(Gt)
        
    
    imagelist = []
    for t in range(T-1):
        G = frames[t]
        node_labels = frame_node_labels[t]
        edge_labels = frame_edge_labels[t]
   
        nx.set_node_attributes(G, {k: {'label': node_labels[k]} for k in node_labels.keys()})
        nx.set_edge_attributes(G, {(e[0], e[1]): {'label': edge_labels[(e[0], e[1])]} for e in G.edges(data=True)})
        A = to_agraph(G) 
        A.layout('dot')                                                             
        A.draw(f'tmp/step{t}.pdf')
        imagelist.append(f'tmp/step{t}.pdf') 
        
    merger = PdfMerger()
    for pdf in imagelist:
        merger.append(pdf)
    merger.write(filename)
    merger.close()
     
    for path in imagelist:
        os.remove(path)
     
     
def analysis(running_vms, app_profiles):
    T, M, N = running_vms.shape
    migrations = [[[defaultdict(int) for i in range(N)] for j in range(N)] for t in range(T-1)]
    evictions = [[defaultdict(int) for i in range(N)] for t in range(T-1)]
    completions = [[defaultdict(int) for i in range(N)] for t in range(T-1)]
    reschedule = [[defaultdict(int) for i in range(N)] for t in range(T-1)]
    migrated_cores = np.zeros(T-1)
    num_of_migrations = 0
    for t in range(T-1):
        for i in range(M):
            migrated = False
            if not app_profiles[i].finished:
                continue
            diff_vms = running_vms[t+1, i, :] - running_vms[t, i, :]
            for j0 in range(N):
                for j1 in range(N):
                    if diff_vms[j0] > 0 and diff_vms[j1] < 0:
                        migrated_vms = int(min(abs(diff_vms[j0]), abs(diff_vms[j1])))
                        diff_vms[j0] -= migrated_vms
                        diff_vms[j1] += migrated_vms
                        migrations[t][j1][j0][app_profiles[i].app.name+"-"+str(app_profiles[i].app.id)] += migrated_vms / app_profiles[i].app.num_vms 
                        migrated_cores[t] += migrated_vms * app_profiles[i].app.cores_per_vm
                        migrated = True
            for j0 in range(N):
                if diff_vms[j0] < 0:
                    migrated_vms = int(abs(diff_vms[j0]))
                    if app_profiles[i].end <= t + 1:
                        completions[t][j0][app_profiles[i].app.name+"-"+str(app_profiles[i].app.id)] += migrated_vms / app_profiles[i].app.num_vms
                    else:
                        evictions[t][j0][app_profiles[i].app.name+"-"+str(app_profiles[i].app.id)] += migrated_vms / app_profiles[i].app.num_vms
                        # FIXME: check why this counter does not work
                        # migrated_cores[t] += migrated_vms * app_profiles[i].app.cores_per_vm  
                if diff_vms[j0] > 0:
                    migrated_vms = int(abs(diff_vms[j0]))
                    if app_profiles[i].start < t:
                        reschedule[t][j0][app_profiles[i].app.name+"-"+str(app_profiles[i].app.id)] += migrated_vms / app_profiles[i].app.num_vms
                        migrated_cores[t] += migrated_vms * app_profiles[i].app.cores_per_vm 
                        migrated = True
            if migrated:
                num_of_migrations += 1 
                        
    return migrations, evictions, completions, reschedule, migrated_cores, num_of_migrations

def per_type_running_vms(running_vms, app_profiles):
    T, M, N = running_vms.shape
    all_types = []
    for i in range(M):
        app_type = app_profiles[i].app.name
        if app_type not in all_types:
            all_types.append(app_type)
            
    running_cores = np.zeros((T, len(all_types), N))
    running_vms_per_app = np.zeros((T, len(all_types), N))
    running_apps = np.zeros((T, len(all_types), N))
    for t in range(T):
        for j in range(N):
            for i in range(M):
                app_type = app_profiles[i].app.name
                type_index = all_types.index(app_type)
                running_cores[t, type_index, j] += running_vms[t, i, j] * app_profiles[i].app.cores_per_vm
                running_vms_per_app[t, type_index, j] += running_vms[t, i, j]
                # print(t, j, app_types[i][0],running_vms[t, i, j] / app_types[i][2])
                running_apps[t, type_index, j] += running_vms[t, i, j] / app_profiles[i].app.num_vms
    return running_cores, running_vms_per_app, running_apps, all_types
                
if __name__ == "__main__":
    overhead_types = ["Alloc", "Recompute", "Latency", "Blackout", "Queuing"]
    
    batch_dir=f"raw/default"
    if len(sys.argv) >= 2:
        batch_dir=f"{sys.argv[1]}"
        
    INTERP_FACTOR = 1
    if len(sys.argv) >= 3:
        INTERP_FACTOR=int(sys.argv[2])
        
    TEXT_ONLY = True
    

    # plot_policies = ["mip", 'mip-app', "greedy", "greedy-fair"]
    # base_policies = "avg_cov + cov + greedy + "
    # suffix_policies = " + 0.0"
    
    plot_policies = ["random", "avg_min", 'avg_cov',]
    base_policies = ""
    suffix_policies = " + geo + greedy + greedy + 0.0"
    
    # plot_policies = ["geo", 'stable', "cov"]
    # base_policies = "avg_cov + "
    # suffix_policies = " + greedy + greedy + 0.0"

    num_total_apps = 0
    TIMESTEPS = None
    num_total_graphs = 0

    date_dirs = glob.glob(f"{batch_dir}/day_*/", recursive = False)
    for date_dir in date_dirs:
        with open(f"{date_dir}/batch_info", "r") as f:
            batch_info = json.load(f)
            num_total_apps += batch_info['num_apps']
            if TIMESTEPS is None:
                TIMESTEPS = batch_info['interval']
            else:
                assert(TIMESTEPS == batch_info['interval'])
            num_total_graphs += batch_info['num_graphs'] 
        
    total_overhead_breakdown = defaultdict(lambda : defaultdict(int))
    total_affected_overhead_breakdown = defaultdict(lambda : defaultdict(int))
    total_migrated_cores = defaultdict(int)
    total_overheads = defaultdict(list)
    # total_execution_times = defaultdict(lambda : [0 for _ in range(num_total_apps)])
    total_overhead_percentages = defaultdict(list)
    total_migration_overhead_percentages = defaultdict(list)
    total_overheads_wo_blackout = defaultdict(list)
    total_migration_overheads = defaultdict(lambda : np.zeros(TIMESTEPS))
    total_num_of_migrations = defaultdict(int)
    all_power_traces = defaultdict(int)
    total_affected_apps = defaultdict(list)
    total_affected_apps_percentage_breakdown = defaultdict(lambda : defaultdict(int))
    bad_profiles = defaultdict(list)

    date_dirs = glob.glob(f"{batch_dir}/day_*/", recursive = False)
    for date_dir in date_dirs:
        date = int(date_dir.replace("/", " ").strip().split("_")[-1])
        with open(f"{date_dir}/batch_info", "r") as f:
            batch_info = json.load(f)
            num_apps = batch_info['num_apps']
            num_graphs = batch_info['num_graphs'] 
        
        for i in range(num_graphs):
            subgraph_dir = f"{date_dir}/subgraphs/graph_{i}"
            print(f"Plotting for {subgraph_dir}")
            all_overheads = dict()
            all_overheads_wo_blackout = dict()
            avg_overhead_breakdown = defaultdict(lambda : defaultdict(int))
            migration_overheads = dict()
            migration_overheads_per_overhead_type = defaultdict(dict)
            all_migrated_cores = dict()
            total_apps_per_subgraph = 0
            
            with open(f"{subgraph_dir}/power_traces", "rb") as f:
                power_traces = pickle.load(f)
                vb_sites = power_traces.columns
            
            plot_timeline(power_traces, f"{subgraph_dir}/power_traces.pdf", ylabel="Power (MW)", aggregate=True)
            # all_power_traces[f"day_{date}_subgraph_{i}"] = np.sum(power_traces, axis=1)
            all_power_traces[plot_policy] += np.sum(power_traces, axis=1)
                    
            for plot_policy in plot_policies:
                policy_dir = f"{subgraph_dir}/{base_policies}{plot_policy}{suffix_policies}"
                if not os.path.exists(policy_dir):
                    continue
                with open(f"{policy_dir}/running_vms", "rb") as f:
                    running_vms = pickle.load(f)
                with open(f"{policy_dir}/app_profiles", "rb") as f:
                    app_profiles = pickle.load(f).tolist()
                with open(f"{policy_dir}/global_profile", "rb") as f:
                    global_profile = pickle.load(f)
                    
                if len(app_profiles) == 0:
                    continue
                    
                for _, overheads in enumerate(global_profile.migration_distribution):
                    if type(overheads) == int:
                        global_profile.migration_distribution[_] = np.zeros(len(overhead_types))
                
                for profile in reversed(app_profiles):
                    
                    if not profile.finished:
                        profile.clear(TIMESTEPS)
                        bad_profiles[plot_policy].append(profile)

                    
                running_cores, running_vms_per_app, running_apps, all_types = per_type_running_vms(running_vms, app_profiles)
                
                T, M, N = running_vms.shape
                plot_multi_timeline(
                    {vb_sites[j]: {app_type : running_apps[:, t, j] for t, app_type in enumerate(all_types)} for j in range(N)},
                    f"{policy_dir}/running_vms.pdf", ylabel="Running Apps")
                        
                migrations, evictions, completions, reschedule, migrated_cores, num_of_migrations = analysis(running_vms, app_profiles)
                all_migrated_cores[plot_policy] = migrated_cores
                total_migrated_cores[plot_policy] += migrated_cores
                total_num_of_migrations[plot_policy] += num_of_migrations
                # plot_transition_graph(running_vms, app_profiles, migrations, evictions, completions, reschedule, power_traces, f"{policy_dir}/migration.pdf")

                num_apps_per_graph = len(app_profiles)
                # migration_overheads[plot_policy] = [np.sum(_) for _ in global_profile.migration_distribution]
                migration_overheads[plot_policy] = [sum([np.sum(app_profile.migration_distribution[t]) for app_profile in app_profiles if app_profile.migration_distribution[t] is not None]) for t in range(T)]

                # print(migration_overheads[plot_policy])
                total_migration_overheads[plot_policy] += np.array(migration_overheads[plot_policy])
                for i, otype in enumerate(overhead_types):
                    migration_overheads_per_overhead_type[plot_policy][otype] = [np.sum(overheads.reshape(len(overhead_types),-1)[i]) for overheads in global_profile.migration_distribution]

                per_app_type_overhead = defaultdict(list)
                per_app_overhead = []
                per_app_overhead_percentage = []
                per_app_migration_overhead_percentage = []
                per_app_overhead_wo_blackout = []
                # a workaround to keep the plot policies in the same order
                for o, otype in enumerate(overhead_types):
                    total_affected_overhead_breakdown[otype][plot_policy] += 0
                for p, profile in enumerate(app_profiles):
                    overhead_breakdown = profile.overhead_breakdown.tolist()+[int(profile.blackout), int(profile.queuing)]

                    per_app_type_overhead[profile.app.name].append(sum(overhead_breakdown))
                    per_app_overhead.append(sum(overhead_breakdown))
                    remap_id = lambda _id : (_id % 5) * (num_total_apps // 5) + _id // 5
                    # total_execution_times[plot_policy][remap_id(profile.app.id)] = sum(overhead_breakdown)
                    per_app_overhead_percentage.append(sum(overhead_breakdown) / profile.exec * 100)
                    per_app_migration_overhead_percentage.append(sum(overhead_breakdown[:-2]) / profile.exec * 100)
                    per_app_overhead_wo_blackout.append(sum(overhead_breakdown[:-2]))
                    for o, otype in enumerate(overhead_types):
                        avg_overhead_breakdown[otype][plot_policy] += overhead_breakdown[o] / num_apps_per_graph
                        total_overhead_breakdown[otype][plot_policy] += overhead_breakdown[o] / num_total_apps
                    if sum(overhead_breakdown) > 0:
                        total_affected_apps[plot_policy].append(profile)
                        total_affected_apps_percentage_breakdown[profile.app.name][plot_policy] += 1 / num_total_apps * 100
                        # print(plot_policy, sum(overhead_breakdown), sum(overhead_breakdown) / profile.exec * 100)
                        for o, otype in enumerate(overhead_types):
                            total_affected_overhead_breakdown[otype][plot_policy] += overhead_breakdown[o]
                    else:
                        # keep the label order
                        total_affected_apps_percentage_breakdown[profile.app.name][plot_policy] += 0

                plot_cdf(per_app_type_overhead, f"{policy_dir}/per_type_overheads.pdf")
                
                all_overheads[plot_policy] = per_app_overhead
                all_overheads_wo_blackout[plot_policy] = per_app_overhead_wo_blackout
                total_overheads[plot_policy].extend(per_app_overhead)
                total_overhead_percentages[plot_policy].extend(per_app_overhead_percentage)
                total_migration_overhead_percentages[plot_policy].extend(per_app_migration_overhead_percentage)
                total_overheads_wo_blackout[plot_policy].extend(per_app_overhead_wo_blackout)
                
            plot_cdf(all_overheads, f"{subgraph_dir}/cdf_of_all_overheads.pdf")
            plot_cdf(all_overheads_wo_blackout, f"{subgraph_dir}/cdf_of_all_overheads_wo_blackout.pdf")
            
            plot_grouped_bar(avg_overhead_breakdown, f"{subgraph_dir}/bar_blackout+overhead_all.pdf")
            plot_stacked_bar(avg_overhead_breakdown, f"{subgraph_dir}/bar_stacked_all.pdf")
            
            plot_timeline(migration_overheads, f"{subgraph_dir}/migration_overheads.pdf")
            plot_multi_timeline(migration_overheads_per_overhead_type, f"{subgraph_dir}/migration_overheads_per_type.pdf")
            
            plot_timeline(all_migrated_cores, f"{subgraph_dir}/migrated_cores.pdf", ylabel="Migrated cores", cumulative=True)
        
    TEXT_ONLY = False
        
    plot_timeline(all_power_traces, f"{batch_dir}/power_traces.pdf", ylabel="Power (MW)", aggregate=True)
    plot_grouped_bar(total_overhead_breakdown, f"{batch_dir}/bar_blackout+overhead_all.pdf")
    plot_stacked_bar(total_overhead_breakdown, f"{batch_dir}/bar_stacked_all.pdf")
    
    plot_timeline(total_migrated_cores, f"{batch_dir}/migrated_cores.pdf", ylabel="Migrated cores", cumulative=True)
    plot_cdf(total_overheads, f"{batch_dir}/cdf_of_all_overheads.pdf")
    plot_cdf(total_overhead_percentages, f"{batch_dir}/cdf_of_all_overhead_percentages.pdf", ylabel="Percentage (%)")
    # plot_dist({"greedy":total_execution_times["greedy"], "mip":total_execution_times["mip"]}, f"{batch_dir}/dist_of_overheads.pdf", ylabel="Overhead (hrs)")
    plot_cdf(total_overheads_wo_blackout, f"{batch_dir}/cdf_of_all_overheads_wo_blackout.pdf")
    # for plot_policy, values in total_overheads_wo_blackout.items():
    #     print(plot_policy, np.percentile(values, 90), np.percentile(values, 95), np.percentile(values, 99))
    plot_timeline(total_migration_overheads, f"{batch_dir}/migration_overheads.pdf")
    plot_timeline(total_migration_overheads, f"{batch_dir}/migration_overheads_cumulative.pdf", cumulative=True, scaling=1/num_total_apps)
    # print(total_overhead_breakdown)
    
    total_overhead_reports = defaultdict(dict)
    for plot_policy, values in total_overhead_percentages.items():
        # total_overhead_reports[plot_policy]['Avg\nOverhead'] = np.average(values) 
        # total_overhead_reports[plot_policy]['50%'] = np.percentile(values, 50)
        total_overhead_reports[plot_policy]['90 Percentile'] = 1 + np.percentile(values, 90) / 100
        total_overhead_reports[plot_policy]['95 Percentile'] = 1 + np.percentile(values, 95) / 100
        total_overhead_reports[plot_policy]['99 Percentile'] = 1 + np.percentile(values, 99) / 100
        # total_overhead_reports[plot_policy]['Average of Affected Apps'] = 1 + np.average([_ for _ in values if _ != 0]) / 100

    total_affected_migration_overhead_percentages = {k: np.average([_ for _ in v if _ != 0]) for k, v in total_migration_overhead_percentages.items()}
    total_affected_overhead_percentages = {k: np.average([_ for _ in v if _ != 0]) for k, v in total_overhead_percentages.items()}
    
    plot_grouped_bar(total_overhead_reports, f"{batch_dir}/statistics.pdf", ylabel="Performance Slowdown", ylim=[1,None])
    plot_grouped_bar(total_affected_overhead_breakdown, f"{batch_dir}/bar_blackout+overhead_affected.pdf")

    print(pd.DataFrame(total_affected_apps_percentage_breakdown).loc[plot_policies])
    plot_stacked_bar(pd.DataFrame(total_affected_apps_percentage_breakdown).loc[plot_policies], f"{batch_dir}/affected_breakdown.pdf", ylabel="Percentage of Apps (%)", overlay=total_affected_migration_overhead_percentages, overlay_ylabel="Migration Overhead (%)")

    total_overhead_reports = defaultdict(dict)
    for plot_policy, values in total_overhead_percentages.items():
        total_overhead_reports[plot_policy]['Avg Overhead'] = 1 + np.average(values) / 100
        # total_overhead_reports[plot_policy]['50%'] = np.percentile(values, 50)
        total_overhead_reports[plot_policy]['90 Percentile'] = 1 + np.percentile(values, 90) / 100
        total_overhead_reports[plot_policy]['95 Percentile'] = 1 + np.percentile(values, 95) / 100
        total_overhead_reports[plot_policy]['99 Percentile'] = 1 + np.percentile(values, 99) / 100
        total_overhead_reports[plot_policy]['Affected Apps'] = len(total_affected_apps[plot_policy]) / num_total_apps * 100
        total_overhead_reports[plot_policy]['Average overhead of Affected Apps'] = 1 + np.average([_ for _ in values if _ != 0]) / 100
        total_overhead_reports[plot_policy]['Average migration overhead of Affected Apps'] = 1 + total_affected_migration_overhead_percentages[plot_policy] / 100
        total_overhead_reports[plot_policy]['Total Migrations'] = total_num_of_migrations[plot_policy]

        for otype in overhead_types:
            total_affected_overhead_breakdown[otype][plot_policy] /= len(total_affected_apps[plot_policy])
    
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    unfinished_apps = {k:{"Unfinished apps": len(v)} for k,v in bad_profiles.items()}
    print(pd.DataFrame(unfinished_apps))
    print(pd.DataFrame(total_overhead_reports))
    print(pd.DataFrame(total_overhead_breakdown))
    print(pd.DataFrame(total_affected_overhead_breakdown))

    output_dir = f"{batch_dir}/output/" 
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(f'{output_dir}/summary', 'wb') as f:
        pickle.dump(total_overhead_reports, f)
                