import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from macros import *


plt.rcParams.update({'figure.max_open_warning': 0})
matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams.update({'font.family': 'serif'})
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'
matplotlib.rcParams['hatch.linewidth'] = 0.5

def read_vb_coords():
    vb_coords = dict()
    with open(f'{DATA_DIR}/nuts2.json') as f:
        vb_coords = json.load(f)
    return vb_coords

def read_vb_capacity():
    capacity = dict()
    with open(f"{DATA_DIR}/coeff_vb.json", "r") as power_f:
        capacity.update(json.load(power_f))
    return capacity

def read_traces(vb_coords, capacity, interp_factor = 1):
    all_types = [ "wind"]
    # all_types = ["solar", "wind"]
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
                renewable_trace[vb_name] *= capacity[vb_name]

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

def cal_carbon(rewewable_traces, thr=0.9, brown_thr = 0.1, server_power=0.5, server_carbon=0.7, start_time = 10*24, time_span=24*7, brown_carbon=0.0005, wind_carbon=0.000011, solar_carbon=0.000041):
    '''
    server_power unit is kw/server
    server_carbon unit is tCO2e/server
    time_span unit is hour
    brown_carbon unit is tCO2e/kwh
    '''
    max_power = renewable_traces.max()
    threshold = max_power * thr
    brown_threshold = max_power * brown_thr

    # calculate the total embodied carbon for servers
    num_servers = threshold // server_power
    server_carbon_total = num_servers * server_carbon 
    server_carbon_total = server_carbon_total / (5*365*24) * time_span
    
    # calculate brown operational carbon needed
    delta = renewable_traces - brown_threshold
    delta[delta>0] = 0
    delta = delta.abs()
    brown_power = delta.abs()
    brown_energy = brown_power[start_time:start_time+time_span].sum() #kwh
    brown_carbon_total = brown_energy * brown_carbon 
    
    # calculate renewable carbon needed
    renewable_power = renewable_traces[renewable_traces<=threshold]
    renewable_energy = renewable_power[start_time:start_time+time_span].sum()
    renewable_carbon_total = renewable_energy
    for farm, energy in renewable_energy.items():
        if farm[-4:] == "wind":
            renewable_carbon_total[farm] *= wind_carbon
        else:
            renewable_carbon_total[farm] *= solar_carbon
    
    # add brown and renewable
    operation_carbon_total = renewable_carbon_total + brown_carbon_total
    return server_carbon_total, operation_carbon_total

def cal_cov(rewewable_traces, thr=0.9, server_carbon=0.7, start_time = 10*24, time_span=24*7):
    sliced_renewable_trace = rewewable_traces[start_time:start_time+time_span]
    cov_site = {} 
    for site in sliced_renewable_trace.keys():
        avg_power = np.sum(sliced_renewable_trace[site]) / len(sliced_renewable_trace)
        cov_power = np.std(sliced_renewable_trace[site]) / avg_power
        cov_site[site] = cov_power
    return cov_site
    
vb_coords = read_vb_coords()
capacity = read_vb_capacity()
renewable_traces, vb_sites = read_traces(vb_coords, capacity, 1)

server_carbon, oper_carbon = cal_carbon(renewable_traces)
site_cov = cal_cov(renewable_traces)
server_carbon.sort_values().plot()
oper_carbon.sort_values().plot()
plt.savefig("plots/brown_motivation.png")
plt.savefig("plots/brown_motivation.pdf")

plt.clf()

embodied_carbon_percentage = {}
carbon_distribution = {}
carbon_ratio = {}
for site, embodied_carbon in server_carbon.items():
    opex = oper_carbon[site]
    capax = embodied_carbon
    if capax + opex > 0:
        embodied_carbon_percentage[site] = opex / (capax + opex) * 100
        carbon_distribution[site] = (opex, capax)
        if capax > 0 and opex > 0:
            carbon_ratio[site] = opex / capax
    else:
        embodied_carbon_percentage[site] = 0
    
carbon_distribution = pd.DataFrame(carbon_distribution)
carbon_distribution.sort_values(by=0, ascending=True, axis=1, inplace=True)
carbon_distribution.iloc[0, :].plot()
carbon_distribution.iloc[1, :].plot()
plt.savefig("plots/brown_motivation_filtered.png")
plt.savefig("plots/brown_motivation_filtered.pdf")
plt.clf()


sites, carbon_ratios = zip(*sorted(carbon_ratio.items(), key=lambda x: x[1]))
xlabels = np.arange(len(sites)) / len(sites) * 100

fig,ax = plt.subplots(figsize=(4, 3))
p1 = ax.plot(xlabels, carbon_ratios, label="Additional Carbon Footprint", zorder=3)
ax.set_yticks(np.arange(1,6,1))
ax.set_yticklabels([str(_)+"x" for _ in np.arange(1,6,1)])
ax.set_xlabel("rMDCs (%)")
ax.set_ylabel("Additional Carbon Footprint")
ax2 = ax.twinx()
p2 = ax2.plot(xlabels, [site_cov[site] for site in sites], ls="-", color="orange", label="CoV of Power Supply", zorder=3)
ax2.set_ylim([0,3])
ax2.set_ylabel("CoV of Power Supply")

lns = p1 + p2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)
plt.grid(ls='--', axis='y', zorder=4)
plt.tight_layout()
plt.savefig("plots/brown_motivation_ratio.png")
plt.savefig("plots/brown_motivation_ratio.pdf")
plt.clf()

operational_dist = {}
embodied_dist= {}
for brown_thr in [0.1, 0.2, 0.3, 0.4, 0.5]:
    carbon_ratio = {}
    server_carbon, oper_carbon = cal_carbon(renewable_traces, brown_thr=brown_thr)
    for site, embodied_carbon in server_carbon.items():
        opex = oper_carbon[site]
        capax = embodied_carbon
        if capax + opex > 0:
            if capax > 0:
                carbon_ratio[opex] = capax
    # avg_carbon_ratio = sum(carbon_ratio.keys()) / sum(carbon_ratio.values())
    embodied_dist[int(brown_thr * 100)] = sum(carbon_ratio.values())
    operational_dist[int(brown_thr * 100)] = sum(carbon_ratio.keys())
    
print(operational_dist)
width = 5
plt.bar(list(embodied_dist.keys()), list(embodied_dist.values()), width=width, label="Embodied")
plt.bar(list(operational_dist.keys()), list(operational_dist.values()), bottom=list(embodied_dist.values()), width=width, label="Operational", zorder=3)
plt.plot(list(operational_dist.keys()), np.array(list(embodied_dist.values())) + np.array(list(operational_dist.values())), color="black", marker="*", ms=4, lw=1, ls="--", zorder=3)
plt.xlabel("Percentages of Guaranteed Available Servers (%)")
plt.ylabel("Total Carbon Emission (tCO2eq)")
plt.legend()
plt.grid(ls='--', axis='y', zorder=4)
plt.tight_layout()
plt.savefig("plots/total_ratio.png")
plt.savefig("plots/total_ratio.pdf")
