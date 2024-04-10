from macros import *
from site_selection import SiteSelection, SubGraphSelection
from app_placement import AppPlacement, PLACEMENT_MIP
from parsing import utility, simulator_validation
from renewable_simulation import BATCH_MIP
from vm_placement import VMPlacement
from factory import VMFactory

plt.rcParams.update({'figure.max_open_warning': 0})
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'font.family': 'serif'})
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'
matplotlib.rcParams['hatch.linewidth'] = 0.5

line_styles = ['-', '--', '-.', ':']
hatches = ["", "\\", "//", "||"]
colors = ['#ff796c', 'plum', '#95d0fc', 'gray']
line_colors = ['#ff796c', 'plum', '#23a8eb', 'gray', 'black']
markers = ['o', '*', 'v', '^']
plt.figure(figsize=(4,5), dpi=300)


avg_server_power = 500 # kwatts/server
avg_server_carbon = 1220.87 # kgco2/server
N_SITES = int(sys.argv[1])

def cal_embodied_carbon(trace):
    subgraphs_power = trace.max().sum()
    subgraphs_num_server = subgraphs_power * 1000 // avg_server_power
    subgraphs_embodied_carbon = subgraphs_num_server * avg_server_carbon
    centralized_power = trace.sum(axis=1).mean()
    centralized_num_server = centralized_power * 1000 // avg_server_power
    centralized_embodied_carbon = centralized_num_server * avg_server_carbon
    return subgraphs_embodied_carbon, centralized_embodied_carbon

vb_coords = read_vb_coords()
capacity = read_vb_capacity()
renewable_traces, vb_sites = read_traces(vb_coords, capacity, interp_factor=1)

site_policies = ["no_prunning", "avg_cov", "cov", "max_avg"]
site_selector = SiteSelection(vb_sites, vb_coords, capacity, renewable_traces, N_SITES, 1)

s_ec, c_ec, ratio = [], [], []
for site_policy in site_policies:
    if site_policy == "no_prunning":
        subgraphs_ec, centralized_ec = cal_embodied_carbon(renewable_traces)
    else:
        start_time, end_time = 0, 0 + 7 * 48
        policy = {"site" : site_policy, "n_sites": N_SITES}
        selected_sites, total_capacity, scaled_traces = site_selector.select(policy, start_time, end_time)    
        subgraphs_ec, centralized_ec = cal_embodied_carbon(renewable_traces[selected_sites])
    s_ec.append(subgraphs_ec)
    c_ec.append(centralized_ec)
    ratio.append(subgraphs_ec / centralized_ec)
    print(f"{site_policy} overprovisioning: {subgraphs_ec / centralized_ec}")

width = 0.2
gap = 0.25
policies = ['Centralized', 'Baseline', "GreenBox"]
ratios = [1, ratio[0], ratio[3]]
colorss = [colors[2],colors[0],colors[1]]
for i in range(3):
    plt.bar((2+i)*gap, ratios[i], width=width, label=policies[i], color=colorss[i], edgecolor="black", zorder=3)
plt.xlabel('Pruning Policies')
plt.ylabel('Normalized Embodied Carbon Emission')
plt.xticks([]) 
    
plt.legend(ncol=3, bbox_to_anchor=(0.5, 1.1), loc='upper center', frameon=False, fontsize=11, columnspacing = 0.8)
plt.grid(which='major', axis='y', linestyle='--',linewidth=1)
plt.tight_layout()
plt.savefig("tmp/embodied_carbon.png")

