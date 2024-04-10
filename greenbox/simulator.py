from macros import *

from site_selection import SiteSelection, SubGraphSelection
from vm_placement import VMPlacement
from factory import VMFactory

if __name__ == "__main__":
    
    ## parameters 
    
    N_SITES = int(sys.argv[1])
    MAX_K = 3
    interp_factor = 1
    INTERVAL = HOURS / interp_factor
    NUM_SUBGRAPHS = N_SITES // MAX_K
    SUBGRAPH_OFFSET = 0
    CLOUD_UTIL = float(sys.argv[8]) / 100
    MAX_CUTOFF = 1.0
    LOW_CUTOFF = 0.0
    RUNNING_HOURS = 2*24
    # RUNNING_HOURS = 24
    START_DATE = 0*24
    REPEAT = 1
    GAP_BETWEEN_REPEAT = 2*24
    MISPREDICTION_RATIO = 0.0
    SUBGRAPH_FACTOR = 0.0
    SLO_AVAIL = float(sys.argv[2]) / 100
    POWER_MIS = float(sys.argv[3]) / 100
    LIFE_MIS = float(sys.argv[4]) / 100
    DIST = float(sys.argv[5]) / 100
    START = int(sys.argv[6]) * 24
    LOOKAHEAD = int(sys.argv[7])
    STATIC = False
    STEP_BY_STEP = False
    if STATIC:
        name = f'site{N_SITES}_slo{sys.argv[2]}_powermis{sys.argv[3]}_lifemis{sys.argv[4]}_dist{sys.argv[5]}_start{sys.argv[6]}_lookahead{sys.argv[7]}_UTIL{sys.argv[8]}_static'
    elif STEP_BY_STEP:
        name = f'site{N_SITES}_slo{sys.argv[2]}_powermis{sys.argv[3]}_lifemis{sys.argv[4]}_dist{sys.argv[5]}_start{sys.argv[6]}_lookahead{sys.argv[7]}_UTIL{sys.argv[8]}_step'
    else:
        name = f'site{N_SITES}_slo{sys.argv[2]}_powermis{sys.argv[3]}_lifemis{sys.argv[4]}_dist{sys.argv[5]}_start{sys.argv[6]}_lookahead{sys.argv[7]}_UTIL{sys.argv[8]}'
    output_file = f"tmp/output/{name}.txt"
    open(output_file, "w").close()

    CORE_TO_POWER_RATIO = 0.5
    MIP_OPTIONS = {"BATCH_MIP_OPT_TIME" : 120.0,
                    "BATCH_MIP_OPT_GOAL" : 0.05,
                    "PLACEMENT_MIP_OPT_TIME" : 120,
                    "PLACEMENT_MIP_OPT_GOAL" : 0.05,
                    "SLO_AVAIL" : SLO_AVAIL}
    
    for sid in range(NUM_SUBGRAPHS):
        logfile = f"logs/mip-model-{sid}.log"
        f = open(logfile, "w").close()

    vb_coords = read_vb_coords()
    capacity = read_vb_capacity()
    renewable_traces, vb_sites = read_traces(vb_coords, capacity, interp_factor)

    ray.init()
    ray_head_service_host = os.getenv("RAY_HEAD_SERVICE_HOST")
    ray_head_service_port_redis_primary = os.getenv("RAY_HEAD_SERVICE_PORT_REDIS_PRIMARY")
    ray_address = f"{ray_head_service_host}:{ray_head_service_port_redis_primary}"
    log_msg(f'Initializing Ray on cluster at address {ray_address}')

    
    batch_name = "site_selection_stable_apps_resource_mix_random"
    # site_policies = ["random", "avg_cov", "cov", "max_avg", "avg_max_avg"]
    site_policies = ["avg_cov"]
    subgraph_policies =  ['cov'] 
    placement_policies = ["greedy"]
    # policies = [ ("avg_cov", "cov")]
    scheduling_policies = ["greedy", "mip-app", "mip"]
    all_policies = list(itertools.product(*[site_policies, subgraph_policies, placement_policies, scheduling_policies]))
    if STEP_BY_STEP:
        all_policies = [["random", "cov", "greedy", "mip"], ["avg_cov", "geo", "greedy", "mip"], ["avg_cov", "cov", "greedy", "greedy"], ["avg_cov", "cov", "greedy", "mip"]]
    
    site_selector = SiteSelection(vb_sites, vb_coords, capacity, renewable_traces, 84, interp_factor)
    # site_selector = SiteSelection(vb_sites, vb_coords, capacity, renewable_traces, N_SITES, interp_factor)
    subgraph_selector = SubGraphSelection(vb_coords, capacity, renewable_traces, N_SITES, interp_factor)
    scheduler = VMPlacement(renewable_traces, N_SITES, INTERVAL)
    factory = VMFactory()

    baseline_capacity = None
    default_workloads = defaultdict(lambda : None)
    previous_subgraphs = defaultdict(lambda : [])
    default_sites = ['UKK2-wind', 'FRH0-wind', 'DK05-wind','DK04-wind', 'FRD1-wind', 'FRG0-wind']

    for ps in all_policies:
        site_policy, subgraph_policy, placement_policy, scheduling_policy = ps
        start_time, end_time = START_DATE, START_DATE + 7 * 24 * interp_factor
        policy = {"site" : site_policy, "n_sites": N_SITES}
        log_msg(f"====== Site selection policy {site_policy} ======")
        selected_sites, total_capacity, scaled_traces = site_selector.select(policy, start_time, end_time, capacity_scaling=baseline_capacity, low_cutoff=LOW_CUTOFF, max_cutoff=MAX_CUTOFF, core_to_power_ratio=CORE_TO_POWER_RATIO, batch_name=batch_name)

        if N_SITES == 24:
            selected_sites = ['UKK3-wind', 'PL84-wind', 'SE22-wind', 'DK05-wind', 'UKK4-wind', 'PL62-wind', 'FRI3-wind', 'PT20-wind', 'FRM0-wind', 'FRG0-wind', 'UKI4-wind', 'IE05-wind', 'DK04-wind', 'NO07-wind', 'ES11-wind', 'UKJ4-wind', 'IS00-wind', 'HR03-wind', 'FRB0-wind', 'UKE1-wind', 'FI1C-wind', 'DK01-wind', 'FRL0-wind', 'PL71-wind']
        elif N_SITES == 6:
            selected_sites = ['UKK2-wind', 'FRH0-wind', 'DK05-wind','DK04-wind', 'FRD1-wind', 'FRG0-wind']
        elif N_SITES == 12:
            selected_sites = ['UKK3-wind', 'NO07-wind', 'FRD1-wind', 'UKJ4-wind', 'UKM6-wind', 'NO02-wind', 'IE05-wind', 'UKH1-wind', 'SE22-wind', 'IS00-wind', 'UKH3-wind', 'FRD2-wind']

        policy['subgraph'] = subgraph_policy
        log_msg(f"====== Subgraph selection policy {subgraph_policy} ======")

        subgraphs = []
        scheduling_interval = RUNNING_HOURS * interp_factor * 1
        interval = GAP_BETWEEN_REPEAT * interp_factor
        # start = START_DATE * interp_factor
        start = START * interp_factor
        for start_time in range(start, start + REPEAT * interval, interval):
            end_time = start_time + scheduling_interval
            for SUBGRAPH_FACTOR in [0.0]:#[1,2,3,4,N_SITES]:
                policy['SUBGRAPH_FACTOR'] = SUBGRAPH_FACTOR
                
                subgraphs = subgraph_selector.select(policy, 0, scheduling_interval, selected_sites, scaled_traces=scaled_traces, previous_subgraphs = previous_subgraphs[SUBGRAPH_FACTOR], factor=SUBGRAPH_FACTOR, max_k = MAX_K, batch_name=batch_name, use_cache=False)
                selected_sites = []
                for i in range(N_SITES // 3):
                    for site in subgraphs[i][0]:
                        selected_sites.append(site)
                subgraphs = subgraph_selector.select(policy, start_time, end_time, selected_sites, scaled_traces=scaled_traces, previous_subgraphs = previous_subgraphs[SUBGRAPH_FACTOR], factor=SUBGRAPH_FACTOR, max_k = MAX_K, batch_name=batch_name, use_cache=False)
                subgraphs = subgraphs[SUBGRAPH_OFFSET:SUBGRAPH_OFFSET+NUM_SUBGRAPHS]
                previous_subgraphs[SUBGRAPH_FACTOR] = [] #subgraphs
                print(subgraphs)

                end_time = start_time + scheduling_interval
                scaled_traces = scaled_traces.loc[start_time:end_time]
                if default_workloads[start_time] is None:
                    aggregated_avg_power = sum([np.average(scaled_traces[site]) for site in selected_sites])
                    total_energy = RUNNING_HOURS * aggregated_avg_power
                    print(total_energy)
                    workloads = factory.create_real_azure_workloads_with_energy(RUNNING_HOURS, total_energy, CLOUD_UTIL, LIFE_MIS, DIST)
                    print(f"number of VMs: {len(workloads)}")
                    default_workloads[start_time] = workloads
                
                policy['placement'] = placement_policy
                log_msg(f"====== Placement policy {placement_policy} ======")
                policy['scheduling'] = scheduling_policy
                log_msg(f"====== Scheduling policy {scheduling_policy} ======", header=True)
                scheduler.placement(policy, start_time, end_time, RUNNING_HOURS, subgraphs, default_workloads[start_time], scaled_traces=scaled_traces, use_ray=True, cloud_util=CLOUD_UTIL, batch_name=batch_name, mip_options=MIP_OPTIONS, power_mis_ratio=POWER_MIS, name=name, lahead=LOOKAHEAD)
            

    # for site_policy in site_policies:
    #     # consider 8 weeks
    #     # sites = dict()
    #     # policy = {"site" : site_policy, "n_sites": N_SITES}
    #     # for start in range(0, 49+1, 7):
    #     #     curr_date = start * 24
    #     #     start_time, end_time = curr_date, curr_date + 7 * 24 * interp_factor
    #     #     selected_sites, total_capacity, scaled_traces = site_selector.select(policy, start_time, end_time, capacity_scaling=baseline_capacity, low_cutoff=LOW_CUTOFF, max_cutoff=MAX_CUTOFF, core_to_power_ratio=CORE_TO_POWER_RATIO, batch_name=batch_name)
    #     #     for idx in range(len(selected_sites)):
    #     #         s = selected_sites[idx]
    #     #         if s in sites:
    #     #             sites[s] += len(selected_sites) - idx
    #     #         else:
    #     #             sites[s] = len(selected_sites) - idx
    #     # sites = [k for k,v in sorted(sites.items(), key=lambda x: x[1], reverse=True)]
    #     # selected_sites = sites[:48]

    #     start_time, end_time = START_DATE, START_DATE + 7 * 24 * interp_factor
    #     policy = {"site" : site_policy, "n_sites": N_SITES}
    #     log_msg(f"====== Site selection policy {site_policy} ======")
    #     selected_sites, total_capacity, scaled_traces = site_selector.select(policy, start_time, end_time, capacity_scaling=baseline_capacity, low_cutoff=LOW_CUTOFF, max_cutoff=MAX_CUTOFF, core_to_power_ratio=CORE_TO_POWER_RATIO, batch_name=batch_name)

    #     if N_SITES == 24:
    #         selected_sites = ['UKK3-wind', 'PL84-wind', 'SE22-wind', 'DK05-wind', 'UKK4-wind', 'PL62-wind', 'FRI3-wind', 'PT20-wind', 'FRM0-wind', 'FRG0-wind', 'UKI4-wind', 'IE05-wind', 'DK04-wind', 'NO07-wind', 'ES11-wind', 'UKJ4-wind', 'IS00-wind', 'HR03-wind', 'FRB0-wind', 'UKE1-wind', 'FI1C-wind', 'DK01-wind', 'FRL0-wind', 'PL71-wind']
    #         # selected_sites = ['UKK3-wind', 'NO07-wind', 'FRH0-wind', 'UKJ4-wind', 'NO02-wind', 'DK04-wind', 'UKK4-wind', 'DK05-wind', 'PL62-wind', 'IE05-wind', 'FRG0-wind', 'SE21-wind', 'UKM6-wind', 'UKE2-wind', 'FI1B-wind', 'SE22-wind', 'EE00-wind', 'UKJ2-wind', 'IE04-wind', 'FI1D-wind', 'DK03-wind', 'UKL1-wind', 'FRD1-wind', 'DK01-wind']
    #     elif N_SITES == 6:
    #         selected_sites = ['UKK2-wind', 'FRH0-wind', 'DK05-wind','DK04-wind', 'FRD1-wind', 'FRG0-wind']
    #     elif N_SITES == 12:
    #         # selected_sites = ['FRH0-wind', 'NO07-wind', 'UKK3-wind', 'DK05-wind', 'UKK4-wind', 'PL84-wind', 'FRI3-wind', 'PT20-wind', 'FRM0-wind', 'DK04-wind', 'PT30-wind', 'ES11-wind']
    #         # selected_sites = ['UKH3-wind', 'IS00-wind', 'NO07-wind', 'UKM6-wind', 'FRG0-wind', 'CH01-wind', 'UKJ4-wind', 'NO06-wind', 'EL41-wind', 'UKK4-wind', 'FRK2-wind', 'EL51-wind']
    #         selected_sites = ['UKK3-wind', 'NO07-wind', 'FRD1-wind', 'UKJ4-wind', 'UKM6-wind', 'NO02-wind', 'IE05-wind', 'UKH1-wind', 'SE22-wind', 'IS00-wind', 'UKH3-wind', 'FRD2-wind']
    #     # scale based on the random capacity
    #     if site_policy == "random":
    #         baseline_capacity = total_capacity
                

    #     for subgraph_policy in subgraph_policies:
    #         if (site_policy, subgraph_policy) not in policies:
    #             continue
    #         policy['subgraph'] = subgraph_policy
    #         log_msg(f"====== Subgraph selection policy {subgraph_policy} ======")

    #         subgraphs = []
    #         scheduling_interval = RUNNING_HOURS * interp_factor * 1
    #         interval = GAP_BETWEEN_REPEAT * interp_factor
    #         # start = START_DATE * interp_factor
    #         start = START * interp_factor
    #         for start_time in range(start, start + REPEAT * interval, interval):
    #             end_time = start_time + scheduling_interval
    #             for SUBGRAPH_FACTOR in [0.0]:#[1,2,3,4,N_SITES]:
    #                 policy['SUBGRAPH_FACTOR'] = SUBGRAPH_FACTOR
                    
    #                 subgraphs = subgraph_selector.select(policy, 0, scheduling_interval, selected_sites, scaled_traces=scaled_traces, previous_subgraphs = previous_subgraphs[SUBGRAPH_FACTOR], factor=SUBGRAPH_FACTOR, max_k = MAX_K, batch_name=batch_name, use_cache=False)
    #                 selected_sites = []
    #                 for i in range(N_SITES // 3):
    #                     for site in subgraphs[i][0]:
    #                         selected_sites.append(site)
    #                 subgraphs = subgraph_selector.select(policy, start_time, end_time, selected_sites, scaled_traces=scaled_traces, previous_subgraphs = previous_subgraphs[SUBGRAPH_FACTOR], factor=SUBGRAPH_FACTOR, max_k = MAX_K, batch_name=batch_name, use_cache=False)
    #                 subgraphs = subgraphs[SUBGRAPH_OFFSET:SUBGRAPH_OFFSET+NUM_SUBGRAPHS]
    #                 previous_subgraphs[SUBGRAPH_FACTOR] = [] #subgraphs
    #                 print(subgraphs)
                    
    #                 # Overprovisioning
    #                 # selected_sites = selected_sites[:3]
    #                 ss = sum([max(scaled_traces[s]) for s in selected_sites])
    #                 c = sum([scaled_traces[s] for s in selected_sites]).max()
    #                 if c == 0.:
    #                     continue
    #                 overprovisioning = ss / c
    #                 embodied_carbon = cal_embodied_carbon(scaled_traces[selected_sites])
    #                 print(f"overprovisioning:{overprovisioning}")
    #                 print("Embodied Carbon", embodied_carbon)
                    
    #                 # static
    #                 # tmp = list(subgraphs[0])
    #                 # tmp[0] = ['UKK2-wind', 'FRH0-wind', 'DK04-wind']
    #                 # tmp2 = list(subgraphs[1])
    #                 # tmp2[0] = ['DK05-wind', 'FRD1-wind', 'FRG0-wind']
    #                 # subgraphs = []
    #                 # subgraphs.append(tmp)
    #                 # subgraphs.append(tmp2)


    #                 # covs = []
    #                 # for subgraph in subgraphs:
    #                 # # subgraph = subgraphs[0]
    #                 #     current_trace = scaled_traces.loc[start_time:end_time]
    #                 #     sum_trace = current_trace[subgraph[0]].sum(axis=1)
    #                 #     avg_p = sum(sum_trace) / (end_time - start_time)
    #                 #     cov_p = np.std(sum_trace) / avg_p
    #                 #     covs.append(cov_p)
    #                 #     print(f"cov:{sum(covs) / len(covs)}")
    #                 #     exit()
    #                 # a = ['UKK2-wind', 'FRH0-wind', 'DK05-wind']
    #                 # b = ['DK04-wind', 'FRD1-wind', 'FRG0-wind']
    #                 # c = []
    #                 # c.append(a)
    #                 # c.append(b)
    #                 # covs = []
    #                 # for subgraph in c:
    #                 #     current_trace = scaled_traces.loc[start_time:end_time]
    #                 #     sum_trace = current_trace[subgraph].sum(axis=1)
    #                 #     avg_p = sum(sum_trace) / (end_time - start_time)
    #                 #     cov_p = np.std(sum_trace) / avg_p
    #                 #     covs.append(cov_p)
    #                 # print(f"cov:{sum(covs) / len(covs)}")
    #                 # exit()

    #                 end_time = start_time + scheduling_interval
    #                 scaled_traces = scaled_traces.loc[start_time:end_time]
    #                 if default_workloads[start_time] is None:
    #                     aggregated_avg_power = sum([np.average(scaled_traces[site]) for site in selected_sites])
    #                     # workloads = factory.create_workloads_with_energy(aggregated_avg_power, INTERVAL, CLOUD_UTIL, -0.2)
    #                     # workloads = factory.create_real_workloads_with_energy(aggregated_avg_power, INTERVAL, CLOUD_UTIL)
    #                     total_energy = RUNNING_HOURS * aggregated_avg_power
    #                     print(total_energy)
    #                     workloads = factory.create_real_azure_workloads_with_energy(RUNNING_HOURS, total_energy, CLOUD_UTIL, LIFE_MIS, DIST)
    #                     print(f"number of VMs: {len(workloads)}")
    #                     default_workloads[start_time] = workloads
                    
    #                 nr_power = 0.
    #                 for i in workloads:
    #                     lifetime = int(min(i.lifetime / HOURS, RUNNING_HOURS))
    #                     nr_power += i.maxpower[:lifetime].sum()
    #                 print(f"central nr_power: {nr_power}, central embodied: {embodied_carbon / overprovisioning}")

    #                 for placement_policy in placement_policies:
    #                     policy['placement'] = placement_policy
    #                     log_msg(f"====== Placement policy {placement_policy} ======")
    #                     for scheduling_policy in scheduling_policies:
    #                         policy['scheduling'] = scheduling_policy
    #                         log_msg(f"====== Scheduling policy {scheduling_policy} ======", header=True)
    #                         scheduler.placement(policy, start_time, end_time, RUNNING_HOURS, subgraphs, default_workloads[start_time], scaled_traces=scaled_traces, use_ray=True, cloud_util=CLOUD_UTIL, batch_name=batch_name, mip_options=MIP_OPTIONS, power_mis_ratio=POWER_MIS, name=name, lahead=LOOKAHEAD)
                
#site_selector.plot_all()
#subgraph_selector.plot_all()
