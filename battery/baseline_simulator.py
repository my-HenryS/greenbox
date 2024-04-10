from macros import *

from site_selection import SiteSelection, SubGraphSelection
from app_placement import AppPlacement, PLACEMENT_MIP
from parsing import utility, simulator_validation
from renewable_simulation import BATCH_MIP
from central_placement import CentralPlacement
from factory import VMFactory


if __name__ == "__main__":
    
    ## parameters     
    N_SITES = int(sys.argv[1])
    MAX_K = 3
    interp_factor = 1
    INTERVAL = HOURS / interp_factor
    NUM_SUBGRAPHS = N_SITES // MAX_K
    SUBGRAPH_OFFSET = 0
    CLOUD_UTIL = 0.9
    RUNNING_HOURS = 72
    SCHEDULING_HOURS = RUNNING_HOURS + 8
    START_DATE = 10*24
    MISPREDICTION_RATIO = 0.0

    MIP_OPTIONS = {"BATCH_MIP_OPT_TIME" : 30.0,
                    "BATCH_MIP_OPT_GOAL" : 0.05,
                    "PLACEMENT_MIP_OPT_TIME" : 120,
                    "PLACEMENT_MIP_OPT_GOAL" : 0.05,
                    "SLO_AVAIL" : 0.8}

    vb_coords = read_vb_coords()
    capacity = read_vb_capacity()
    renewable_traces, vb_sites = read_traces(vb_coords, capacity, interp_factor)

    ray.init()
    ray_head_service_host = os.getenv("RAY_HEAD_SERVICE_HOST")
    ray_head_service_port_redis_primary = os.getenv("RAY_HEAD_SERVICE_PORT_REDIS_PRIMARY")
    ray_address = f"{ray_head_service_host}:{ray_head_service_port_redis_primary}"
    log_msg(f'Initializing Ray on cluster at address {ray_address}')
    
    batch_name = "central"
    site_policies = ["random", "avg_cov"]
    subgraph_policies =  ['cov'] 
    placement_policies = ["greedy"]
    policies = [ ("avg_cov", "cov")]
    scheduling_policies = ["greedy"]
    
    site_selector = SiteSelection(vb_sites, vb_coords, capacity, renewable_traces, N_SITES, interp_factor)
    subgraph_selector = SubGraphSelection(vb_coords, capacity, renewable_traces, N_SITES, interp_factor)
    scheduler = CentralPlacement(renewable_traces, N_SITES, INTERVAL)
    factory = VMFactory()

    default_workloads = defaultdict(lambda : None)
    for site_policy in site_policies:
        start_time, end_time = 0, 0 + 7 * 24 * interp_factor
        policy = {"site" : site_policy, "n_sites": N_SITES}
        log_msg(f"====== Site selection policy {site_policy} ======")
        selected_sites, total_capacity, scaled_traces = site_selector.select(policy, start_time, end_time)
        print(selected_sites)
        
        for subgraph_policy in subgraph_policies:
            if (site_policy, subgraph_policy) not in policies:
                continue
            policy['subgraph'] = subgraph_policy
            log_msg(f"====== Subgraph selection policy {subgraph_policy} ======")

            scheduling_interval = SCHEDULING_HOURS * interp_factor * 1
            start_time = START_DATE * interp_factor
            end_time = start_time + scheduling_interval
                
            subgraphs = subgraph_selector.select(policy, start_time, end_time, selected_sites, scaled_traces=scaled_traces, max_k = MAX_K, batch_name=batch_name, use_cache=False)
            subgraphs = subgraphs[SUBGRAPH_OFFSET:SUBGRAPH_OFFSET+NUM_SUBGRAPHS]

            scaled_traces = scaled_traces.loc[start_time:end_time]
            if default_workloads[start_time] is None:
                aggregated_avg_power = sum([np.average(scaled_traces[site]) for site in selected_sites])
                workloads = factory.create_real_azure_workloads_with_energy(aggregated_avg_power, INTERVAL, CLOUD_UTIL, 0)
                print(workloads)
                default_workloads[start_time] = workloads

            nr_power = sum([i.maxpower[:48].sum() for i in workloads])
            print(f"central nr_power: {nr_power}")

            # for placement_policy in placement_policies:
            #     policy['placement'] = placement_policy
            #     log_msg(f"====== Placement policy {placement_policy} ======")
            #     for scheduling_policy in scheduling_policies:
            #         policy['scheduling'] = scheduling_policy
            #         log_msg(f"====== Scheduling policy {scheduling_policy} ======", header=True)
            #         scheduler.placement(policy, start_time, end_time, RUNNING_HOURS, subgraphs, default_workloads[start_time], scaled_traces=scaled_traces, cloud_util=CLOUD_UTIL, batch_name=batch_name, mip_options=MIP_OPTIONS, power_mis_ratio=0.00)
        
