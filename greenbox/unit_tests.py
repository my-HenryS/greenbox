from macros import *
# from factory import create_workloads, App

from site_selection import SiteSelection, SubGraphSelection
from app_placement import AppPlacement
from parsing import utility, simulator_validation
from simulator import *

# def test_case_one():
#     ray.init(local_mode=True)
#     n_sites = 2
#     policy = {"site":"custom", "subgraph":"custom", "placement":"greedy", "n_sites":2}
#     subgraphs = [[["A", "B"]]]
#     traces = pd.DataFrame()
#     traces['A'] = [100, 80, 70, 55, 55, 70, 80, 100]
#     traces['B'] = [10, 30, 40, 55, 55, 40, 30, 10]
#     # traces['A'] = [100, 100, 100, 90, 70, 50, 40, 40, 40]
#     # traces['B'] = [40, 40, 40, 50, 70, 90, 100, 100, 100]
#     # traces['A'] = [20, 80, 120, 80, 20, 80, 120, 80, 20]
#     # traces['B'] = [60]*9
#     app_1 = App(name="cofii-e4s", num_vms=4, vm_type=0, cores_per_vm=4, completion_time=5*60, migration=2, recomputation=0, latency = 1)
#     app_5 = App(name="dnn_large", num_vms=1, vm_type=1, cores_per_vm=12, completion_time=5*60, migration=7, recomputation=60, latency = 1)
#     apps, apps_per_step = create_workloads([14,8], 4, "even", template=[app_1, app_5])
#     print(len(apps))
    
#     AppPlacement.max_cores = 320
#     AppPlacement.max_power = 160
#     app_scheduler = AppPlacement(traces, n_sites, app_distribution=None)
#     for scheduling_policy in ["greedy", "mip"]:
#         policy["scheduling"] = scheduling_policy
#         avg_t = app_scheduler.place_app(policy, start_time=0, end_time=len(traces), subgraphs=subgraphs, scaled_factor=1.0, low_cutoff=0.0, apps=apps, ray=False)


# def test_case_alloc():
#     ray.init(local_mode=True)
#     n_sites = 2
#     policy = {"site":"custom", "subgraph":"custom", "placement":"greedy", "n_sites":2}
#     subgraphs = [[["A", "B"]]]
#     traces = pd.DataFrame()
#     # traces['A'] = [100, 80, 70, 55, 55, 70, 80, 100, 100]
#     # traces['B'] = [10, 30, 40, 55, 55, 40, 30, 10, 10]
#     traces['A'] = [100, 100, 100, 90, 70, 50, 40, 40, 40, 40, 40]
#     traces['B'] = [40, 40, 40, 50, 70, 90, 100, 100, 100, 100, 100]
#     # traces['A'] = [20, 80, 120, 80, 20, 80, 120, 80, 20]
#     # traces['B'] = [60]*9
#     app_1 = App(name="cofii-e4s", num_vms=3, vm_type=0, cores_per_vm=4, completion_time=3*60, migration=2, recomputation=10, latency = 1)
#     app_5 = App(name="dnn_large", num_vms=1, vm_type=1, cores_per_vm=12, completion_time=4*60, migration=7, recomputation=60, latency = 1)
#     apps, apps_per_step = create_workloads([30,15], 4, "backlog", template=[app_1, app_5])
#     print(len(apps))
    
#     AppPlacement.max_cores = 320
#     AppPlacement.max_power = 160
#     app_scheduler = AppPlacement(traces, n_sites, app_distribution=None, interp_factor=1)
#     for scheduling_policy in ["greedy", "mip"]:
#         policy["scheduling"] = scheduling_policy
#         avg_t = app_scheduler.place_app(policy, start_time=0, end_time=len(traces)-1, subgraphs=subgraphs, scaled_factor=1.0, low_cutoff=0.0, apps=apps, batch_name="alloc")


# AppPlacement.max_cores = 320
# AppPlacement.max_power = 320
    
# def test_case_one():
#     n_sites = 2
#     policy = {"site":"custom", "subgraph":"custom", "placement":"greedy", "n_sites":2}
#     subgraphs = [[["A", "B"]]]
#     traces = pd.DataFrame()

#     traces['A'] = [100, 100, 50, 50, 50, 50]
#     traces['B'] = [0, 0, 50, 50, 50, 50]

#     app_1 = App(name="cofii-e4s", num_vms=3, vm_type=0, cores_per_vm=4, completion_time=4*60, migration=2, recomputation=10, latency = 1)
#     app_4 = App(name="db_large", num_vms=3, vm_type=0, cores_per_vm=4, completion_time=4*60, migration=20, recomputation=0, latency = 0.7)
#     apps, apps_per_step = create_workloads([4,4], 4, "backlog", template=[app_1, app_4])
    
#     app_scheduler = AppPlacement(traces, n_sites, app_distribution=None, interp_factor=1)
#     for scheduling_policy in ["greedy", 'greedy-fair', "mip"]:
#         policy["scheduling"] = scheduling_policy
#         avg_t = app_scheduler.place_app(policy, start_time=0, end_time=len(traces), subgraphs=subgraphs, scaled_factor=1.0, low_cutoff=0.0, apps=apps, batch_name="test one")
    
# def test_case_two():
#     n_sites = 2
#     policy = {"site":"custom", "subgraph":"custom", "placement":"greedy", "n_sites":2}
#     subgraphs = [[["A", "B"]]]
#     traces = pd.DataFrame()

#     traces['A'] = [100, 100, 50, 50, 50, 50]
#     traces['B'] = [0, 0, 50, 50, 50, 50]

#     app_1 = App(name="cofii-e4s", num_vms=3, vm_type=0, cores_per_vm=4, completion_time=4*60, migration=2, recomputation=10, latency = 1)
#     # app_4 = App(name="db_large", num_vms=3, vm_type=0, cores_per_vm=4, completion_time=4*60, migration=20, recomputation=0, latency = 0.7)
#     app_5 = App(name="dnn_large", num_vms=1, vm_type=1, cores_per_vm=12, completion_time=5*60, migration=7, recomputation=60, latency = 1)
#     apps, apps_per_step = create_workloads([4,4], 4, "backlog", template=[app_1, app_5])
    
#     app_scheduler = AppPlacement(traces, n_sites, app_distribution=None, interp_factor=1)
#     for scheduling_policy in ["greedy", 'greedy-fair', "mip"]:
#         policy["scheduling"] = scheduling_policy
#         avg_t = app_scheduler.place_app(policy, start_time=0, end_time=len(traces), subgraphs=subgraphs, scaled_factor=1.0, low_cutoff=0.0, apps=apps, batch_name="test two")

# def test_case_three():
#     n_sites = 2
#     policy = {"site":"custom", "subgraph":"custom", "placement":"greedy", "n_sites":2}
#     subgraphs = [[["A", "B"]]]
#     traces = pd.DataFrame()

#     traces['A'] = [50, 50, 100, 100, 100, 100]
#     traces['B'] = [50, 50, 0, 0, 0, 0]

#     app_1 = App(name="cofii-e4s", num_vms=3, vm_type=0, cores_per_vm=4, completion_time=4*60, migration=2, recomputation=10, latency = 1)
#     # app_4 = App(name="db_large", num_vms=3, vm_type=0, cores_per_vm=4, completion_time=4*60, migration=20, recomputation=0, latency = 0.7)
#     app_5 = App(name="dnn_large", num_vms=1, vm_type=1, cores_per_vm=12, completion_time=5*60, migration=7, recomputation=60, latency = 1)
#     apps, apps_per_step = create_workloads([4,4], 4, "backlog", template=[app_1, app_5])
    
#     app_scheduler = AppPlacement(traces, n_sites, app_distribution=None, interp_factor=1)
#     for scheduling_policy in ["greedy", 'greedy-fair', "mip"]:
#         policy["scheduling"] = scheduling_policy
#         avg_t = app_scheduler.place_app(policy, start_time=0, end_time=len(traces), subgraphs=subgraphs, scaled_factor=1.0, low_cutoff=0.0, apps=apps, batch_name="test three")
        
# def test_case_four():
#     n_sites = 2
#     policy = {"site":"custom", "subgraph":"custom", "placement":"greedy", "n_sites":2}
#     subgraphs = [[["A", "B"]]]
#     traces = pd.DataFrame()

#     traces['A'] = [50, 50, 50, 100, 100, 100]
#     traces['B'] = [50, 50, 50, 0, 0, 0]

#     # app_1 = App(name="cofii-e4s", num_vms=3, vm_type=0, cores_per_vm=4, completion_time=4*60, migration=2, recomputation=10, latency = 1)
#     # app_4 = App(name="db_large", num_vms=3, vm_type=0, cores_per_vm=4, completion_time=4*60, migration=20, recomputation=0, latency = 0.7)
#     app_5_0 = App(name="dnn_large-3hrs", num_vms=1, vm_type=1, cores_per_vm=12, completion_time=3*60, migration=7, recomputation=60, latency = 1)
#     app_5_1 = App(name="dnn_large-4hrs", num_vms=1, vm_type=1, cores_per_vm=12, completion_time=4*60, migration=7, recomputation=60, latency = 1)
#     apps, apps_per_step = create_workloads([4,4], 4, "backlog", template=[app_5_0, app_5_1])
    
    
#     app_scheduler = AppPlacement(traces, n_sites, app_distribution=None, interp_factor=1)
#     for scheduling_policy in ["greedy", 'greedy-fair', "mip"]:
#         policy["scheduling"] = scheduling_policy
#         avg_t = app_scheduler.place_app(policy, start_time=0, end_time=len(traces), subgraphs=subgraphs, scaled_factor=1.0, low_cutoff=0.0, apps=apps, batch_name="test four")
        
  
# def test_case_five():
#     n_sites = 2
#     policy = {"site":"custom", "subgraph":"custom", "placement":"greedy", "n_sites":2}
#     subgraphs = [[["A", "B"]]]
#     traces = pd.DataFrame()

#     traces['A'] = [50, 50, 50, 100, 100, 100]
#     traces['B'] = [50, 50, 50, 0, 0, 0]

#     # app_1 = App(name="cofii-e4s", num_vms=3, vm_type=0, cores_per_vm=4, completion_time=4*60, migration=2, recomputation=10, latency = 1)
#     # app_4 = App(name="db_large", num_vms=3, vm_type=0, cores_per_vm=4, completion_time=4*60, migration=20, recomputation=0, latency = 0.7)
#     app_5_0 = App(name="dnn_large-30min-epoch", num_vms=1, vm_type=1, cores_per_vm=12, completion_time=4*60, migration=7, recomputation=30, latency = 1)
#     app_5_1 = App(name="dnn_large-40min-epoch", num_vms=1, vm_type=1, cores_per_vm=12, completion_time=4*60, migration=7, recomputation=40, latency = 1)
#     apps, apps_per_step = create_workloads([4,4], 4, "backlog", template=[app_5_0, app_5_1])
    
    
#     app_scheduler = AppPlacement(traces, n_sites, app_distribution=None, interp_factor=1)
#     for scheduling_policy in ["greedy", 'greedy-fair', "mip"]:
#         policy["scheduling"] = scheduling_policy
#         avg_t = app_scheduler.place_app(policy, start_time=0, end_time=len(traces), subgraphs=subgraphs, scaled_factor=1.0, low_cutoff=0.0, apps=apps, batch_name="test five")
        

# def test_case_six():
#     n_sites = 2
#     policy = {"site":"custom", "subgraph":"custom", "placement":"greedy", "n_sites":2}
#     subgraphs = [[["A", "B"]]]
#     traces = pd.DataFrame()

#     traces['A'] = [50,50,50,0,0,0]
#     traces['B'] = [0,0,50,50,50,50]

#     app_4 = App(name="db_large", num_vms=3, vm_type=0, cores_per_vm=4, completion_time=4*60, migration=20, recomputation=0, latency = 0.7)
#     apps, apps_per_step = create_workloads([4], 4, "backlog", template=[app_4])
    
    
#     app_scheduler = AppPlacement(traces, n_sites, app_distribution=None, interp_factor=1)
#     for scheduling_policy in ["greedy", 'greedy-fair', "mip"]:
#         policy["scheduling"] = scheduling_policy
#         avg_t = app_scheduler.place_app(policy, start_time=0, end_time=len(traces), subgraphs=subgraphs, scaled_factor=1.0, low_cutoff=0.0, apps=apps, batch_name="test_six")
      
      

# def test_case_blackout():
#     n_sites = 2
#     policy = {"site":"custom", "subgraph":"custom", "placement":"greedy", "n_sites":2}
#     subgraphs = [[["A", "B"]]]
#     traces = pd.DataFrame()

#     traces['A'] = [50,50,50,0,0,0]
#     traces['B'] = [0,0,50,50,50,50]

#     app_4 = App(name="db_large", num_vms=3, vm_type=0, cores_per_vm=4, completion_time=4*60, migration=20, recomputation=0, latency = 0.7)
#     apps, apps_per_step = create_workloads([4], 4, "backlog", template=[app_4])
    
    
#     app_scheduler = AppPlacement(traces, n_sites, app_distribution=None, interp_factor=1)
#     for scheduling_policy in ["mip"]:
#         policy["scheduling"] = scheduling_policy
#         avg_t = app_scheduler.place_app(policy, start_time=0, end_time=len(traces), subgraphs=subgraphs, scaled_factor=1.0, low_cutoff=0.0, apps=apps, batch_name="test_blackout")

from vm_mip_model import VM, NEW_MIP
VM_A = VM("a", cores=5, memory=128, lifetime=7*HOURS, start_time=1*HOURS)
VM_B = VM("b", cores=5, memory=128, lifetime=4*HOURS, start_time=1*HOURS)
VM_C = VM("c", cores=5, memory=16, lifetime=4*HOURS, start_time=1*HOURS)
VM_D = VM("d", cores=1, memory=128, lifetime=7*HOURS, start_time=1*HOURS)

def test_case_one():
    '''
    forsee the predicted power is not enough, pre-migrate out 
    '''
    traces = pd.DataFrame()
    traces['A'] = [0, 50, 50, 50, 10, 10, 10, 10, 10]
    traces['B'] = [0, 0, 0, 50, 60, 50, 50, 50, 50]
    traces['C'] = [0, 0, 0, 10, 10, 10, 10, 10, 10]
    traces = np.transpose(traces.to_numpy())
    
    vms = [copy.deepcopy(VM_A) for _ in range(2)]
    model = NEW_MIP()
    model.migrate(0, 7, 60, traces, traces, vms, 3, 0.9999, 10000, 0.01, None, None, None, None, "mip", 0)

def test_case_two():
    '''
    long lived VM prefer stable site
    short lived VM prefer unstable site
    '''
    traces = pd.DataFrame()
    traces['A'] = [0, 50, 50, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    traces['B'] = [0, 0, 26, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    traces['C'] = [0, 0, 26, 20, 20, 12, 20, 10, 20, 10, 20, 10]
    traces = np.transpose(traces.to_numpy())
    
    vms = [VM_A, VM_B]
    model = NEW_MIP()
    model.migrate(0, 10, 60, traces, traces, vms, 3, 0.9999, 10000, 0.01, None, None, None, None, "mip", 0)

def test_case_three():
    '''
    prioritize migrate-sensitve VMs in stable sites
    '''
    traces = pd.DataFrame()
    traces['A'] = [0, 50, 50, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    traces['B'] = [0, 0, 26, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    traces['C'] = [0, 0, 26, 20, 20, 12, 20, 10, 20, 10, 20, 10]
    traces = np.transpose(traces.to_numpy())
    
    vms = [VM_B, VM_C]
    model = NEW_MIP()
    model.migrate(0, 10, 60, traces, traces, vms, 3, 0.9999, 10000, 0.01, None, None, "mip", 0)

###
# priority order: 
# 1. live migration (if total power is enough)
# 2. live migration light VM (total power is not enough but use brown energy to support)
# 3. brown-power heavy VM (heavy VM migration overhead is too much)
###

def test_case_four():
    '''
    1st priority
    prioritize migrate-insensitive VM (VM_C with low memory size)
    '''
    traces = pd.DataFrame()
    traces['A'] = [0, 50, 50, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    traces['B'] = [0, 0, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    traces['C'] = [0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    traces = np.transpose(traces.to_numpy())
    
    vms = [VM_A, VM_C]
    model = NEW_MIP()
    model.migrate(0, 10, 60, traces, traces, vms, 3, 0.9999, 10000, 0.01, None, None, None, None, "mip", 0)

def test_case_five():
    '''
    2nd priority
    prioritize using brown energy to power the light VM migration
    '''
    traces = pd.DataFrame()
    traces['A'] = [0, 50, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40]
    traces['B'] = [0, 0, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    traces['C'] = [0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    traces = np.transpose(traces.to_numpy())
    
    vms = [VM_A, VM_C]
    model = NEW_MIP()
    model.migrate(0, 10, 60, traces, traces, vms, 3, 0.9999, 10000, 0.01, None, "mip", 0)

def test_case_six():
    '''
    3nd priority
    prioritize using brown energy to power the heavy VM
    '''
    traces = pd.DataFrame()
    traces['A'] = [0, 50, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40]
    traces['B'] = [0, 0, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    traces['C'] = [0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    traces = np.transpose(traces.to_numpy())
    
    vms = [copy.deepcopy(VM_A) for _ in range(2)]
    model = NEW_MIP()
    model.migrate(0, 10, 60, traces, traces, vms, 3, 0.9999, 10000, 0.01, None, None, None, None, "mip", 0)

if __name__ == "__main__":

    # ray.init(local_mode=True)
    # interp_factor = 1
    # gp.Model("test")

    # # app_distribution = read_arrival_trace()
    # vb_coords = read_vb_coords()
    # capacity = read_vb_capacity()
    # renewable_traces, vb_sites = read_traces(vb_coords, capacity, interp_factor)

    # test_case_one()
    # test_case_two()
    # test_case_three()
    # test_case_four()
    test_case_five()
    # test_case_six()
    # test_case_blackout()
    # test_case_latency()

