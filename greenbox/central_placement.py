from macros import *
from factory import *

class CENTRAL_MIP():
    def __init__(self, power, ):
        pass

    def schedule(power, cur_timestamp, timestamps, interval, vms, init_states=None, SLO_AVAIL = 0.99, MIP_OPT_TIME = 10000, MIP_OPT_GOAL = 0.01):
        TIME, NUM_VM, t = timestamps, len(vms), cur_timestamp
        NUM_LOW_PRIO_VM = len([vm for vm in vms if vm.priority == 1])

        model = gp.Model()
        placement = model.addVars(TIME+1, NUM_VM, vtype=GRB.BINARY)
        booted_vms = model.addVars(TIME, NUM_VM, vtype=GRB.BINARY)
        complete = model.addVars(TIME+1, NUM_VM, vtype=GRB.BINARY)
        progress = model.addVars(TIME, NUM_VM, vtype=GRB.CONTINUOUS)
        exec_time = model.addVars(TIME, NUM_VM, vtype=GRB.CONTINUOUS)
        compl_time = model.addVars(NUM_VM, vtype=GRB.CONTINUOUS)
        has_completed = model.addVars(NUM_VM, vtype=GRB.BINARY)
        nr_power = model.addVars(TIME, vtype=GRB.CONTINUOUS)
        avail_per_vm = model.addVars(NUM_VM, vtype=GRB.CONTINUOUS, lb=0, ub=1)
        vm_overhead = model.addVars(TIME, vtype=GRB.CONTINUOUS)

        tmpVar2 = model.addVars(NUM_VM, vtype=GRB.CONTINUOUS)
        tmpVar3 = model.addVars(NUM_VM, vtype=GRB.CONTINUOUS)
        sum_avail_current_vms = model.addVar(vtype=GRB.CONTINUOUS)

        if init_states is None:
            model.addConstrs(placement[0, m] == 0  for m in range(NUM_VM))
            init_progress = np.zeros(NUM_VM)
            global_avail, num_complete_vm = 0, 0
        else:
            init_placement, init_progress = init_states["init_placement"], init_states["init_progress"]
            model.addConstrs(placement[0, m] == init_placement[m] for m in range(NUM_VM))        
            global_avail, num_complete_vm = init_states["global_avail"], init_states["num_complete_vm"]

        # objective 
        nr_total_power = gp.quicksum(nr_power[t] for t in range(TIME))
        model.setObjective(nr_total_power, GRB.MINIMIZE)

        ''' here we use the average survival rate of VMs as SLO '''
        model.addConstrs(tmpVar2[m] == compl_time[m] * avail_per_vm[m] for m in range(NUM_VM))
        model.addConstrs(tmpVar3[m] == (cur_timestamp + timestamps - vms[m].start_time / interval) * avail_per_vm[m] for m in range(NUM_VM))
        model.addConstrs((has_completed[m] == 1) >> (tmpVar2[m] == vms[m].lifetime / interval) for m in range(NUM_VM))
        model.addConstrs((has_completed[m] == 0) >> (tmpVar3[m] == exec_time[TIME-1, m]) for m in range(NUM_VM))
        model.addConstr(sum_avail_current_vms == gp.quicksum(avail_per_vm[m] for m in range(NUM_VM) if vms[m].priority == 1))
        model.addConstr(sum_avail_current_vms + global_avail * num_complete_vm >= SLO_AVAIL * (NUM_LOW_PRIO_VM + num_complete_vm))
        model.addConstrs(avail_per_vm[m] == 1 for m in range(NUM_VM) if vms[m].priority == 0)

        # power constraints: VM power consumption <= nr power
        model.addConstrs(vm_overhead[t] == gp.quicksum(booted_vms[t, m] * vms[m].maxpower[cur_timestamp+t] for m in range(NUM_VM)) for t in range(TIME))
        model.addConstrs(vm_overhead[t] <= nr_power[t] for t in range(TIME))

        # vm execution constraints
        model.addConstrs(gp.quicksum(booted_vms[t, m]) >= progress[t, m] for m in range(NUM_VM) for t in range(TIME))
        model.addConstrs(gp.quicksum(progress[t0, m] for t0 in range(t+1)) + init_progress[m] / interval == exec_time[t, m] for m in range(NUM_VM) for t in range(TIME))
        model.addConstrs(exec_time[t, m] <= vms[m].lifetime / interval for t in range(TIME) for m in range(NUM_VM))

        # complete constraints: bound complete[t, m] and compl_time[m]
        model.addConstrs((complete[t, m] == 1) >> (exec_time[t, m] >= vms[m].lifetime / interval) for t in range(TIME) for m in range(NUM_VM))

        model.addConstrs(gp.quicksum(complete[t, m] for t in range(TIME+1)) == 1 for m in range(NUM_VM))
        model.addConstrs(gp.quicksum(complete[t, m] * (t + 1) for t in range(TIME+1)) + cur_timestamp - vms[m].start_time / interval == compl_time[m] for m in range(NUM_VM))
        model.addConstrs(gp.quicksum(complete[t, m] for t in range(TIME)) == has_completed[m] for m in range(NUM_VM))

        model.Params.TIME_LIMIT = MIP_OPT_TIME
        model.Params.MIPGap = MIP_OPT_GOAL
        model.Params.MIPFocus = 2
        model.Params.NonConvex = 2
        model.optimize()

        placement_array = np.round(np.array([[placement[t, m].x for m in range(NUM_VM)] for t in range(TIME+1)])).astype(int)
        booted_vm_array = np.round(np.array([[booted_vms[t, m].x for m in range(NUM_VM)] for t in range(TIME)])).astype(int)
        nr_power_array = np.array([nr_power[t] for t in range(TIME)], dtype=float)
        
        return placement_array, booted_vm_array, nr_power_array



class CentralPlacement:

    def __init__(self, traces, n_sites, interval):
        self.traces = traces
        self.n_sites = n_sites
        self.interval = interval
        self.DEFAULT_MIP_OPTIONS = {"BATCH_MIP_OPT_TIME" : 120.0,
                    "BATCH_MIP_OPT_GOAL" : 0.002,
                    "PLACEMENT_MIP_OPT_TIME" : 120,
                    "PLACEMENT_MIP_OPT_GOAL" : 0.05}
    
    def placement(self, policy, start_time, end_time, running_hours, workloads, scaled_traces=None, cloud_util=1.0, batch_name="default", mip_options=None):
        site_policy = policy["site"]
        subgraph_policy = policy["subgraph"]
        scheduling_policy = policy["scheduling"]

        full_policy_name = f"{site_policy} + {subgraph_policy} + {scheduling_policy}"

        max_timestamps = end_time - start_time
        # overlapped mip options
        if mip_options is None:
            mip_options = self.DEFAULT_MIP_OPTIONS
            
        if scaled_traces is not None:
            current_renewable_traces = scaled_traces
        else:
            current_renewable_traces = self.traces 
                  
        num_vms = len(workloads)
        log_msg("Num of vms scheduled", num_vms)

        traces = current_renewable_traces.to_numpy().transpose()
        scheduler = CENTRAL_SCHEDULER(opt_time=mip_options["BATCH_MIP_OPT_TIME"], opt_goal=mip_options["BATCH_MIP_OPT_GOAL"], slo_avail=mip_options["SLO_AVAIL"],interval=self.interval)

        vm_placement = defaultdict(lambda : defaultdict(list))
        max_running_steps = int(running_hours * HOURS // self.interval)
        for t in range(max_running_steps):
            remaining_energy = scheduler.remaining_energy(cloud_util)

            # decide arriving vms for the NEXT TIMESTAMP based on the next remaining energy
            arrived_vms = []
            all_schedulable_vms = workloads # [vm for vm in workloads if vm.start < t + 1]
            for vm in all_schedulable_vms:
                if vm.avg_power < positive_remaining_energy and (max_running_steps-t-1) >= vm.lifetime / self.interval:
                    arrived_vms.append(vm)
                    vm.start_time = (t + 1) * self.interval
                    positive_remaining_energy -= vm.avg_power
                else:
                    continue
            workloads = [vm for vm in workloads if vm not in arrived_vms]
            num_remaining_vms = len(workloads)
                
            vm_power = [vm.avg_power for vm in arrived_vms]
            log_msg(f"Time:{t}, Remaining: {num_remaining_vms}, Added: {len(arrived_vms)}, Energy before added VMs: {sum(remaining_energy):.2f}", replace=False)
             
            vm_placement[i][t] = []
                      
            remaining_vms = []
            for i, vm in enumerate(arrived_vms):
                if sum(vm_placement_matrix[i]) == 0:
                    remaining_vms.append(vm)
                    continue
                graph_i = vm_placement_matrix[i].tolist().index(1)
                vm_placement[graph_i][t].append(vm)
                remaining_energy[graph_i] -= vm.avg_power
            # print(len(arrived_vms), len(remaining_vms))

            subgraph_sorted_by_energy = sorted(enumerate(subgraphs), key=lambda x: remaining_energy[x[0]], reverse=True)
            for i, vm in enumerate(remaining_vms):
                selected_subgraph = subgraph_sorted_by_energy[i % len(subgraphs)][0]
                vm_placement[selected_subgraph][t].append(vm)


            schedulers.migrate(new_vms = vm_placement[i][t])

        simulation_results = []
        cdf_t = []
        utilizations = []
        max_cores = []
        vm_profiles_all_subgraphs = []
        power_up_ratios = []
        completed_vms = []
        solutions = dict()
        nr_energy = 0
        for i in range(len(subgraphs)):
            if use_ray:
                scheduler_finish_future = schedulers[i].finish.remote()
                max_t, avg_t, dist_t, completed_vm, vm_profiles, global_profile, utilization, running_vms = ray.get(scheduler_finish_future)
            else:
                max_t, avg_t, dist_t, completed_vm, vm_profiles, global_profile, utilization, running_vms = schedulers[i].finish()
            # log_msg(i, len(dist_t), avg_t, completed_vm, subgraphs[i][3])

            subgraph_sites = subgraphs[i][0]
            per_site_trace = current_renewable_traces[subgraph_sites]
            max_cores.append(np.max(np.sum(per_site_trace, axis=1)) * max_timestamps)
            utilizations.append(sum(utilization[:int(max_timestamps)]))
            completed_vms.extend(completed_vm)
            solutions[i] = running_vms
            nr_energy += sum(global_profile.nr_energy_used)

            power_up_ratios = []

            if np.isnan(avg_t):
                avg_t = max_timestamps

            if avg_t != 0:
                simulation_results.append(avg_t)
                cdf_t.extend(dist_t)  
                vm_profiles_all_subgraphs.extend(vm_profiles) 
                
            batch_dir = f'raw/{batch_name}/day_{start_time//self.interval//24}'
            if not os.path.exists(batch_dir):
                os.makedirs(batch_dir, exist_ok=True)
            
            with open(f"{batch_dir}/batch_info", "w") as f:
                batch_info = dict()
                batch_info['num_vms'] = num_vms
                batch_info['num_graphs'] = len(subgraphs)
                batch_info['max_timestamps'] = max_timestamps
                json.dump(batch_info, f)

            subgraph_dir = f'{batch_dir}/subgraphs/graph_{i}'
            if not os.path.exists(subgraph_dir):
                os.makedirs(subgraph_dir, exist_ok=True)
                
            policy_dir = f'{subgraph_dir}/{full_policy_name}'
            if not os.path.exists(policy_dir):
                os.mkdir(policy_dir)
            
            with open(f'{policy_dir}/global_profile', 'wb') as f:
                pickle.dump(global_profile, f)
            
            with open(f'{policy_dir}/vm_profiles', 'wb') as f:  
                pickle.dump(vm_profiles, f)
                
            with open(f'{policy_dir}/running_vms', 'wb') as f:
                pickle.dump(running_vms, f)
                
            with open(f'{policy_dir}/power_traces', 'wb') as f:
                pickle.dump(current_renewable_traces[subgraphs[i][0]], f)

        # log_msg(f"Average completion time across graphs: {sum(simulation_results) / len(simulation_results):.2f} hours.")
        # vm_profiles_all_subgraphs = [profile for profile in vm_profiles_all_subgraphs if profile.finished]
        # log_msg(f"Avg JCT: {np.average([profile.end - profile.start for profile in vm_profiles_all_subgraphs]):.2f}. 95% Slowdown: {np.percentile([(profile.end - profile.start) / profile.exec for profile in vm_profiles_all_subgraphs], 95):.2f}. Migration: {np.average([profile.overhead for profile in vm_profiles_all_subgraphs]):.2f}. Queuing: {np.average([profile.queuing for profile in vm_profiles_all_subgraphs]):.2f}. Blackout: {np.average([profile.interrupt for profile in vm_profiles_all_subgraphs]):.2f}.", emphasize=True)
        log_msg(f"Avg Avail: {np.average([profile.avail() for profile in vm_profiles_all_subgraphs]):.2f}. Migration: {np.average([profile.overhead for profile in vm_profiles_all_subgraphs]):.2f}. Queuing: {np.average([profile.queuing for profile in vm_profiles_all_subgraphs]):.2f}. Blackout: {np.average([profile.interrupt for profile in vm_profiles_all_subgraphs]):.2f}.", emphasize=True)
        
        affected_vm_profiles = [profile for profile in vm_profiles_all_subgraphs if profile.overhead + profile.queuing + profile.interrupt > 0 ]
        log_msg(f"Avg JCT of affected workloads: {np.average([profile.avail() for profile in affected_vm_profiles]):.2f}. Migration: {np.average([profile.overhead for profile in affected_vm_profiles]):.2f}. Queuing: {np.average([profile.queuing for profile in affected_vm_profiles]):.2f}. Blackout: {np.average([profile.interrupt for profile in affected_vm_profiles]):.2f}. # of affected workloads: {len(affected_vm_profiles)}", emphasize=True)
        
        log_msg(f"Total completed VMs {len(completed_vms)}")
        log_msg(f"Total NR Energy {nr_energy:.3f}")
        
        for i in range(len(subgraphs)):
            # print(utilizations[i], max_cores[i])
            power_up_ratios.append(utilizations[i]/max_cores[i])
        log_msg(f"VB Utilization: {np.average(power_up_ratios) :.2f}") # , np.std(power_up_ratios))

        return
