from macros import *

from vm_mip_model import GreenBoxSim
from factory import *

class VMPlacement:

    def __init__(self, traces, n_sites, interval):
        self.traces = traces
        self.n_sites = n_sites
        self.interval = interval
        self.DEFAULT_MIP_OPTIONS = {"BATCH_MIP_OPT_TIME" : 120.0,
                    "BATCH_MIP_OPT_GOAL" : 0.05,
                    "PLACEMENT_MIP_OPT_TIME" : 120,
                    "PLACEMENT_MIP_OPT_GOAL" : 0.05}
    
    # def placement(self, policy, start_time, end_time, running_hours, subgraphs, workloads, scaled_traces=None, cloud_util=1.0, use_ray=False, default_vm_arrival=None, batch_name="default", mip_options=None, power_mis_ratio=0., name=""):
    def placement(self, policy, start_time, end_time, running_hours, subgraphs, workloads, scaled_traces=None, cloud_util=1.0, use_ray=False, default_vm_arrival=None, batch_name="default", mip_options=None, power_mis_ratio=0., name="", lahead=5):
        site_policy = policy["site"]
        subgraph_policy = policy["subgraph"]
        n_sites = policy["n_sites"]
        placement_policy = policy["placement"]
        scheduling_policy = policy["scheduling"]

        full_policy_name = f"{site_policy} + {subgraph_policy} + {placement_policy} + {scheduling_policy}"

        start_date = DATE_OFFSET + dt.timedelta(start_time / 24 / (HOURS / self.interval))
        start_date = start_date.date()
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
        if default_vm_arrival is not None:
            log_msg("Num of vms scheduled from default", sum([len(_) for _ in default_vm_arrival]))

        schedulers = []
        for i, subgraph in enumerate(subgraphs):
            if use_ray:
                sim_create_func = GreenBoxSim.remote
            else:
                sim_create_func = GreenBoxSim
            np.set_printoptions(threshold=sys.maxsize)
            # print(subgraph[0], current_renewable_traces[subgraph[0]].to_numpy().transpose())
            # print(np.sum(current_renewable_traces[subgraph[0]].to_numpy(), axis=1))

            # misprediction of power trace
            np.random.seed(1)
            random_scale = np.random.uniform(1, 1, len(current_renewable_traces.index))
            if power_mis_ratio < 0:
                random_scale = np.random.uniform(1+power_mis_ratio, 1, len(current_renewable_traces.index))
            elif power_mis_ratio > 0:
                random_scale = np.random.uniform(1, 1+power_mis_ratio, len(current_renewable_traces.index))
            # random_scale = np.random.uniform(1-power_mis_ratio, 1+power_mis_ratio, len(current_renewable_traces.index))
            traces = current_renewable_traces[subgraph[0]].to_numpy().transpose()
            predicted_traces = np.array([random_scale * trace for trace in traces])
            log_msg(f"predicted trace:{predicted_traces}, trace:{traces}")
            schedulers.append(
                sim_create_func(scheduling_policy, 
                    traces,
                    predicted_traces,
                    [],
                    len(current_renewable_traces[subgraph[0]].columns),
                    opt_time=mip_options["BATCH_MIP_OPT_TIME"], 
                    opt_goal=mip_options["BATCH_MIP_OPT_GOAL"], 
                    slo_avail=mip_options["SLO_AVAIL"], 
                    max_timestamps=max_timestamps,
                    interval = self.interval,
                    subgraph_id=i,
                    lahead=lahead) 
                )

        vm_placement = defaultdict(lambda : defaultdict(list))
        max_running_steps = int(running_hours * HOURS // self.interval)
        for t in range(max_running_steps):
            # if t == 2:
            #     exit()
            if use_ray:
                remaining_energy = [schedulers[i].remaining_energy.remote(cloud_util) for i, subgraph in enumerate(subgraphs)]
                remaining_energy = ray.get([actor for actor in remaining_energy])
            else:
                remaining_energy = [schedulers[i].remaining_energy(cloud_util) for i, subgraph in enumerate(subgraphs)]
                
            # remaining_energy = [np.sum([_ for _ in e if _ > 0])  for e in remaining_energy]
            remaining_energy = [np.sum([_ for _ in e])  for e in remaining_energy]
            log_msg(f"timestamp:{t}, remaining_energy:{remaining_energy}")
            # remaining_energy = [np.sum(_ for _ in e if _ > 0) for e in remaining_energy]

            # decide arriving vms for the NEXT TIMESTAMP based on the next remaining energy
            if default_vm_arrival is not None:
                arrived_vms = default_vm_arrival[t]
                num_remaining_vms = sum([len(_) for _ in default_vm_arrival[t+1:]])
            else:
                total_remaining_energy = sum(remaining_energy)
                # positive_remaining_energy = sum([_ for _ in remaining_energy])
                positive_remaining_energy = sum([_ for _ in remaining_energy if _ > 0])
                arrived_vms = []
                all_schedulable_vms = workloads # [vm for vm in workloads if vm.start < t + 1]
                for vm in all_schedulable_vms:
                    # if vm.avg_power < positive_remaining_energy and (max_running_steps-t-1) >= vm.lifetime / self.interval:
                    # print(vm.avg_power, positive_remaining_energy)
                    if vm.avg_power < positive_remaining_energy:
                        arrived_vms.append(vm)
                        vm.start_time = (t + 1) * self.interval
                        positive_remaining_energy -= vm.avg_power
                    else:
                        # log_msg(f"not scheudled:{vm}")
                        continue
                workloads = [vm for vm in workloads if vm not in arrived_vms]
                num_remaining_vms = len(workloads)
            
            vm_power = [vm.avg_power for vm in arrived_vms]
            log_msg(f"Time:{t}, arrived_vms:{arrived_vms}")
            log_msg(f"Time:{t}, Remaining: {num_remaining_vms}, Added: {len(arrived_vms)}, Energy before added VMs: {sum(remaining_energy):.2f}", replace=False)
             
            for i in range(len(subgraphs)):
                vm_placement[i][t] = []
                
            def greedy_placement():
                remaining_vms = [_ for _ in arrived_vms]
                vm_placement_matrix = np.zeros((len(arrived_vms), len(subgraphs)))
                positive_remaining_energy = sum([_ for _ in remaining_energy if _ > 0])
                if positive_remaining_energy > 0:
                    core_per_subgraph = sum(vm_power) / positive_remaining_energy
    
                    for j, subgraph in enumerate(subgraphs):
                        if remaining_energy[j]  <= 0:
                            continue
                        my_cores = remaining_energy[j] * core_per_subgraph
                        selected_vms = []
                        for vm in remaining_vms:
                            if vm.avg_power <= my_cores:
                                selected_vms.append(vm)
                                vm_placement_matrix[arrived_vms.index(vm), j] = 1
                                my_cores -= vm.avg_power
                            if my_cores <= 0:
                                break

                        remaining_vms = [vm for vm in remaining_vms if vm not in selected_vms]
                return vm_placement_matrix, remaining_vms
                        
            # schedule based on energy
            if placement_policy == "greedy":
                vm_placement_matrix, remaining_vms = greedy_placement()

            if placement_policy == "even":
                vms_per_site = int(math.ceil(len(arrived_vms) / len(subgraphs)))
                for i, g in enumerate(subgraphs):
                    vm_placement[i][t] = arrived_vms[i*vms_per_site:(i+1)*vms_per_site]

            if placement_policy == "mip":
                if use_ray:
                    predicted_remaining_energy = [schedulers[i].predicted_remaining_energy.remote(cloud_util) for i, subgraph in enumerate(subgraphs)]
                    predicted_remaining_energy = ray.get([actor for actor in predicted_remaining_energy])
                else:
                    predicted_remaining_energy = [schedulers[i].predicted_remaining_energy(cloud_util) for i, subgraph in enumerate(subgraphs)]
                
                predicted_remaining_energy = np.array(predicted_remaining_energy)
                
                placement = PLACEMENT_MIP(predicted_remaining_energy, len(subgraphs), step=max_timestamps, timestep=self.interval)

                vm_placement_hint, _ = greedy_placement()
                est_time, vm_placement_matrix = placement.place(arrived_vms, t, init_placement=None, hint=vm_placement_hint)

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
            
            # # store the VM placement here
            # for tmp_i in vm_placement.keys():
            #     for i, curr_vm in enumerate(vm_placement[tmp_i][t]):
            #         with open(f"vm_trace/vm_{start_time}_{cloud_util}_{tmp_i}_{i}_{t}", "wb") as trace:
            #             pickle.dump(curr_vm, trace)
            # # restore the VM placement 
            # tmp_placement = defaultdict(list)
            # for tmp_i in range(len(subgraphs)):
            #     count = 0
            #     while True:
            #         file_str = f"vm_trace/vm_{start_time}_{cloud_util}_{tmp_i}_{count}_{t}"
            #         if os.path.isfile(file_str):
            #             with open(file_str, "rb") as trace:
            #                 curr_vm = pickle.load(trace)
            #                 tmp_placement[tmp_i].append(curr_vm)
            #             count += 1
            #         else:
            #             break

            if use_ray:
                per_process_results = [] 
                for i in range(len(subgraphs)):
                    # log_msg(f"{i} {sum([vm.avg_power for vm in vm_placement[i][t]])} {remaining_energy[i]}")
                    per_process_results.append(schedulers[i].migrate.remote(new_vms = vm_placement[i][t], get_some_hints=True))
                    # per_process_results.append(schedulers[i].migrate.remote(new_vms = tmp_placement[i], get_some_hints=True))
            
                for i in range(len(subgraphs)):
                    ray.get(per_process_results[i])
            else:
                for i in range(len(subgraphs)):
                    schedulers[i].migrate(new_vms = vm_placement[i][t], get_some_hints=True)
                    # schedulers[i].migrate(new_vms = tmp_placement[i], get_some_hints=True)

        simulation_results = []
        cdf_t = []
        utilizations = []
        max_cores = []
        vm_profiles_all_subgraphs = []
        power_up_ratios = []
        completed_vms = []
        solutions = dict()
        nr_energy = []
        r_energy = []
        low_priority_nr_energy = []
        low_priority_r_energy = []
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
            nr_energy.append(sum(global_profile.nr_energy_used))
            r_energy.append(sum(global_profile.r_energy_used))
            low_priority_nr_energy.append(sum(global_profile.low_priority_nr_energy_used))
            low_priority_r_energy.append(sum(global_profile.low_priority_r_energy_used))

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
        log_msg(f"Avg Low Priority Avail: {np.average([profile.avail() for profile in vm_profiles_all_subgraphs if profile.vm.priority == 1]):.2f}. Migration: {np.average([profile.overhead for profile in vm_profiles_all_subgraphs]):.2f}. Queuing: {np.average([profile.queuing for profile in vm_profiles_all_subgraphs]):.2f}. Blackout: {np.average([profile.interrupt for profile in vm_profiles_all_subgraphs]):.2f}.", emphasize=True)
        
        affected_vm_profiles = [profile for profile in vm_profiles_all_subgraphs if profile.overhead + profile.queuing + profile.interrupt > 0]
        if len(affected_vm_profiles) > 0:
            log_msg(f"Avg JCT of affected workloads: {np.average([profile.avail() for profile in affected_vm_profiles]):.2f}. Migration: {np.average([profile.overhead for profile in affected_vm_profiles]):.2f}. Queuing: {np.average([profile.queuing for profile in affected_vm_profiles]):.2f}. Blackout: {np.average([profile.interrupt for profile in affected_vm_profiles]):.2f}. # of affected workloads: {len(affected_vm_profiles)}", emphasize=True)
        
        log_msg(f"Total completed VMs {len(completed_vms)}")
        log_msg(f"Total NR Energy {nr_energy}, Total Total:{sum(nr_energy)}")
        log_msg(f"Total R Energy {r_energy}, Total Total:{sum(r_energy)}")
        log_msg(f"Total Total Carbon [nr carbon, r carbon, total carbon]: {cal_operational_carbon(sum(nr_energy), sum(r_energy))}")
        log_msg(f"Total Low Priority NR Energy {low_priority_nr_energy}, Total Total:{sum(low_priority_nr_energy)}")
        log_msg(f"Total Low Priority R Energy {low_priority_nr_energy}, Total Total:{sum(low_priority_nr_energy)}")
        log_msg(f"Total Low Priority Carbon [nr carbon, r carbon, total carbon]: {cal_operational_carbon(sum(low_priority_nr_energy), sum(low_priority_r_energy))}")
        for i in range(len(nr_energy)):
            log_msg(f"[{i}]Carbon footprint [nr carbon, r carbon, total carbon] {cal_operational_carbon(nr_energy[i], r_energy[i])}")
        output_file = f'tmp/output/{name}.txt'
        with open(output_file, "a") as f:
            f.write(f"Avg Low priority Avail {np.average([profile.avail() for profile in vm_profiles_all_subgraphs if profile.vm.priority == 1]):.2f}\n")
            f.write(f"Low priority downtime {np.sum([profile.downtime() for profile in vm_profiles_all_subgraphs if profile.vm.priority == 1]):.2f}\n")
            f.write(f"Total NR Energy {nr_energy}, Total Total:{sum(nr_energy)}\n")
            f.write(f"Total R Energy {r_energy}, Total Total:{sum(r_energy)}\n")
            f.write(f"Total Total Carbon [nr carbon, r carbon, total carbon]: {cal_operational_carbon(sum(nr_energy), sum(r_energy))}\n")
            f.write(f"Total Low Priority NR Energy {low_priority_nr_energy}, Total Total:{sum(low_priority_nr_energy)}\n")
            f.write(f"Total R Energy {low_priority_r_energy}, Total Total:{sum(low_priority_r_energy)}\n")
            f.write(f"Total Total Carbon [nr carbon, r carbon, total carbon]: {cal_operational_carbon(sum(low_priority_nr_energy), sum(low_priority_r_energy))}\n")
            for i in range(len(nr_energy)):
                f.write(f"[{i}]Carbon footprint [nr carbon, r carbon, total carbon] {cal_operational_carbon(nr_energy[i], r_energy[i])}\n")
            f.write("\n\n")
        # per_type_overhead = defaultdict(list)
        # per_type_migration = defaultdict(list)
        # total_migrations = 0
        # for profile in vm_profiles_all_subgraphs:
        #     per_type_overhead[profile.vm.name].append(profile.overhead_breakdown.tolist()+[int(profile.blackout)])
        #     per_type_migration[profile.vm.name].append(profile.migration_log)
        #     total_migrations += len(profile.migration_timestamp)
        # log_msg(f"Total migrations: {total_migrations}")
            
        # x = np.arange(len(current_renewable_traces))
        # for i, subgraph in enumerate(subgraphs):
        #     for site in subgraph[0]:
        #         plt.plot(x, current_renewable_traces[site], label=site)
        #     plt.plot(x, sum(current_renewable_traces[site] for site in subgraph[0]), label="total", color="purple")
        #     plt.xlabel("Hours")
        #     plt.ylabel("Power (MW)")
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.savefig(f"{subgraph_dir}/power_trace.pdf")
        #     plt.clf()

        for i in range(len(subgraphs)):
            # print(utilizations[i], max_cores[i])
            power_up_ratios.append(utilizations[i]/max_cores[i])
        log_msg(f"VB Utilization: {np.average(power_up_ratios) :.2f}") # , np.std(power_up_ratios))

        return


class PLACEMENT_MIP(object):
    def __init__(self, energy, num_subgraphs, step, timestep):
        self.energy = energy
        self.energy[self.energy < 0] = 0
        self.num_subgraphs = num_subgraphs
        self.max_steps = step
        self.timestep = timestep
        self.lookahead = 16
        self.sites_per_subgraph = 3

    def place(self, vms, start_time, init_placement=None, hint = None):
        model = gp.Model("batch", env=GRB_ENV)
        logfilename = f"logs/mip-vm-placement.log"
        if start_time == 0:
            logfile = open(logfilename, "w").close()
        model.Params.LogToConsole = 0
        model.Params.OutputFlag = 1
        model.Params.LogFile = logfilename

        T, M, N, K = min(self.lookahead, self.max_steps - start_time), len(vms), self.num_subgraphs, self.sites_per_subgraph

        if M == 0:
            return 0, None
    
        # vars 
        placement = model.addVars(M, N, vtype=GRB.BINARY)
        placement.BranchPriority=10
        vm_dist = model.addVars(M, K, vtype=GRB.INTEGER)
        progress = model.addVars(T, M, vtype=GRB.BINARY)
        # sufficient = model.addVars(T, M, vtype=GRB.BINARY)
        complete = model.addVars(T, M, vtype=GRB.BINARY)
        total_complete = model.addVar(vtype=GRB.CONTINUOUS) 
        avg_complete = model.addVar(vtype=GRB.CONTINUOUS) 
        t_complete = [None for i in range(M)]
        powered_core = None
        total_core = None
        total_progress = None
        
        aux0 = model.addVars(M, N, K, vtype=GRB.INTEGER)

        if hint is not None:
            assert(len(hint.shape) == 2)
            M_H, N_H = hint.shape
            for i in range(M_H):
                for j in range(N_H):
                    placement[i, j].start = int(hint[i, j])     
            model.update()


        '''placement constraint'''
        model.addConstrs(gp.quicksum(placement[i, j] for j in range(N)) <= 1 for i in range(M))
        # model.addConstrs(gp.quicksum(vm_dist[i, k] for k in range(K)) <= vms[i].num_vms for i in range(M))
        
        '''accumulative execution time constraints'''
        # model.addConstrs(gp.quicksum(progress[t, i] for t in range(T)) <= vms[i].completion_time / self.timestep for i in range(M))

        '''tasks completion constraints'''
        for t in range(T):
            # model.addConstrs(progress[t, i] * vms[i].num_vms <= gp.quicksum(vm_dist[i, k] for k in range(K)) for i in range(M))
            model.addConstrs(progress[t, i] <= gp.quicksum(placement[i, j] for j in range(N))  for i in range(M))
            model.addConstrs(gp.quicksum(progress[t0, i] for t0 in set(range(t+1))) >= vms[i].completion_time * complete[t, i] / self.timestep for i in range(M))

        '''powered cores & vm alloc'''
        # model.addConstrs(aux0[i, j, k] == vm_dist[i, k] * placement[i, j] for k in range(K) for j in range(N) for i in range(M))

        # powered_core = np.array([[[gp.quicksum(aux0[i, j, k] * progress[t, i] * vms[i].cores_per_vm for i in range(M)) for k in range(K)] for j in range(N)] for t in range(T)])
        powered_core = np.array([[gp.quicksum(placement[i, j] * progress[t, i] * vms[i].avg_power for i in range(M)) for j in range(N)] for t in range(T)])
        
        for t in range(T):
            model.addConstrs(powered_core[t, j] <= sum(self.energy[j, :, start_time+t]) for j in range(N))
            # model.addConstrs(powered_core[t, j, k] <= self.energy[j, k, start_time+t] for k in range(K) for j in range(N))

        '''max job completion constraints'''

        model.addConstrs(gp.quicksum(complete[t, i] for t in range(T)) <= 1 for i in range(M))
        t_complete = [(1-gp.quicksum(complete[t, i] for t in range(T)))*(T+10)+gp.quicksum(complete[t, i]*(t+1) for t in range(T)) for i in range(M)]

        model.addConstr(avg_complete == gp.quicksum(t_complete[i] for i in range(M)))


        # (5) objective

        '''Optimizing model'''

        model.setObjective(avg_complete / M, GRB.MINIMIZE)  
        # model.setObjective(avg_complete / M - np.average([vms[i].completion_time / self.timestep for i in range(M)]), GRB.MINIMIZE)  
        # total_core = gp.quicksum(powered_core[t, j] for j in range(N) for t in range(T))
        # model.setObjective(total_core, GRB.MINIMIZE)  
        # model.setObjectiveN(total_complete, 0, 1)
        # model.setObjectiveN(avg_complete / M, 0, 1)
        # model.setObjectiveN(total_core, 1, 0)
        model.Params.TIME_LIMIT = PLACEMENT_MIP_OPT_TIME
        model.Params.MIPGap = PLACEMENT_MIP_OPT_GOAL
        # model.Params.Presolve = 0
        model.Params.MIPFocus=3
        # model.Params.Cuts=0
        model.optimize()
        model.printQuality()

        ## results
        # log_msg(f"Objective: {model.objVal}")
        # log_msg(f"Total cores: {total_core.getValue()}")
        # log_msg(f"Completion time: {total_complete.x*timestep}")

        placement_x = np.array([[placement[i, j].x for j in range(N)] for i in range(M)], dtype=int)
        # progress_x = np.array([[progress[t, i].x for i in range(M)] for t in range(T)])
        
        np.set_printoptions(threshold=sys.maxsize)
        # print("Diff from hint", sum(sum(abs(placement_x-hint))) // 2)
        return model.objVal, placement_x
  
