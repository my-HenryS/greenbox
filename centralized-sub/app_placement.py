from macros import *

from renewable_simulation import BATCH_MIP, VBSimulator
from factory import App

class AppPlacement:
    max_cores = 320
    max_power = 20 # (MV)
    BATCH_MIP_OPT_TIME = None
    BATCH_MIP_OPT_GOAL = None


    def __init__(self, traces, n_sites, interp_factor, app_distribution=None):
        self.traces = traces
        self.n_sites = n_sites
        self.app_distribution = app_distribution
        self.interp_factor = interp_factor
        self.completion_cdf_results = defaultdict(dict)
        self.breakdown_results = defaultdict(dict)
        self.carbon_results = defaultdict(dict)
        # self.utilization_results = defaultdict(dict)
        self.app_distribution_results = defaultdict(dict)
        self.solution_results = defaultdict(dict)
    
    def placement(self, policy, start_time, end_time, subgraphs, scaled_factor, cloud_util=1.0, app_arrival_ratio=0.75, low_cutoff=0.0, max_cutoff=1.0, misprediction_ratio=0.0, default_apps=None, use_ray=True, default_app_arrival=None, batch_name="default"):
        site_policy = policy["site"]
        subgraph_policy = policy["subgraph"]
        n_sites = policy["n_sites"]
        placement_policy = policy["placement"]
        scheduling_policy = policy["scheduling"]

        selected_sites = [site for subgraph in subgraphs for site in subgraph[0]]

        full_policy_name = f"{site_policy} + {subgraph_policy} + {placement_policy} + {scheduling_policy} + {low_cutoff}"

        start_date = DATE_OFFSET + dt.timedelta(start_time / 24 / self.interp_factor)
        start_date = start_date.date()
        interval = end_time - start_time

        current_renewable_traces = self.traces.loc[start_time:end_time] * scaled_factor
        current_renewable_traces *= AppPlacement.max_cores / AppPlacement.max_power
        
        upper_cut_off_ratio = max_cutoff
        lower_cut_off_ratio = low_cutoff

        cloud_carbon_footprint = 0.0
        vb_carbon_footprint = 0.0

        # TODO: add smarter cutoff based on the average power 

        max_powers = []
        for site in selected_sites:
            max_power = max(current_renewable_traces[site])
            max_powers.append(max_power)
            total_power = sum(current_renewable_traces[site])
            max_power_cutoff = upper_cut_off_ratio * max_power
            min_power_cutoff = lower_cut_off_ratio * max_power
            current_renewable_traces[site] = current_renewable_traces[site].clip(upper=max_power_cutoff)
            total_power_after_upper_cutoff = sum(current_renewable_traces[site])
            current_renewable_traces[site] = current_renewable_traces[site].clip(lower=min_power_cutoff)
            total_power_after_lower_cutoff = sum(current_renewable_traces[site])
            total_power_for_cloud = max_power_cutoff * interval
            
            vb_carbon_footprint += total_power_after_lower_cutoff - total_power_after_upper_cutoff
            cloud_carbon_footprint += total_power_for_cloud

        print(max_powers)
        # log_msg("VB Footprint vs Cloud Footprint", vb_carbon_footprint / cloud_carbon_footprint)
        self.carbon_results[start_date][full_policy_name] = 100 * vb_carbon_footprint / cloud_carbon_footprint

        aggregated_max_power_for_cloud = sum([max(current_renewable_traces[site]) for site in selected_sites])
        aggregated_avg_power_for_cloud = sum([np.average(current_renewable_traces[site]) for site in selected_sites])
        

        # log_msg(aggregated_max_power_for_cloud, aggregated_avg_power_for_cloud)

        app_templates = create_templates()
        num_cores_per_vm = np.average([app.total_cores for app in app_templates])
        hours_per_vm = np.average([app.completion_time for app in app_templates]) / (60 / self.interp_factor) 
        max_app_per_hour = aggregated_avg_power_for_cloud / num_cores_per_vm / hours_per_vm
        # print(max_app_per_hour, num_cores_per_vm, hours_per_vm, aggregated_avg_power_for_cloud)
        # num_apps = int(max_app_per_hour * (sum(self.app_distribution) / max(self.app_distribution) / 24 ) * (interval * 1/2))
        cloud_num_apps = int(max_app_per_hour * app_arrival_ratio * interval * cloud_util)
        # num_apps = 70 #10000
        num_app_scale_factor = 900 / cloud_num_apps
        if cloud_num_apps != 900:
            cloud_num_apps = 900
            current_renewable_traces *= num_app_scale_factor
        log_msg("Cloud Utilization", cloud_util)
        # log_msg("Cloud Completion", hours_per_vm)

        if default_apps is None:
            # all_apps, apps_per_step = create_workloads([cloud_num_apps//5]*5, (interval * 1 / 2), "even", gap=self.interp_factor)
            all_apps, apps_per_step = create_workloads(cloud_num_apps, int(app_arrival_ratio * interval), "backlog", gap=self.interp_factor, template=app_templates)
            # all_apps, apps_per_step = create_workloads([num_apps//3]*3, (interval * 1/3), "custom", self.app_distribution)
            self.app_distribution_results[start_date][full_policy_name] = apps_per_step
        else:
            all_apps = default_apps
            
        num_apps = len(all_apps)
        all_apps_copy = copy.deepcopy(all_apps)
        log_msg("Num of apps for cloud", cloud_num_apps)
        log_msg("Num of apps scheduled", num_apps)
        if default_app_arrival is not None:
            log_msg("Num of apps scheduled from default", sum([len(_) for _ in default_app_arrival]))

        schedulers = []
        for i, subgraph in enumerate(subgraphs):
            if use_ray:
                schedulers.append(
                        VBSimulator.remote(scheduling_policy, 
                            current_renewable_traces[subgraph[0]].to_numpy().transpose(),
                            [], 
                            len(current_renewable_traces[subgraph[0]].columns),
                            opt_time=BATCH_MIP_OPT_TIME, 
                            opt_goal=BATCH_MIP_OPT_GOAL, 
                            step=interval,
                            timestep = 60 // self.interp_factor,
                            subgraph_id=i,
                            misprediction_ratio = misprediction_ratio) 
                        )
            else:
                schedulers.append(
                        VBSimulator(scheduling_policy, 
                            current_renewable_traces[subgraph[0]].to_numpy().transpose(),
                            [], 
                            len(current_renewable_traces[subgraph[0]].columns), 
                            opt_time=BATCH_MIP_OPT_TIME, 
                            opt_goal=BATCH_MIP_OPT_GOAL, 
                            step=interval,
                            timestep = 60 // self.interp_factor,
                            subgraph_id=i,
                            misprediction_ratio = misprediction_ratio)
                        )
                        


        all_energy = np.array([current_renewable_traces[subgraph[0]].values.sum(axis=1) for subgraph in subgraphs])
        app_placement = defaultdict(lambda : defaultdict(list))
        app_arrival = []
        total_t_schedule = []
        total_t_placement = []

        for t in range(interval):
            # arrived_apps = [app for app in all_apps if app.start == t]
            
            energy = [current_renewable_traces.iloc[t][subgraph[0]].values.sum(axis=0) for subgraph in subgraphs]
            if use_ray:
                remaining_energy = [schedulers[i].remaining_energy.remote(cloud_util) for i, subgraph in enumerate(subgraphs)]
                remaining_energy = ray.get([actor for actor in remaining_energy])
            else:
                remaining_energy = [schedulers[i].remaining_energy(cloud_util) for i, subgraph in enumerate(subgraphs)]
                
            remaining_energy = [np.sum(e) for e in remaining_energy]
            # decide arriving apps based on the remaining energy
            if default_app_arrival is not None:
                arrived_apps = default_app_arrival[t]
                num_remaining_apps = sum([len(_) for _ in default_app_arrival[t+1:]])
            else:
                total_remaining_energy = sum(remaining_energy)
                positive_remaining_energy = sum([_ for _ in remaining_energy if _ > 0])
                arrived_apps = []
                if t % self.interp_factor == 0:
                    all_schedulable_apps = all_apps # [app for app in all_apps if app.start < t + 1]
                    for app in all_schedulable_apps:
                        if app.total_cores < positive_remaining_energy:
                            arrived_apps.append(app)
                            app.start = t + app.base_start
                            positive_remaining_energy -= app.total_cores
                        else:
                            break
                all_apps = [app for app in all_apps if app not in arrived_apps]
                app_arrival.append(arrived_apps)
                num_remaining_apps = len(all_apps)
                
            cores = [app.total_cores for app in arrived_apps]
            log_msg(f"Time:{t}, Remaining: {num_remaining_apps}, Added: {len(arrived_apps)}, Energy diff: {sum(remaining_energy):.2f}", replace=True)

            def sanity_check(apps, subgraph_trace):
                SANITY_THRESHOLD = 100
                N = len(subgraph_trace)
                if sum(subgraph_trace) > SANITY_THRESHOLD:
                    return True

                for app in sorted(apps, key=lambda app: app.total_cores):
                    required_vms = app.num_vms
                    for j in range(N):
                        if subgraph_trace[j] >= app.cores_per_vm:
                            scheduled_vms = min(subgraph_trace[j] // app.cores_per_vm, required_vms)
                            subgraph_trace[j] -= scheduled_vms * app.cores_per_vm
                            required_vms -= scheduled_vms
                        if required_vms == 0:
                            break
                    if required_vms > 0:
                        return False
                return True
                    
                        


            for i in range(len(subgraphs)):
                app_placement[i][t] = []
                
      
            def greedy_placement():
                remaining_apps = [_ for _ in arrived_apps]
                app_placement_matrix = np.zeros((len(arrived_apps), len(subgraphs)))
                positive_remaining_energy = sum([_ for _ in remaining_energy if _ > 0])
                if positive_remaining_energy > 0:
                    core_per_subgraph = sum(cores) / positive_remaining_energy
    
                    for j, subgraph in enumerate(subgraphs):
                        if remaining_energy[j]  <= 0:
                            continue
                        my_cores = remaining_energy[j] * core_per_subgraph
                        selected_apps = []
                        for app in remaining_apps:
                            if app.total_cores < my_cores:
                                selected_apps.append(app)
                                app_placement_matrix[arrived_apps.index(app), j] = 1
                                my_cores -= app.total_cores
                            if my_cores <= 0:
                                break

                        remaining_apps = [app for app in remaining_apps if app not in selected_apps]
                return app_placement_matrix, remaining_apps
                        
            # schedule based on energy
            if placement_policy == "greedy":
                app_placement_matrix, remaining_apps = greedy_placement()

            if placement_policy == "even":
                apps_per_site = int(math.ceil(len(arrived_apps) / len(subgraphs)))
                for i, g in enumerate(subgraphs):
                    app_placement[i][t] = arrived_apps[i*apps_per_site:(i+1)*apps_per_site]

            if placement_policy == "mip":
                if use_ray:
                    predicted_remaining_energy = [schedulers[i].predicted_remaining_energy.remote(cloud_util) for i, subgraph in enumerate(subgraphs)]
                    predicted_remaining_energy = ray.get([actor for actor in predicted_remaining_energy])
                else:
                    predicted_remaining_energy = [schedulers[i].predicted_remaining_energy(cloud_util) for i, subgraph in enumerate(subgraphs)]
                
                predicted_remaining_energy = np.array(predicted_remaining_energy)
                
                placement = PLACEMENT_MIP(predicted_remaining_energy, len(subgraphs), step=interval, timestep=60//self.interp_factor)

                app_placement_hint, _ = greedy_placement()
                t_start = time.time()
                est_time, app_placement_matrix = placement.place(arrived_apps, t, init_placement=None, hint=app_placement_hint)
                t_placement = time.time() - t_start
                # print(t_placement)
                total_t_placement.append(t_placement)

            remaining_apps = []
            for i, app in enumerate(arrived_apps):
                if sum(app_placement_matrix[i]) == 0:
                    remaining_apps.append(app)
                    continue
                graph_i = app_placement_matrix[i].tolist().index(1)
                app_placement[graph_i][t].append(app)
            # print(len(arrived_apps), len(remaining_apps))
                
            subgraph_sorted_by_energy = sorted(enumerate(subgraphs), key=lambda x: remaining_energy[x[0]], reverse=True)
            for i, app in enumerate(remaining_apps):
                selected_subgraph = subgraph_sorted_by_energy[i % len(subgraphs)][0]
                app_placement[selected_subgraph][t].append(app)


            if use_ray:
                per_process_results = [] 
                for i in range(len(subgraphs)):
                    # log_msg(f"{i} {sum([app.total_cores for app in app_placement[i][t]])} {remaining_energy[i]}")
                    per_process_results.append(schedulers[i].migrate.remote(new_apps = app_placement[i][t]))
            
                ts_schedule = []
                for i in range(len(subgraphs)):
                    _, t_schedule = ray.get(per_process_results[i])
                    ts_schedule.append(t_schedule)
                total_t_schedule.append(max(ts_schedule))
            else:
                ts_schedule = []
                for i in range(len(subgraphs)):
                    _, t_schedule = schedulers[i].migrate(new_apps = app_placement[i][t])
                    ts_schedule.append(t_schedule)
                # if t == 0:
                #     log_msg(f"Subgraph {i}")
                
            # log_msg(f"Step {t}")
            # for i in range(len(subgraphs)):
            #     print(len(schedulers[i].apps), schedulers[i].t)
        log_msg("")
        log_msg(f"Total schedule time {sum(total_t_placement):.2f} + {sum(total_t_schedule):.2f} s")
        simulation_results = []
        cdf_t = []
        utilizations = []
        max_cores = []
        app_profiles_all_subgraphs = []
        power_up_ratios = []
        completed_vms = []
        solutions = dict()
        for i in range(len(subgraphs)):
            if use_ray:
                scheduler_finish_future = schedulers[i].finish.remote()
                max_t, avg_t, dist_t, completed_vm, app_profiles, global_profile, utilization, running_vms = ray.get(scheduler_finish_future)
            else:
                max_t, avg_t, dist_t, completed_vm, app_profiles, global_profile, utilization, running_vms = schedulers[i].finish()
            # log_msg(i, len(dist_t), avg_t, completed_vm, subgraphs[i][3])

            subgraph_sites = subgraphs[i][0]
            per_site_trace = current_renewable_traces[subgraph_sites]
            max_cores.append(np.max(np.sum(per_site_trace, axis=1)) * interval * app_arrival_ratio)
            utilizations.append(sum(utilization[:int(interval * app_arrival_ratio)]))
            completed_vms.append(completed_vm)
            solutions[i] = running_vms

            power_up_ratios = []

            if np.isnan(avg_t):
                avg_t = interval

            if avg_t != 0:
                simulation_results.append(avg_t)
                cdf_t.extend(dist_t)  
                app_profiles_all_subgraphs.extend(app_profiles) 
                
            batch_dir = f'raw/{batch_name}/day_{start_time//self.interp_factor//24}'
            if not os.path.exists(batch_dir):
                os.makedirs(batch_dir, exist_ok=True)
            
            with open(f"{batch_dir}/batch_info", "w") as f:
                batch_info = dict()
                batch_info['num_apps'] = num_apps
                batch_info['num_graphs'] = len(subgraphs)
                batch_info['interval'] = interval
                json.dump(batch_info, f)

            subgraph_dir = f'{batch_dir}/subgraphs/graph_{i}'
            if not os.path.exists(subgraph_dir):
                os.makedirs(subgraph_dir, exist_ok=True)
                
            policy_dir = f'{subgraph_dir}/{full_policy_name}'
            if not os.path.exists(policy_dir):
                os.mkdir(policy_dir)
            
            with open(f'{policy_dir}/global_profile', 'wb') as f:
                pickle.dump(global_profile, f)
            
            with open(f'{policy_dir}/app_profiles', 'wb') as f:  
                pickle.dump(app_profiles, f)
                
            with open(f'{policy_dir}/running_vms', 'wb') as f:
                pickle.dump(running_vms, f)
                
            with open(f'{policy_dir}/power_traces', 'wb') as f:
                pickle.dump(current_renewable_traces[subgraphs[i][0]], f)

        # log_msg(f"Average completion time across graphs: {sum(simulation_results) / len(simulation_results):.2f} hours.")
        app_profiles_all_subgraphs = [profile for profile in app_profiles_all_subgraphs if profile.finished]
        log_msg(f"Avg JCT: {np.average([profile.end - profile.start for profile in app_profiles_all_subgraphs]):.2f}. 95% Slowdown: {np.percentile([(profile.end - profile.start) / profile.exec for profile in app_profiles_all_subgraphs], 95):.2f}. Migration: {np.average([profile.overhead for profile in app_profiles_all_subgraphs]):.2f}. Queuing: {np.average([profile.queuing for profile in app_profiles_all_subgraphs]):.2f}. Blackout: {np.average([profile.blackout for profile in app_profiles_all_subgraphs]):.2f}.", emphasize=True)
        log_msg(f"Allocation: {np.average([profile.overhead_breakdown[0] for profile in app_profiles_all_subgraphs]):.2f}. Recompute: {np.average([profile.overhead_breakdown[1] for profile in app_profiles_all_subgraphs]):.2f}. Latency: {np.average([profile.overhead_breakdown[2] for profile in app_profiles_all_subgraphs]):.2f}.")
        
        affected_app_profiles = [profile for profile in app_profiles_all_subgraphs if profile.overhead + profile.queuing + profile.blackout > 0 ]
        log_msg(f"Avg JCT of affected workloads: {np.average([profile.end - profile.start for profile in affected_app_profiles]):.2f}. Migration: {np.average([profile.overhead for profile in affected_app_profiles]):.2f}. Queuing: {np.average([profile.queuing for profile in affected_app_profiles]):.2f}. Blackout: {np.average([profile.blackout for profile in affected_app_profiles]):.2f}. # of affected workloads: {len(affected_app_profiles)}", emphasize=True)
        
        log_msg(f"Total completed Apps {sum(completed_vms)}")
        
        per_type_overhead = defaultdict(list)
        per_type_migration = defaultdict(list)
        total_migrations = 0
        for profile in app_profiles_all_subgraphs:
            per_type_overhead[profile.app.name].append(profile.overhead_breakdown.tolist()+[int(profile.blackout)])
            per_type_migration[profile.app.name].append(profile.migration_timestamp)
            total_migrations += len(profile.migration_timestamp)
        log_msg(f"Total migrations: {total_migrations}")
            
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

        self.completion_cdf_results[start_date][full_policy_name] = cdf_t
        self.breakdown_results[start_date][full_policy_name] = app_profiles_all_subgraphs
        self.solution_results = solutions

        return np.average(cdf_t), app_arrival, all_apps_copy


    def plot_all(self):
        if len(self.completion_cdf_results) == 0:
            return 

        for date in self.completion_cdf_results.keys():
            self.plot(self.completion_cdf_results[date], date)
        
        aggregated_cdf_results = defaultdict(list)

        for date in self.completion_cdf_results:
            for i, (policy, cdf_t) in enumerate(self.completion_cdf_results[date].items()):
                aggregated_cdf_results[policy] += cdf_t

        self.plot(aggregated_cdf_results)


        plot_dir = f"{PLOT_DIR}/{self.n_sites}/scheduling_policies/"

        line_styles = ['-', '--', '-.', ':']
        hatches = ["", "\\", "//", "||"]
        colors = ['blue', "red", "orange", "green", 'black']
        plt.figure(figsize=(8, 4), dpi=300)

        hist = list(list(self.app_distribution_results.values())[0].values())[0]
        x_values = np.arange(len(hist))
        plt.plot(x_values, hist, linestyle='-', color="blue", linewidth=2)

        plt.xlabel("Hours")
        plt.ylabel("Job arrival rate")
        plt.tight_layout()
        plt.grid(visible=True, which='major', axis='y', color='#000000', linestyle='--')
        plt.savefig(f"{plot_dir}/job_arrival_rate.pdf")
        plt.clf()
        
        

    def plot(self, completion_cdf, date=None):
        if date:
            plot_dir = f"{PLOT_DIR}/{self.n_sites}/scheduling_policies/{date}"
        else:
            plot_dir = f"{PLOT_DIR}/{self.n_sites}/scheduling_policies/"

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)


        line_styles = ['-', '--', '-.', ':']
        hatches = ["", "\\", "//", "||"]
        colors = ['blue', "red", "orange", "green", 'black']
        plt.figure(figsize=(8, 4), dpi=300)

        for i, (policy, cdf_t) in enumerate(completion_cdf.items()):
            cdf_t = sorted(cdf_t, reverse=False)
            x_values = np.arange(len(cdf_t))/len(cdf_t) * 100
            plt.plot(x_values, cdf_t, line_styles[i % len(line_styles)], label=policy, color=colors[i % len(colors)], linewidth=2)

        plt.legend(ncol=1, bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=8)
        plt.xlabel("CDF (%)")
        plt.ylabel("Job completion time (hrs)")
        plt.tight_layout()
        plt.grid(visible=True, which='major', axis='y', color='#000000', linestyle='--')
        plt.savefig(f"{plot_dir}/cdf_of_completion.pdf")
        plt.clf()

        x = [i*1+j*0.25 for i in range(4) for j in range(3)]
        x_labels = []
        y_values = []
        for i, (policy, cdf_t) in enumerate(completion_cdf.items()):
            x_labels.append(policy)
            y_values.append(sum(cdf_t) / len(cdf_t))
        
        # for i in range(4):
        #     for j in range(3):
        #         plt.bar(x[i*3+j], y_values[i*3+j], zorder=3, color=colors[j], width=0.25)
        # plt.xticks(x, labels=x_labels)
        plt.bar(x_labels, y_values, zorder=3, width=0.25)

        plt.xlabel("Policy")
        plt.ylabel("Average job completion time (hrs)")
        plt.tick_params(axis='x', rotation=45)
        plt.grid(visible=True, which='major', axis='y', color='#000000', linestyle='--', zorder=0)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/average_completion.pdf")
        plt.clf()



        if date:
            x_labels = []
            execution = []
            overhead = []
            queuing = []
            blackout = []
            for i, (policy, breakdowns) in enumerate(self.breakdown_results[date].items()):
                x_labels.append(policy)
                execution.append(np.average([profile.exec for profile in breakdowns]))
                overhead.append(np.average([profile.overhead for profile in breakdowns]))
                queuing.append(np.average([profile.queuing for profile in breakdowns]))
                blackout.append(np.average([profile.blackout for profile in breakdowns]))
                # print([profile.queuing for profile in breakdowns])
            
            
            execution, overhead, queuing, blackout = np.array(execution), np.array(overhead), np.array(queuing), np.array(blackout)
            # print(execution, overhead, queuing)
            plt.bar(x_labels, execution, zorder=3, width=0.3, label="execution")
            plt.bar(x_labels, queuing, bottom=execution, zorder=3, width=0.3, label="queuing")
            plt.bar(x_labels, blackout, bottom=execution+queuing, zorder=3, width=0.3, label="blackout")
            plt.bar(x_labels, overhead, bottom=execution+queuing+blackout, zorder=3, width=0.3, label="overhead")

            plt.xlabel("Policy")
            plt.ylabel("Average job completion time (hrs)")
            plt.tick_params(axis='x', rotation=30)
            plt.grid(visible=True, which='major', axis='y', color='#000000', linestyle='--', zorder=0)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/completion_breakdown.pdf")
            plt.clf()

            plt.bar(x_labels, queuing, zorder=3, width=0.3, label="queuing")
            plt.bar(x_labels, blackout, bottom=queuing, zorder=3, width=0.3, label="blackout")
            plt.bar(x_labels, overhead, bottom=queuing+blackout, zorder=3, width=0.3, label="overhead")

            plt.xlabel("Policy")
            plt.ylabel("Migration overhead (hrs)")
            plt.tick_params(axis='x', rotation=30)
            plt.grid(visible=True, which='major', axis='y', color='#000000', linestyle='--', zorder=0)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/migration_overhead.pdf")
            plt.clf()


            x_labels = []
            carbons = []
            for i, (policy, carbon) in enumerate(self.carbon_results[date].items()):
                x_labels.append(policy)
                carbons.append(carbon)

            plt.bar(x_labels, carbons, zorder=3, width=0.3, label="execution")

            plt.xlabel("Policy")
            plt.ylabel("% of Cloud Carbon Footprint")
            plt.tick_params(axis='x', rotation=30)
            plt.grid(visible=True, which='major', axis='y', color='#000000', linestyle='--', zorder=0)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/carbon_footprint.pdf")
            plt.clf()



class PLACEMENT_MIP(object):
    def __init__(self, energy, num_subgraphs, step, timestep):
        self.energy = energy
        self.energy[self.energy < 0] = 0
        self.num_subgraphs = num_subgraphs
        self.max_steps = step
        self.timestep = timestep
        self.lookahead = 16
        self.sites_per_subgraph = 3

    def place(self, apps, start_time, init_placement=None, hint = None):
        model = gp.Model("batch", env=GRB_ENV)
        logfilename = f"logs/mip-app-placement.log"
        if start_time == 0:
            logfile = open(logfilename, "w").close()
        model.Params.LogToConsole = 0
        model.Params.OutputFlag = 1
        model.Params.LogFile = logfilename

        T, M, N, K = min(self.lookahead, self.max_steps - start_time), len(apps), self.num_subgraphs, self.sites_per_subgraph

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
        # model.addConstrs(gp.quicksum(vm_dist[i, k] for k in range(K)) <= apps[i].num_vms for i in range(M))
        
        '''accumulative execution time constraints'''
        # model.addConstrs(gp.quicksum(progress[t, i] for t in range(T)) <= apps[i].completion_time / self.timestep for i in range(M))

        '''tasks completion constraints'''
        for t in range(T):
            # model.addConstrs(progress[t, i] * apps[i].num_vms <= gp.quicksum(vm_dist[i, k] for k in range(K)) for i in range(M))
            model.addConstrs(progress[t, i] <= gp.quicksum(placement[i, j] for j in range(N))  for i in range(M))
            model.addConstrs(gp.quicksum(progress[t0, i] for t0 in set(range(t+1))) >= apps[i].completion_time * complete[t, i] / self.timestep for i in range(M))

        '''powered cores & vm alloc'''
        # model.addConstrs(aux0[i, j, k] == vm_dist[i, k] * placement[i, j] for k in range(K) for j in range(N) for i in range(M))

        # powered_core = np.array([[[gp.quicksum(aux0[i, j, k] * progress[t, i] * apps[i].cores_per_vm for i in range(M)) for k in range(K)] for j in range(N)] for t in range(T)])
        powered_core = np.array([[gp.quicksum(placement[i, j] * progress[t, i] * apps[i].total_cores for i in range(M)) for j in range(N)] for t in range(T)])
        
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
        # model.setObjective(avg_complete / M - np.average([apps[i].completion_time / self.timestep for i in range(M)]), GRB.MINIMIZE)  
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
  
