from macros import *

def random_combination(iterable,r):
    i = 0
    pool = tuple(iterable)
    n = len(pool)
    rng = range(n)
    while True:
        yield [pool[j] for j in random.sample(rng, r)] 

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
    plt.savefig(f"{PLOT_DIR}/vb_sites.jpg")
    plt.clf()

def calc_dist(coords, sites):
    lat, lon = zip(*[coords[site.split("-")[0]] for site in sites])
    # x, y = [], []
    # for i in range(len(lat)):
    #     (tmp_x, tmp_y, _, _) = utm.from_latlon(lat[i], lon[i])
    #     x.append(tmp_x)
    #     y.append(tmp_y)
    centroid = (np.mean(lat), np.mean(lon))
    dist = 0
    for i in range(len(lat)):
        coord = (lat[i], lon[i])
        dist += h3.point_dist(coord, centroid, unit="km")
    return dist

class SiteSelection:
    def __init__(self, vb_sites, vb_coords, capacity, traces, n_sites, interp_factor):
        # site pruning
        self.power_cdf_results = dict()
        self.power_cov_cdf_results = dict()
        self.total_power_results = dict()
        
        self.vb_sites = vb_sites
        self.vb_coords = vb_coords
        self.traces = traces
        self.capacity = capacity
        self.n_sites = n_sites
        self.interp_factor = interp_factor


    def plot(self, site_policy, start_time, end_time, selected_sites, rest_sites, trace):
        plot_dir = f"{PLOT_DIR}/{self.n_sites}/{site_policy}"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)
        
        interval = end_time - start_time
        x = np.arange(0, interval + 1,1)
        sum_ = 0
        for site in selected_sites:
            plt.plot(x, trace.loc[start_time:end_time][site], label=site)
            sum_ += sum(trace.loc[start_time:end_time][site]) / interval

        plt.plot(x, sum(trace.loc[start_time:end_time][site] for site in selected_sites), label="total", color="purple")

        plt.xlabel("Hours")
        plt.ylabel("Power (MW)")
        plt.savefig(f"{plot_dir}/selected_sites.jpg")
        plt.clf()


        for site in rest_sites:
            plt.plot(x, trace.loc[start_time:end_time][site], label=site)
        plt.plot(x, sum(trace.loc[start_time:end_time][site] for site in rest_sites), label="total", color="purple")
        plt.xlabel("Hours")
        plt.ylabel("Power (MW)")
        plt.savefig(f"{plot_dir}/remaining_sites.jpg")
        plt.clf()

        plot_sites(self.vb_coords, selected_sites)

    def plot_all(self):
        plot_dir = f"{PLOT_DIR}/{self.n_sites}/site_policies/"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)

        line_styles = ['-', '--', '-.', ':']
        colors = ['black', 'blue', "red", "orange"]
        plt.figure(figsize=(8, 4), dpi=300)
        site_policies = set([policy for i, (policy, cdf_p) in enumerate(self.power_cdf_results.items())])


        for i, (policy, cdf_p) in enumerate(self.power_cdf_results.items()):
            cdf_p = sorted(cdf_p, reverse=False)
            x_values = np.arange(len(cdf_p))/len(cdf_p) * 100
            plt.plot(x_values, cdf_p, line_styles[i % len(site_policies)], label=policy, color=colors[i], linewidth=2)

        plt.legend(ncol=1, bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=8)
        plt.xlabel("CDF (%)")
        plt.ylabel("Average power generation per site (MW)")
        plt.tight_layout()
        plt.grid(visible=True, which='major', axis='y', color='#000000', linestyle='--')
        plt.savefig(f"{plot_dir}/cdf_of_power.pdf")
        plt.clf()

        for i, (policy, cdf_p) in enumerate(self.power_cov_cdf_results.items()):
            cdf_p = sorted(cdf_p, reverse=False)
            x_values = np.arange(len(cdf_p))/len(cdf_p) * 100
            plt.plot(x_values, cdf_p, line_styles[i % len(site_policies)], label=policy, color=colors[i], linewidth=2)

        plt.legend(ncol=1, bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=8)
        plt.xlabel("CDF (%)")
        plt.ylabel("CoV of power generation per site")
        plt.tight_layout()
        plt.grid(visible=True, which='major', axis='y', color='#000000', linestyle='--')
        plt.savefig(f"{plot_dir}/cdf_of_power_cov.pdf")
        plt.clf()


        line_styles = ['--', '-.', ':', '-']
        colors = ['black', 'blue', "red", "orange"]

        for i, (policy, trace) in enumerate(self.total_power_results.items()):
            plt.plot(trace, line_styles[i % len(line_styles)], label=policy, color=colors[i % len(colors)], linewidth=2)

        plt.legend(ncol=1, bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=8)
        plt.xlabel("Time (hrs)")
        plt.ylabel("Total power generation (MW)")
        plt.tight_layout()
        plt.grid(visible=True, which='major', axis='y', color='#000000', linestyle='--')
        plt.savefig(f"{plot_dir}/total_power.pdf")
        plt.clf()

    def select(self, policy, start_time, end_time, capacity_scaling=None,  low_cutoff=0.0, max_cutoff=1.0, core_to_power_ratio=1.0, batch_name="default"):
        site_policy = policy["site"]
        
        upper_cut_off_ratio = max_cutoff
        lower_cut_off_ratio = low_cutoff

        # TODO: add smarter cutoff based on the average power 
        scaled_traces = self.traces
        scaled_factor = core_to_power_ratio

        max_powers = []
        for site in self.vb_sites:
            max_power = max(scaled_traces[site])
            max_powers.append(max_power)
            max_power_cutoff = upper_cut_off_ratio * max_power
            min_power_cutoff = lower_cut_off_ratio * max_power
            scaled_traces[site] = scaled_traces[site].clip(upper=max_power_cutoff)
            scaled_traces[site] = scaled_traces[site].clip(lower=min_power_cutoff)
        
        if site_policy == "random":
            # random pick
            random.shuffle(self.vb_sites)
            selected_sites = self.vb_sites[:self.n_sites]
        else:
        # if site_policy in ["avg_min", "avg", "avg_cov", "max_avg"]:
            def get_metrics(start_time, end_time):
                test_trace = scaled_traces.loc[start_time:end_time]
                metrics = dict()
                for site in self.vb_sites:
                    per_site_trace = np.array(test_trace[site])
                    avg_p = sum(per_site_trace) / (end_time - start_time)
                    if avg_p == 0.0:
                        metrics[site] = (0, float('inf'), 0, 0, float('inf'), float('inf'))
                        continue
                    cov_p = np.std(per_site_trace) / avg_p
                    min_p = min(per_site_trace)
                    max_p = max(per_site_trace)
                    max_avg_p = avg_p / max_p
                    diff_p = sum(np.abs(np.diff(per_site_trace))) / avg_p
                    cov_diff_p = np.std(np.abs(np.diff(per_site_trace))) / avg_p
                    metrics[site] = (avg_p, cov_p, min_p, max_avg_p, diff_p, cov_diff_p)
                return metrics

            ## calculate metrics of each site
            base_metrics = get_metrics(start_time, end_time)

            ## test the stableness of each metric
            metric = ["avg power", "cov of power", "min power", "max avg power", "diff of power", "cov of diff of power"]
            order = [True, False, True, True, False, False]
            ranks = []
            for i in range(len(metric)):
                base_site_rank = sorted(self.vb_sites, key=lambda s:base_metrics[s][i], reverse=order[i])
                ranks.append(base_site_rank)

            ## pick sites based on metric
            scores = dict()
            for site in self.vb_sites:
                score = 0.0
                if site_policy == "avg_min":
                    metrics = [0,2]
                elif site_policy == "avg_cov":
                    # metrics = [0,1]
                    metrics = [3]
                elif site_policy == "avg":
                    metrics = [0]
                elif site_policy == "cov":
                    metrics = [1]
                elif site_policy == "max_avg":
                    metrics = [0, 1]
                elif site_policy == "avg_max_avg":
                    metrics = [0, 3]
                for i in metrics:
                    score += ranks[i].index(site)
                scores[site] = score

            scores = sorted(scores.items(), key=lambda x:x[1])
            selected_sites = [s[0] for s in scores[:self.n_sites]]

            # print("selected", np.average([base_metrics[s][1] for s in selected_sites]))
            # rest_sites = [s for s in self.vb_sites if s not in selected_sites]
            # print("non-selected", np.average([base_metrics[s][1] for s in rest_sites if base_metrics[s][1] != float('inf')]))
            

        rest_sites = [s for s in self.vb_sites if s not in selected_sites]
        total_capacity = sum([self.capacity[site] for site in selected_sites])
        if capacity_scaling:
            scaled_factor *= capacity_scaling / total_capacity
        scaled_traces = self.traces * scaled_factor

        self.power_cdf_results[site_policy] = [np.average(scaled_traces.loc[start_time:end_time][site]) for site in selected_sites]

        self.total_power_results[site_policy] = np.sum([scaled_traces.loc[start_time:end_time][site] for site in selected_sites], axis=0)

        self.power_cov_cdf_results[site_policy] = [np.std(scaled_traces.loc[start_time:end_time][site]) / np.average(scaled_traces.loc[start_time:end_time][site]) for site in selected_sites if np.average(scaled_traces.loc[start_time:end_time][site]) != 0]
        self.plot(site_policy, start_time, end_time, selected_sites, rest_sites, self.traces)

        # log_msg(f"{len([s for s in selected_sites if 'solar' in s])} solar sites")
        log_msg(f"Total capacity: {sum([self.capacity[site] for site in selected_sites]):.2f} MW, Scaling factor: {scaled_factor:.2f}")
        
        output_dir = f'output/{batch_name}/'
        if not os.path.exists(f"{output_dir}"):
            os.makedirs(f"{output_dir}", exist_ok=True)

        outfile = f"{output_dir}/site_cov.json"
        site_cov = dict()
        # if os.path.exists(outfile):
        #     with open(outfile) as fp:
        #         site_cov = json.load(fp)
        site_cov[site_policy] = [np.std(scaled_traces.loc[start_time:end_time][site]) / np.average(scaled_traces.loc[start_time:end_time][site]) for site in selected_sites if np.average(scaled_traces.loc[start_time:end_time][site]) != 0]
        # with open(outfile, "w") as out_f:
        #     json.dump(site_cov, out_f)

        return selected_sites, total_capacity, scaled_traces


class SubGraphSelection:

    def __init__(self, vb_coords, capacity, traces, n_sites, interp_factor):
        self.power_cdf_results_subgraph = defaultdict(dict)
        self.power_cov_cdf_results_subgraph = defaultdict(dict)
        self.power_stable_cdf_results_subgraph = defaultdict(dict)

        self.vb_coords = vb_coords
        self.traces = traces
        self.capacity = capacity
        self.n_sites = n_sites
        self.interp_factor = interp_factor


    def select(self, policy, start_time, end_time, vb_sites, scaled_traces=None, previous_subgraphs=[], factor=0.0, max_k=3, batch_name="default", use_cache=False):

        site_policy = policy["site"]
        subgraph_policy = policy["subgraph"]
        # n_sites = policy["n_sites"]
        n_sites = self.n_sites

        plot_dir = f"{PLOT_DIR}/{n_sites}/{site_policy}/{subgraph_policy}"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)
        full_policy_name = f"{site_policy} + {subgraph_policy}"

        start_date = DATE_OFFSET + dt.timedelta(start_time / 24 / self.interp_factor)
        start_date = start_date.date()
        previous_subgraphs = [v[0] for v in previous_subgraphs]
        
        output_dir = f'output/{batch_name}/'
        subgraphs_obj_file = f"{output_dir}/subgraphs.obj"
        if use_cache and os.path.exists(subgraphs_obj_file):
            with open(subgraphs_obj_file, "rb") as in_f:
                subgraphs = pickle.load(in_f)
            return subgraphs
        
        if scaled_traces is not None:
            self.traces = scaled_traces

        all_combinations = []
        # all_samples = [list(v) for v in itertools.combinations(vb_sites, max_k)]
        
        def nCr(n,r):
            f = math.factorial
            return f(n) // f(r) // f(n-r)
        len_all_samples = nCr(len(vb_sites), max_k)
        
        max_subgraphs = 1000000
        if len_all_samples >= max_subgraphs*2:
            all_samples = random_combination(vb_sites, max_k)
            log_msg(f"Total {len_all_samples} combinations")
        else:
            all_samples = [list(v) for v in itertools.combinations(vb_sites, max_k)]
            log_msg(f"Total {len(all_samples)} combinations")
        for combination in all_samples: #random_combination(vb_sites, max_k):
            # dist_check = True
            # for site0, site1 in itertools.combinations(combination, 2):
            #     coords0 = self.vb_coords[site0.split("-")[0]]
            #     coords1 = self.vb_coords[site1.split("-")[0]]
            #     dist = h3.point_dist(coords0, coords1, unit='km')
            #     if 2 * dist / 100 > 50 and max_k != n_sites:
            #         dist_check = False
            #         break

            # all_types = set([site.split("-")[1] for site in combination])
            # type_check = len(all_types) == 2

            # if dist_check:
                # all_combinations.append(combination)
            all_combinations.append(combination)
            if len(all_combinations) >= max_subgraphs:
                break
        log_msg(f"Total {len(all_combinations)} combinations")


        self.traces = self.traces.loc[start_time:end_time]
        results = []

        all_sites = {s:i for i, s in enumerate(self.traces.columns)}
        np_traces = self.traces.to_numpy()

        def obtain_metrics(combinations, i, total_cores):
            # global combinations
            os.system("taskset -p 0xfffff %d > /dev/null" % os.getpid())
            results = []
            per_core_workload = math.ceil(len(combinations) / total_cores)
            log_msg(f"{i} started")
            # pd.options.compute.use_numexpr = True
            for combination in combinations[i*per_core_workload:(i+1)*per_core_workload]:
                t1 = time.time()
                indexes = [all_sites[s] for s in combination]
                traces = [np_traces[:, index] for index in indexes]
                total_energy = np.sum(traces, axis=0)
                t2 = time.time()
                avg = np.sum(total_energy) / len(total_energy)
                if avg == 0:
                    continue
                cov = np.std(total_energy) / avg
                min_ = np.min(total_energy) # / avg
                total_dist = sum([h3.point_dist(self.vb_coords[site0.split("-")[0]], self.vb_coords[site1.split("-")[0]], unit='km') for site0, site1 in itertools.combinations(combination, 2)])
                # total_dist = None
                result = (combination, cov, avg, min_, total_dist)
                if np.isnan(cov):
                    log_msg(combination)
                    for site in combination:
                        log_msg(self.traces[site])
                    log_msg(total_energy)
                    exit(0)
                results += [result]
                # print(time.time() - t2, t2-t1)
            
            log_msg(f"{i} finished")
            return results


        # pool = multiprocessing.Pool(4)
        # results = zip(*pool.map(obtain_metrics, all_combinations))
        if len(all_combinations) >= 5000:
            if len(all_combinations) >= 100000:
                num_jobs = 3
            else:
                num_jobs = 8
            per_pool_results = Parallel(n_jobs=num_jobs)(delayed(obtain_metrics)(all_combinations, i, num_jobs) for i in range(num_jobs))
            for result in per_pool_results:
                results += result
        else:
            results = obtain_metrics(all_combinations, 0, 1)

        
        if subgraph_policy == "stable":
            sort_func = lambda x: x[3] / x[2]
        elif subgraph_policy == "cov":
            sort_func = lambda x: 1 / x[1]
        elif subgraph_policy == "geo":
            sort_func = lambda x: 1 / x[4]

        total_stable_energy = 0
        total_avg_energy = 0
        total_cov = 0
        migration_amount = 0
        subgraphs = []
        nodes = set()

        sorted_results = list(sorted(results, key=sort_func, reverse=True))
        
        if len(previous_subgraphs) != 0:
            sorted_subgraphs = list(zip(*sorted_results))[0]
            sorted_previous_subgraphs = sorted(previous_subgraphs, key=lambda g: sorted_subgraphs.index(g))
            for previous_g in sorted_previous_subgraphs:
                index = sorted_subgraphs.index(previous_g)
                if index <= len(sorted_results)*factor:
                    selected_subgraph = sorted_results[index]
                    assert(not nodes.intersection(set(selected_subgraph[0])))
                    subgraphs.append(selected_subgraph)
                    nodes.update(selected_subgraph[0])
                    total_stable_energy += selected_subgraph[3]
                    total_avg_energy += selected_subgraph[2]
                    total_cov += selected_subgraph[1]
              
        least_cov = 2
        subgraph1 = sorted_results[0]
        subgraph2 = sorted_results[1]
        for result in sorted_results:
            if len(nodes) == n_sites:
                break
            if n_sites == 6:
                # choose the most average two subgraphs
                tmp_nodes = set(result[0])
                for another_result in sorted_results:
                    if tmp_nodes.intersection(set(another_result[0])):
                        continue
                    curr_cov = result[1]
                    another_cov = another_result[1]
                    avg = (curr_cov + another_cov) / 2
                    if avg < least_cov:
                        subgraph1 = result
                        subgraph2 = another_result
                        least_cov = avg
                    break
                continue

            if nodes.intersection(set(result[0])):
                continue
            dist = calc_dist(self.vb_coords, result[0])
            result = tuple(list(result) + [dist])
            subgraphs.append(result)
            nodes.update(result[0])
            # if result[0] in previous_subgraphs:
            #     migration_amount += len(result[0])
            migration_amount = None
            total_stable_energy += result[3]
            total_avg_energy += result[2]
            total_cov += result[1]
        
        dist = calc_dist(self.vb_coords, subgraph1[0])
        print(f"subgraph1 distance: {dist}")
        dist = calc_dist(self.vb_coords, subgraph2[0])
        print(f"subgraph2 distance: {dist}")
        if n_sites == 6:
            subgraphs.append(subgraph1)
            subgraphs.append(subgraph2)
        
        self.power_cdf_results_subgraph[start_date][full_policy_name] = [subgraph[2] for subgraph in subgraphs]
        self.power_cov_cdf_results_subgraph[start_date][full_policy_name] = [subgraph[1] for subgraph in subgraphs]
        self.power_stable_cdf_results_subgraph[start_date][full_policy_name] = [subgraph[3] for subgraph in subgraphs]
        

        # log_msg(date, subgraphs)
        log_msg(f"Date {start_date}: Stable energy {total_stable_energy:.2f} MW; CoV {total_cov / len(subgraphs):.2f}; Migration {migration_amount}")
        
        # if not os.path.exists(f"{plot_dir}/{start_date}"):
        #     os.makedirs(f"{plot_dir}/{start_date}", exist_ok=True)
            
        # if not os.path.exists(f"{output_dir}"):
        #     os.makedirs(f"{output_dir}", exist_ok=True)
            
        # outfile = f"{output_dir}/subgraph_cov.json"
        # subgraph_cov = dict()
        # if os.path.exists(outfile):
        #     with open(outfile) as fp:
        #         subgraph_cov = json.load(fp)
        # subgraph_cov[full_policy_name] = [subgraph[1] for subgraph in subgraphs]
        # with open(outfile, "w") as out_f:
        #     json.dump(subgraph_cov, out_f)
            
        # subgraphs_obj_file = f"{output_dir}/subgraphs.obj"
        # with open(subgraphs_obj_file, "wb") as out_f:
        #     pickle.dump(subgraphs, out_f)

        self.plot(policy, start_time, end_time, subgraphs, self.traces)
        return subgraphs

    def plot(self, policy, start_time, end_time, subgraphs, trace):
        site_policy = policy["site"]
        subgraph_policy = policy["subgraph"]
        n_sites = policy["n_sites"]

        plot_dir = f"{PLOT_DIR}/{self.n_sites}/{site_policy}/{subgraph_policy}"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)
        
        interval = end_time - start_time
        x = np.arange(0, interval + 1,1)
        i = 0
        for subgraph in subgraphs:
            sites = subgraph[0]
            for site in sites:
                plt.plot(x, trace.loc[start_time:end_time][site], label=site)

            plt.plot(x, sum(trace.loc[start_time:end_time][site] for site in sites), label="total", color="purple")

            plt.xlabel("Hours")
            plt.ylabel("Power (MW)")
            plt.savefig(f"{plot_dir}/subgraph{i}.jpg")
            plt.clf()
            i += 1


    def plot_all(self):
        for date in self.power_cdf_results_subgraph.keys():
            self.plot_date(date)

    def plot_date(self, date):
        plot_dir = f"{PLOT_DIR}/{self.n_sites}/subgraph_policies/{date}"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)

        line_styles = ['-', '--', '-.', ':']
        hatches = ["", "\\", "//", "||"]
        colors = ['blue', "red", "orange", "green", 'black']
        plt.figure(figsize=(8, 4), dpi=300)

        for i, (policy, cdf_p) in enumerate(self.power_cov_cdf_results_subgraph[date].items()):
            cdf_p = sorted(cdf_p, reverse=False)
            x_values = np.arange(len(cdf_p))/len(cdf_p) * 100
            plt.plot(x_values, cdf_p, line_styles[i % len(line_styles)], label=policy, color=colors[i % len(colors)], linewidth=2)

        plt.legend(ncol=1, bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=8)
        plt.xlabel("CDF (%)")
        plt.ylabel("CoV of power generation per subgraph")
        plt.tight_layout()
        plt.grid(visible=True, which='major', axis='y', color='#000000', linestyle='--')
        plt.savefig(f"{plot_dir}/cdf_of_power_cov_subgraph.pdf")
        plt.clf()

        for i, (policy, cdf_p) in enumerate(self.power_cdf_results_subgraph[date].items()):
            cdf_p = sorted(cdf_p, reverse=False)
            x_values = np.arange(len(cdf_p))/len(cdf_p) * 100
            plt.plot(x_values, cdf_p, line_styles[i % len(line_styles)], label=policy, color=colors[i % len(colors)], linewidth=2)

        plt.legend(ncol=1, bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=8)
        plt.xlabel("CDF (%)")
        plt.ylabel("Average generation per subgraph (MW)")
        plt.tight_layout()
        plt.grid(visible=True, which='major', axis='y', color='#000000', linestyle='--')
        plt.savefig(f"{plot_dir}/cdf_of_power_avg_subgraph.pdf")
        plt.clf()


        for i, (policy, cdf_p) in enumerate(self.power_stable_cdf_results_subgraph[date].items()):
            cdf_p = sorted(cdf_p, reverse=False)
            x_values = np.arange(len(cdf_p))/len(cdf_p) * 100
            plt.plot(x_values, cdf_p, line_styles[i % len(line_styles)], label=policy, color=colors[i % len(colors)], linewidth=2)

        plt.legend(ncol=1, bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=8)
        plt.xlabel("CDF (%)")
        plt.ylabel("Stable generation per subgraph (MW)")
        plt.tight_layout()
        plt.grid(visible=True, which='major', axis='y', color='#000000', linestyle='--')
        plt.savefig(f"{plot_dir}/cdf_of_power_stable_subgraph.pdf")
        plt.clf()




        # sorted_results_match_last_day = list(sorted([result for result in sorted_results[:int(num_subgraphs*factor)] if result[0] in previous_subgraphs], key=sort_func, reverse=True))
        # for result in sorted_results_match_last_day:
        #     if result[3] == 0 or nodes.intersection(set(result[0])):
        #         continue
        #     subgraphs.append(result)
        #     nodes.update(result[0])
        #     total_stable_energy += result[3] * result[2]