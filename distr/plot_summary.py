from plot_detail import *

def plot_timeline_with_bar(results, bar_results, filename, xlabel="Hours", ylabel="Migrate Overhead (hrs)", cumulative=False, step=False, aggregate=False, scaling=1.0, markevery=1):
    if TEXT_ONLY:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), dpi=300, gridspec_kw={'width_ratios': [3, 1.2]})
    ax0, ax1 = axes
    max_y_value = - float('inf')
    plot_func = ax0.plot
    plot_policies = []
    for i, (plot_policy, series) in enumerate(results.items()):
        series = np.array(series) * scaling
        plot_policies.append(plot_policy)
        plot_func(np.arange(len(series)), series, label=plot_policy, color=line_colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)], marker=markers[i % len(markers)], linewidth=2, markevery=markevery)
        max_y_value = max(max_y_value, max(series))

    ax0.legend(ncol=4, fontsize=12, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.24))
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)
    ax0.set_ylim([0, 1.2*max_y_value])
    ax0.locator_params(axis='y', nbins=5)
        
    ax0.grid(visible=True, which='major', axis='y', color='#000000', linestyle='--')
    
    for i, r in enumerate(bar_results):
        ax1.bar(i, r, width=0.5, edgecolor='black', color=colors[i % len(colors)], hatch=hatches[i], zorder=4)
    ax1.grid(visible=True, which='major', axis='y', color='#000000', linestyle='--', zorder=2)
    ax1.set_ylim([0, 1.2*max(bar_results)])
    ax1.set_ylabel("Coefficient of Variation")
    ax1.set_xticks(np.arange(3))
    ax1.set_xticklabels(plot_policies, rotation=30)
        
    fig.tight_layout()
    fig.savefig(f"{filename}")
    plt.clf()


if __name__ == "__main__":
    overhead_types = ["Alloc", "Recompute", "Latency", "Blackout", "Queuing"]
    
    batch_dir=f"raw/default"
    if len(sys.argv) >= 2:
        batch_dir=f"{sys.argv[1]}"
        
    INTERP_FACTOR = 1
    if len(sys.argv) >= 3:
        INTERP_FACTOR=int(sys.argv[2])
        
    TEXT_ONLY = False

    num_total_apps = 0
    num_total_graphs = 0
    TIMESTEPS = None
    date_dirs = glob.glob(f"{batch_dir}/day_*/", recursive = False)
    for date_dir in date_dirs:
        date = int(date_dir.replace("/", " ").strip().split("_")[-1])
        with open(f"{date_dir}/batch_info", "r") as f:
            batch_info = json.load(f)
            num_total_apps += batch_info['num_apps']
            if TIMESTEPS is None:
                TIMESTEPS = batch_info['interval']
            else:
                assert(TIMESTEPS == batch_info['interval'])
            num_total_graphs += batch_info['num_graphs'] 

    input_dir = f"{batch_dir}/output"
    with open(f"{input_dir}/summary", 'rb') as summary_file:
        summary_reports = pickle.load(summary_file)
        summary_reports = pd.DataFrame(summary_reports)
        # summary_reports = summary_reports.loc[['Avg Overhead', '95 Percentile', 'Affected Apps', 'Average overhead of Affected Apps', 'CoV of Power']]
        # summary_reports = summary_reports.loc[['95 Percentile', 'Affected Apps', 'Average overhead of Affected Apps', 'Average migration overhead of Affected Apps']]
        summary_reports = summary_reports.loc[['Avg Overhead', '95 Percentile', 'Affected Apps', 'Average overhead of Affected Apps']]
        
        summary_reports.rename(columns={'cov' : "cov-aware", "avg_min" : "stable", "greedy-fair" : "round-robin"},  inplace=True)
        summary_reports.rename(index = {'Average overhead of Affected Apps' : 'Slowdown of\nAffected Apps', 'Average migration overhead of Affected Apps' : 'Overhead of\nAffected Apps', '95 Percentile' : '95 Percentile\nSlowdown', 'Affected Apps' : 'Affected Apps', 'Avg Overhead' : 'Average\nSlowdown'}, inplace=True)
        print(summary_reports) 
    
    # plot_grouped_bar_subplots(summary_reports, f"{input_dir}/summary.pdf", ylabels=['CoV' "Percentage", 'Slowdown', 'Slowdown', 'Slowdown'], ylims=[[0, None],[0, None],[1, None],[1, None]])
    plot_grouped_bar_subplots(summary_reports, f"{input_dir}/summary.pdf", ylabels=[ "Slowdown", "Slowdown", 'Percentage', 'Slowdown'], ylims=[[1, None], [1, None],[0, None],[1, None]])

    # if "site_identification" in batch_dir:
    with open(f"{input_dir}/all_traces", 'rb') as trace_file:
        all_power_traces = pickle.load(trace_file)
        print(all_power_traces)
        all_power_traces.rename({'random' : "random", 'avg_cov' : "cov-aware", "avg_min" : "stable"},  inplace=True)
        plot_timeline_with_bar(all_power_traces[14]*160 / 320, [0.39, 0.28, 0.22], f"{input_dir}/power_traces.pdf", ylabel="# of Cores")
        
