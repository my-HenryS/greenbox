from macros import *

random.seed(10)
VM_COUNT = 5000
MAX_TIME = 48
dataset_dir = "data_azure/datasetv1/datasetv1.csv"
dataset = pd.read_csv(dataset_dir)
dataset = dataset[['vmid', 'lifetime', 'vmcategory', 'vmcorecountbucket', 'vmmemorybucket']]
dataset['lifetime'] = dataset['lifetime'].clip(upper=MAX_TIME)
sampled_dataset = dataset.sample(VM_COUNT, random_state=1)


dir = "/mnt/nvme0n1p1/zibog2/azure_datasetv1"
cpu_usage_file = f"{dir}/cpu_usage_{VM_COUNT}.csv" # 48h 
if os.path.isfile(cpu_usage_file):
    cpu_usage = pd.read_csv(cpu_usage_file)
else:
    headers = ["timestamp", "vmid", "mincpu", "maxcpu", "avgcpu"]
    usage_list = []
    for i in range(1, 126):
        cpu_usage_dir = f"{dir}/vm_cpu_readings-file-{i}-of-125.csv"
        tmp_usage = pd.read_csv(cpu_usage_dir, header=None, index_col=False,names=headers,delimiter=',')
        tmp_usage = tmp_usage[tmp_usage['vmid'].isin(sampled_dataset['vmid'])]
        usage_list.append(tmp_usage)
    cpu_usage = pd.concat(usage_list, axis=0)
    cpu_usage.to_csv(cpu_usage_file)
print("finished loading the usage")


maxs = []
avgs = []
mins = []
for i, vm in sampled_dataset.iterrows():
    start = time.time()
    id = vm['vmid']
    usage = cpu_usage[cpu_usage['vmid'] == id]
    usage['timestamp'] = usage['timestamp'].transform(lambda x: x - x.min())
    tmp_maxs = []
    tmp_avgs = []
    tmp_mins = []
    for i in range(3600, 3600 * 96 + 1, 3600):
        tmp_usage = usage[usage['timestamp'] <= i]
        tmp_maxs.append(tmp_usage['maxcpu'].mean())
        tmp_avgs.append(tmp_usage['avgcpu'].mean())
        tmp_mins.append(tmp_usage['mincpu'].mean())
    maxs.append(tmp_maxs)
    avgs.append(tmp_avgs)
    mins.append(tmp_mins)
    end = time.time()
    print("time", end-start)
dataset_usage = sampled_dataset.copy()
dataset_usage['maxcpu'] = maxs
dataset_usage['avgcpu'] = avgs
dataset_usage['mincpu'] = mins
dataset_usage.to_csv(f"data_azure/datasetv1/datasetv1_usage_{MAX_TIME}.csv")