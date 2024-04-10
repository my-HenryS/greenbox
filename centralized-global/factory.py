from macros import *

class App:
    def __init__(self, name, num_vms, vm_type, init_placement=None, cores_per_vm=4, migration=2, completion_time=100, recomputation=0, latency=0, start=0):
        self.id = -1
        self.name = name
        self.num_vms = num_vms
        self.vm_type = vm_type
        self.cores_per_vm = cores_per_vm
        self.placement = init_placement
        self.completion_time = completion_time
        self.alloc_overhead = migration
        self.latency = latency
        self.recomputation = recomputation
        self.start = start
        self.base_start = 0
    
    @property
    def total_cores(self):
        return self.num_vms * self.cores_per_vm

class AppFactory:
    def __init__(self):
        return
    
    def create_templates(self):            
        app_1 = App(name="cofii-e4s", num_vms=8, vm_type=0, cores_per_vm=8, completion_time=120, migration=2, recomputation=5, latency = 1)
        app_2 = App(name="db", num_vms=3, vm_type=0, cores_per_vm=8, completion_time=120, migration=2, recomputation=0, latency = 0.8)
        app_3 = App(name="dnn", num_vms=1, vm_type=1, cores_per_vm=20, completion_time=180, migration=5, recomputation=5, latency = 1)
        app_4 = App(name="db_large", num_vms=3, vm_type=0, cores_per_vm=8, completion_time=120, migration=20, recomputation=0, latency = 0.8)
        app_5 = App(name="dnn_large", num_vms=1, vm_type=1, cores_per_vm=20, completion_time=180, migration=5, recomputation=50, latency = 1)

        templates = [app_1] + [app_2, app_4] + [app_3, app_5] * 2
        
        # templates = [app_1, app_2, app_3, app_3, app_3, app_3, app_4, app_5]
        # templates = [app_1, app_2, app_3, app_4, app_4, app_4, app_4, app_5]
        # templates = [app_1, app_2, app_3, app_4, app_5, app_5, app_5, app_5]
        # templates = [copy.deepcopy(app_5)] * 5
        # templates = [app_4, app_5]
        # templates = [app_1]
        
        for t in set(templates):
            t.completion_time *= 2 
        return templates

    def create_workloads(self, total_apps, interval, policy="even", distribution = None, template=None, gap=1):

        # we can scale the migration overhead based on the estimated link speed of VB 

        if template is None:
            template = self.create_templates()

        num_per_app = [total_apps // len(template)] * len(template)
            
        apps = []
        _id = 0
        interval = int(interval)
        apps_per_step = []

        if policy == "even":
            for j in range(max(num_per_app)):
                for i in range(len(template)):
                    if j >= num_per_app[i]:
                        break
                    new_app = copy.deepcopy(template[i])
                    apps.append(new_app)
            
            # seed = 0.5
            # random.shuffle(apps, lambda: seed)
            for new_app in apps:
                new_app.id = _id
                new_app.start = (_id // math.ceil(len(apps) / (interval // gap))) * gap
                _id += 1
                # print(new_app.start, new_app.name)
                
        if policy == "custom":
            i = 0
            app_t = 0
            while i < interval:
                apps_per_step += distribution.tolist()
                i += len(distribution)
            apps_per_step = np.array(apps_per_step[:interval], dtype=int)
            total_apps = sum(num_per_app)
            apps_per_step = apps_per_step * total_apps // sum(apps_per_step)
            for t in range(interval):
                for j in range(apps_per_step[t]):
                    new_app = copy.deepcopy(template[app_t])
                    new_app.id = _id
                    new_app.start = t
                    apps.append(new_app)
                    _id += 1
                    app_t += 1
                    app_t %= 3
                    
        if policy == "backlog":
            num_per_app = np.array(num_per_app)
            num_batches = min(num_per_app)
            ratios = num_per_app // num_batches
            
            for b in range(num_batches):
                for i in range(len(template)):
                    for j in range(ratios[i]):
                        new_app = copy.deepcopy(template[i])
                        apps.append(new_app)
                    
            # seed = 0.5
            # random.shuffle(apps, lambda: seed)
            for new_app in apps:
                new_app.id = _id
                new_app.base_start = random.random() #0
                new_app.start = new_app.base_start
                new_app.completion_time *= random.random()+1/2 #random.randint(1,3) / 2
                # print(new_app.completion_time)
                _id += 1
            # print([(app.id, app.name) for app in apps])

        return apps,apps_per_step
    
    def create_workloads_with_energy(self, avg_power, interval, util):
        app_templates = self.create_templates()
        num_cores_per_vm = np.average([app.total_cores for app in app_templates])
        hours_per_vm = np.average([app.completion_time for app in app_templates]) / (60 / self.interp_factor) 
        max_app_per_hour = avg_power / num_cores_per_vm / hours_per_vm
        num_apps = int(max_app_per_hour * interval * util)
        all_apps, apps_per_step = self.create_workloads(num_apps, int(interval), "backlog", gap=self.interp_factor, template=app_templates)
    
        return all_apps
    
    
# encapsulates the static properties of an VM 
class VM:
    # CORE_TO_POWER = 52.124 # 4.95W per core
    MAX_POWER = 310
    MIN_POWER = 112
    TOTAL_CORES = 40
    STATIC_PER_CORE = MIN_POWER / TOTAL_CORES
    DYNAMIC_PER_CORE = (MAX_POWER - MIN_POWER) / TOTAL_CORES
    def __init__(self, name, cores, memory, lifetime, maxcpu=[], avgcpu=[], mincpu=[], pred_lifetime=0, start_time=0, priority=0):
    # def __init__(self, name, cores, memory, lifetime, cpuusage=[], pred_lifetime=0, start_time=0, priority=0):
        self.id = -1
        self.name = name
        self.cores = cores
        self.memory = memory
        self.lifetime = lifetime
        self.profile = None
        self.priority = priority
        self.pred_lifetime = lifetime if pred_lifetime == 0 else pred_lifetime
        self.pred_origin = lifetime if pred_lifetime == 0 else pred_lifetime
        if len(maxcpu) > 0:
            self.maxpower = self.DYNAMIC_PER_CORE * cores * maxcpu / 100 + self.STATIC_PER_CORE * cores
            self.avgpower = self.DYNAMIC_PER_CORE * cores * avgcpu / 100 + self.STATIC_PER_CORE * cores
            self.minpower = self.DYNAMIC_PER_CORE * cores * mincpu / 100 + self.STATIC_PER_CORE * cores
            self.avg_power = np.max(self.maxpower)
        else:
            self.maxpower = []
            self.avgpower = []
            self.minpower = []
            self.avg_power = self.CORE_TO_POWER*self.cores
        
        self.start_time = start_time
    
    def add_profile(self, profile):
        self.profile = profile
        
    def __str__(self):
        return f"(ID: {self.id}, Cores: {self.cores}, Mem: {self.memory}, Priority: {self.priority}, Power: {self.avg_power}, starttime:{self.start_time}, Lifetime: {self.lifetime}, PredLifetime: {self.pred_lifetime}, PredOrigin: {self.pred_origin})\n"
        # return f"(ID: {self.id}, Cores: {self.cores}, Mem: {self.memory}, Power: {self.avg_power}, Lifetime: {self.lifetime}, PredLifetime: {self.pred_lifetime}, PredOrigin: {self.pred_origin}, Power:{self.power})\n"
        # return f"(ID: {self.id}, Cores: {self.cores}, Mem: {self.memory}, Power: {self.avg_power}, Lifetime: {self.lifetime}, PredLifetime: {self.pred_lifetime})\n"
    
    def __repr__(self):
        return str(self)
    
    
# encapsulates the runtime properties of an VM 
class VMProfile():
    def __init__(self, vm, T):
        self.vm = vm
        self.finished = False
        self.init(T)
        self.vm.add_profile(self)

    def init(self, T):
        self.queuing, self.interrupt, self.overhead = 0, 0, 0
        self.exec, self.start, self.end, self.lifetime, self.pred_lifetime = 0, self.vm.start_time, -1, self.vm.lifetime, self.vm.pred_lifetime
        self.migration_log = [0 for _ in range(T)]
        self.per_step_progress = [0 for _ in range(T)]
        
    def has_finished(self):
        return self.vm.lifetime - self.exec <= 1e-9
    
    def lifetime_mispredict(self):
        if self.exec >= self.vm.pred_lifetime and self.exec < self.vm.lifetime:
            return True
        else:
            return False

    def avail(self):
        if self.has_finished():
            return self.lifetime / (self.end - self.start)
        else:
            return self.exec / (self.exec + self.interrupt)
            
    def downtime(self):
        if self.has_finished():
            return self.end - self.start - self.lifetime
        else:
            return self.interrupt
    
    def predict_avail(self, extra_exec=0, extra_overhead=0):
        if self.has_finished():
            return self.lifetime / (self.end - self.start)
        else:
            return (self.exec + extra_exec) / (self.exec + extra_exec + self.interrupt + extra_overhead)
        
    # def avail(self):
    #     if self.has_finished():
    #         return 1 - (self.end - self.start - self.lifetime) / self.lifetime
    #     else:
    #         return 1 - self.interrupt / self.exec
        
    # def predict_avail(self, extra_exec=0, extra_overhead=0):
    #     if self.has_finished():
    #         return 1 - (self.end - self.start - self.lifetime) / self.lifetime
    #     else:
    #         return 1 - (self.interrupt + extra_overhead) / (self.exec + extra_exec)
    
    # def predict_avail(self, extra_exec=0, extra_overhead=0):
    #     if self.has_finished():
    #         return self.lifetime / (self.end - self.start)
    #     else:
    #         return (self.exec + extra_exec) / (self.exec + extra_exec + self.interrupt + extra_overhead)

    def __str__(self):
        return f"Queuing: {self.queuing}, Interrupt: {self.interrupt}, Overhead: {self.overhead}, Exec: {self.exec}, Start: {self.start}, End: {self.end}, Lifetime: {self.lifetime}"

    def __repr__(self):
        return str(self)
    
class VMFactory:
    def __init__(self):
        return
        
    def create_real_azure_workloads(self, lifetime_mis_ratio=0., distribution=0.):
        # dataset_dir = "data_azure/datasetv1/datasetv1.csv"
        dataset_dir = "data_azure/datasetv1/datasetv1_usage_48.csv"
        #dataset_dir = "data_azure/datasetv1/datasetv1_usage.csv"
        dataset = pd.read_csv(dataset_dir)

        dataset = dataset[['vmid', 'lifetime', 'maxcpu', 'avgcpu', 'mincpu', 'vmcategory', 'vmcorecountbucket', 'vmmemorybucket']]
        dataset['maxcpu'] = dataset['maxcpu'].apply(lambda x: np.array(eval(x)))
        dataset['avgcpu'] = dataset['avgcpu'].apply(lambda x: np.array(eval(x)))
        dataset['mincpu'] = dataset['mincpu'].apply(lambda x: np.array(eval(x)))
        dataset['vmmemorybucket'] = dataset['vmmemorybucket'].clip(lower=1)
        # dataset = dataset[['vmid', 'lifetime', 'cpuusage', 'vmcategory', 'vmcorecountbucket', 'vmmemorybucket']]
        # dataset['lifetime'] = dataset['lifetime'].clip(upper=48)
        # dataset['cpuusage'] = dataset['cpuusage'].apply(lambda x: np.array(eval(x)))

        if distribution >= 0.:
            np.random.seed(10)
            length = len(dataset.index)
            distr = np.array(['Delay-insensitive'] * int(length * distribution) + ['Interactive'] * (length - int(length * distribution)))
            np.random.shuffle(distr)
            dataset['vmcategory'] = distr

        vms = []
        for i, vm in dataset.iterrows():
            new_vm = VM(i, cores=vm['vmcorecountbucket'], memory=vm['vmmemorybucket'], lifetime=int(vm['lifetime'])*HOURS, maxcpu=vm['maxcpu'], avgcpu=vm['avgcpu'], mincpu=vm['mincpu'], start_time=1*HOURS, priority=1)
            # new_vm = VM(i, cores=vm['vmcorecountbucket'], memory=vm['vmmemorybucket'], lifetime=int(vm['lifetime'])*HOURS, cpuusage=vm['cpuusage'], start_time=1*HOURS, priority=1)
            new_vm.id = i
            new_vm.start_time = random.random()
            if vm['vmcategory'] == 'Interactive':
                new_vm.priority = 0
            if len(vm['maxcpu']) < 1 or len(vm['avgcpu']) < 1:
                log_msg("cpu power is 0")
                continue

            random_lifetime = max(int(vm['lifetime'] * (1+lifetime_mis_ratio)), 1)
            if random_lifetime > int(vm['lifetime']):
                random_lifetime = np.random.randint(vm['lifetime'], random_lifetime+1)
            elif random_lifetime < int(vm['lifetime']):
                random_lifetime = np.random.randint(random_lifetime, vm['lifetime']+1)
            new_vm.pred_lifetime = random_lifetime * HOURS
            new_vm.pred_origin = random_lifetime * HOURS
            vms.append(new_vm)
        return vms
    
    def create_real_azure_workloads_with_energy(self, running_hours, total_energy, util, lifetime_mis_ratio=0., distribution=0.):
        vm_traces = self.create_real_azure_workloads(lifetime_mis_ratio, distribution)
        remaining_energy = total_energy * util
        
        vms  = []
        # random.shuffle(vm_traces)
        for vm in vm_traces:
            # vm_energy = vm.avg_power
            #TODO: change to interval
            vm_energy = np.mean(vm.avgpower) * min(running_hours, vm.lifetime / HOURS)
            print("VM ENERGY", vm_energy)
            if remaining_energy < vm_energy:
                break
            remaining_energy -= vm_energy
            vms.append(vm)
        return vms

    def create_templates(self):
        VM_A = VM("A", cores=16, memory=128, lifetime=6*HOURS, start_time=1*HOURS, priority=1)
        # VM_A = VM("A", cores=16, memory=128, lifetime=6*HOURS, start_time=1*HOURS, priority=0)
        VM_B = VM("B", cores=16, memory=64, lifetime=4*HOURS, start_time=1*HOURS, priority=0)
        VM_C = VM("C", cores=16, memory=256, lifetime=5*HOURS, start_time=1*HOURS, priority=0)

        misprediction_ratio = 0.0
        random_lifetime = np.random.randint(VM_A.lifetime, VM_A.lifetime * (1+misprediction_ratio))
        VM_A.pred_lifetime = random_lifetime
        random_lifetime = np.random.randint(VM_B.lifetime, VM_B.lifetime * (1+misprediction_ratio))
        VM_B.pred_lifetime = random_lifetime
        random_lifetime = np.random.randint(VM_C.lifetime, VM_C.lifetime * (1+misprediction_ratio))
        VM_C.pred_lifetime = random_lifetime

        # templates = [VM_A, VM_B, VM_C]
        # templates = [VM_A, VM_B] * 4 + [VM_C] * 1
        templates = [VM_A] * 1 + [VM_B, VM_C] * 3
        
        return templates
    
    def create_workloads(self, total_vms, interval, policy="backlog", distribution = None, template=None):

        if template is None:
            template = self.create_templates()

        num_per_vm = [total_vms // len(template)] * len(template)
        vms = []
        _id = 0
        interval = int(interval)

        if policy == "backlog":
            num_per_vm = np.array(num_per_vm)
            num_batches = min(num_per_vm)
            ratios = num_per_vm // num_batches
            
            for b in range(num_batches):
                for i in range(len(template)):
                    for j in range(ratios[i]):
                        new_vm = copy.deepcopy(template[i])
                        vms.append(new_vm)
                    
            # seed = 0.5
            # random.shuffle(apps, lambda: seed)
            for new_vm in vms:
                new_vm.id = _id
                new_vm.start_time = random.random() #0
                # new_vm.lifetime *= random.random()+1/2 
                _id += 1
            # print([(app.id, app.name) for app in apps])

        return vms
        
    def create_workloads_with_energy(self, avg_power, interval, util, lifetime_mis_ratio=0.):
        vm_templates = self.create_templates()
        avg_power_per_vm = np.average([vm.avg_power for vm in vm_templates])
        hours_per_vm = np.average([vm.lifetime for vm in vm_templates]) / interval
        max_app_per_hour = avg_power / avg_power_per_vm / hours_per_vm
        num_vms = int(max_app_per_hour * interval * util)
        all_vms = self.create_workloads(num_vms, int(interval), "backlog", template=vm_templates)
    
        return all_vms
    

    def create_vmtrace(self):
        head = ["lifetime", "priority"]
        vm_dist_file = "data/vm_dist.txt"
        vm_lifetime_file = "data/vm_lifetime.txt"
        f = open(vm_dist_file)
        data = f.read()
        vm_dist = json.loads(data)
        f = open(vm_lifetime_file)
        data = f.read()
        vm_lifetime = json.loads(data)
        
        vms = []
        np.random.seed(10)
        for i in range(100000):
            priority_dist = np.divide([vm_dist["0"], vm_dist["1"]], vm_dist["0"] + vm_dist["1"])
            priority = np.random.choice(np.arange(0, 2), p=priority_dist)
            lifetime = np.random.choice(np.arange(1, 49), p=vm_lifetime[str(priority)])
            new_vm = VM(i, cores=16, memory=256, lifetime=lifetime*HOURS, start_time=1*HOURS, priority=priority)
            new_vm.id = i
            new_vm.start_time = random.random() #0

            misprediction_ratio = 0.2
            random_lifetime = np.random.randint(new_vm.lifetime, new_vm.lifetime * (1+misprediction_ratio))
            # new_vm.pred_lifetime = min(random_lifetime, 48*HOUS)
            new_vm.pred_lifetime = max(random_lifetime, 48)
            new_vm.pred_lifetime = random_lifetime
            vms.append(new_vm)
        return vms

    def create_real_workloads_with_energy(self, avg_power, interval, util):
        vm_traces = self.create_vmtrace()
        # remaining_energy = avg_power * interval * util
        remaining_energy = avg_power * util
        
        vms  = []
        for vm in vm_traces:
            vm_energy = vm.avg_power
            # vm_energy = power_per_vm
            if remaining_energy < vm_energy:
                break
            remaining_energy -= vm_energy
            vms.append(vm)
        return vms

if __name__ == "__main__":
    start = time.time()
    a = VMFactory()
    vms = a.create_real_azure_workloads_with_energy(1500, 60, 90)
    end = time.time()
    print(end - start)
