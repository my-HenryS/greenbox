
import numpy as np
from numpy.lib.function_base import i0, kaiser
import gurobipy as gp
from gurobipy import GRB

import datetime as dt
import pandas as pd
import os
from dateutil import tz
import math
import random
import matplotlib.pyplot as plt
import copy
import sys

class App:
    def __init__(self, name, num_vms, vm_type, init_placement=None, cores_per_vm=4, migration=2, completion_time=100, recomputation=0, latency=0):
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
    
    @property
    def total_cores(self):
        return self.num_vms * self.cores_per_vm


class Policy:
    RANDOM = "random"
    MEMORY = "memory"
    def __init__(self, name, energy, apps, regions):
        self.name = name
        self.energy = energy
        self.apps = apps
        self.regions = regions

        self.t = 0
        self.max_t = len(self.energy[0])
        self.results = {}

    def migrate(self):
        pass

class BATCH_MIP(Policy):
    def __init__(self, *args, step=-1, **kwargs):
        super(BATCH_MIP, self).__init__(*args, **kwargs)
        self.step = step

    def migrate(self, step, init=False, pre_migration=False):
        total_migration = 0
        total_migration_matrix = np.zeros((4,4))

        self._migrate(0, step, self.apps, init, pre_migration)

    def _migrate(self, start_time, step, apps, init=False, pre_migration=False):
        timestep = 15
        log_msg(f"Creating model")
        model = gp.Model("batch")
        step += 1
        T, M, N = step, len(apps), self.regions

        log_msg(T,M,N)
    
        # vars 
        tasks = model.addVars(T, M, N, vtype=GRB.INTEGER)
        data = model.addVars(T, M, N, vtype=GRB.CONTINUOUS, lb=0, ub=1)
        sufficient = model.addVars(T, M, vtype=GRB.BINARY)
        progress = model.addVars(T, M, vtype=GRB.CONTINUOUS, lb=0)
        overhead = model.addVars(T, M, vtype=GRB.INTEGER)
        complete = model.addVars(T, M, vtype=GRB.BINARY)
        powered_core = model.addVars(T, N, vtype=GRB.INTEGER)
        alloc = model.addVars(T, N, vtype=GRB.INTEGER)
        last_iteration_residual = [None for i in range(M)]
        total_complete = model.addVar(vtype=GRB.CONTINUOUS) 
        avg_complete = model.addVar(vtype=GRB.CONTINUOUS) 
        t_complete = [None for i in range(M)]
        running_core = model.addVars(T, N, vtype=GRB.INTEGER) # [[None for n in range(N)] for t in range(T)] 
        total_core = None
        allocation_overhead = None
        vms_to_boot = model.addVars(T, M, N, vtype=GRB.INTEGER, lb=0)
        # aux
        aux0 = model.addVars(T, M, vtype=GRB.CONTINUOUS)
        aux1 = model.addVars(T, N, vtype=GRB.CONTINUOUS)
        aux2 = model.addVars(T, N, vtype=GRB.INTEGER)


        # (1) initial placement 
        if init:
            log_msg(f"initial placement")
            model.addConstrs(tasks[0, i, j] == self.apps[i].placement[j] for j in range(N) for i in range(M))

        # (2) progress and completion
        log_msg(f"sufficient VM constraint")
        model.addConstrs(gp.quicksum([tasks[t, i, j] for j in range(N)])  >= (self.apps[i].num_vms * sufficient[t, i]) for i in range(M) for t in range(T))

        log_msg(f"progress constraint ")
        #model.addConstrs(aux0[t, i] == sufficient[t, i] - overhead[t, i] * self.apps[i].alloc_overhead / timestep for i in range(M) for t in range(T))
        model.addConstrs(progress[t, i] <= sufficient[t, i] - overhead[t, i] * self.apps[i].alloc_overhead / timestep for i in range(M) for t in range(T))
        
        log_msg(f"accumulative execution time constraints")
        model.addConstrs(gp.quicksum(progress[t, i] for t in range(T)) >= self.apps[i].completion_time / timestep for i in range(M))
        model.addConstrs(gp.quicksum(progress[t, i] for t in range(T)) <= self.apps[i].completion_time / timestep + 1 for i in range(M))

        log_msg(f"tasks completion constraints")
        model.addConstrs(gp.quicksum(progress[t0, i] for t0 in set(range(t+1))) >= self.apps[i].completion_time * complete[t, i] / timestep for t in range(T) for i in range(M))
        model.addConstrs(gp.quicksum(complete[t, i] for t in range(T)) == 1 for i in range(M))
        # make sure no progress after complete
        model.addConstrs(gp.quicksum(complete[t0, i] for t0 in set(range(t+1))) <= 1 - progress[t, i] for t in range(T) for i in range(M))
        
        log_msg(f"max job completion constraints")
        t_complete = [gp.quicksum(complete[t, i]*t for t in range(T)) for i in range(M)]
        last_iteration_residual = [gp.quicksum(progress[t, i] for t in range(T)) - self.apps[i].completion_time / timestep for i in range(M)]
        model.addConstrs(t_complete[i] - last_iteration_residual[i] <= total_complete for i in range(M))
        model.addConstr(avg_complete == gp.quicksum(t_complete[i] - last_iteration_residual[i] for i in range(M)))
            
        # (3) running cores (to disable pre-migration, we set running_cores = powered_cores)
        log_msg(f"required running cores") 
        running_core = np.array([[gp.quicksum(tasks[t, i, j] * self.apps[i].cores_per_vm for i in range(M)) for j in range(N)] for t in range(T)])
        model.addConstrs(running_core[t, j] == powered_core[t, j] for j in range(N) for t in range(T))
        
        # (4) powered cores
        log_msg(f"powered cores & vm alloc")    
        for t in range(T):
            for j in range(N):
                model.addConstr(powered_core[t, j] <= self.energy[j, start_time+t])
        
        # (5) overhead
        for t in range(T):
            for j in range(N):
                if t > 0:
                    # model.addConstr(aux2[t, j] == gp.max_(running_core[t, j], powered_core[t-1, j]))
                    model.addConstr(gp.quicksum(vms_to_boot[t, i, j] for i in range(M)) >= running_core[t, j] - powered_core[t-1, j])
                    model.addConstrs(vms_to_boot[t, i, j] <= tasks[t, i, j] for i in range(M))
                else:
                    model.addConstrs(vms_to_boot[t, i, j] == 0 for i in range(M))

            for i in range(M):
                model.addConstr(overhead[t, i] == gp.max_([vms_to_boot[t, i, j] for j in range(N)]))
                model.addConstr(15 >= overhead[t, i] * self.apps[i].alloc_overhead / timestep)



        # (6) objective
        total_core = gp.quicksum(powered_core[t, j] for j in range(N) for t in range(T))

        log_msg(f"Optimizing model")
        # model.setObjective(total_complete*timestep + allocation_overhead + total_core / 50, GRB.MINIMIZE)  
        # model.setObjective(total_complete + total_core, GRB.MINIMIZE)  
        # model.setObjective(total_complete, GRB.MINIMIZE)  
        # model.setObjective(avg_complete / M, GRB.MINIMIZE)  
        # model.setObjective(total_core, GRB.MINIMIZE)  
        model.setObjectiveN(total_complete, 0, 1)
        model.setObjectiveN(total_core, 1, 0)
        # model.Params.TIME_LIMIT = 300.0
        # model.Params.MIPGap = 0.01
        #model.Params.Presolve = 0
        data = data
        model.Params.MIPFocus=3
        model.Params.Cuts=0
        model.optimize()
        model.printQuality()

        ## results
        log_msg(f"Objective: {model.objVal}")
        log_msg(f"Total cores: {total_core.getValue()}")
        log_msg(f"Completion time: {total_complete.x*timestep}")

        np.set_printoptions(precision=2)
        for t in range(0, int(total_complete.x) + 1):
            log_msg(f"Timestamp {t}")
            log_msg(f"Progress {np.array([progress[t, i].x for i in range(M)], dtype=float)}")
            all_tasks = np.array([[tasks[t, i, j].x for i in range(M)] for j in range(N)], dtype=float)
            log_msg(f"Tasks {np.array2string(all_tasks, prefix='[%s] Tasks ' % (dt.datetime.now()))}")
            log_msg(f"Running Cores {np.array([running_core[t, j].getValue() for j in range(N)], dtype=float)}")
            log_msg(f"Powered Cores {np.array([powered_core[t, j].x for j in range(N)], dtype=float)}")
            log_msg(f"VMs to Boot {sum(np.array([vms_to_boot[t, i, j].x for j in range(N) for i in range(M)], dtype=float))}")
            # all_vms_to_boot = np.array([[vms_to_boot[t, i, j].x for i in range(M)] for j in range(N)], dtype=float)
            # log_msg(f"VMs to Boot {np.array2string(all_vms_to_boot, prefix='[%s] VMs to Boot ' % (dt.datetime.now()))}")
        
        # for i in range(M):
        #     #log_msg([progress[t, i].x for t in range(T)])
        #     #log_msg(gp.quicksum(progress[t, i] for t in range(T)).getValue())
        #     log_msg(f"Complete time of App {i}: {t_complete[i].getValue() - last_iteration_residual[i].getValue()}")
        #     #log_msg(f"Complete time of App {i}: {t_complete[i].getValue(), last_iteration_residual[i].getValue()}")
        return 



class GREEDY(Policy):
    def __init__(self, *args, step=-1, **kwargs):
        super(GREEDY, self).__init__(*args, **kwargs)
        self.step = step

    def migrate(self, step, init=False):
        MAX_STEP = 200
        TIMESTEP = 15
        MIN_ENERGY = min([app.cores_per_vm for app in self.apps])
        T, M, N = MAX_STEP, len(self.apps), self.regions
        running_vms = np.zeros((T, M, N))
        power_per_vm = np.array([app.cores_per_vm for app in self.apps])
        vm_to_progress = np.array([app.num_vms for app in self.apps])
        progress = np.array([app.completion_time for app in self.apps], dtype=np.float64)

        completed_vm = []

        for t in range(MAX_STEP):
            if t == 0:
                running_vms[0, :, :] = np.zeros((M, N))
            else:
                running_vms[t, :, :] = running_vms[t-1, :, :]
                for i in completed_vm:
                    running_vms[t, i, :] = np.zeros((N))

            power = np.matmul(power_per_vm, running_vms[t, :, :])

            # remove redundant
            for j in range(N):
                if power[j] > self.energy[j,t]:
                    over = power[j] - self.energy[j,t]
                    dv = [[i, running_vms[t, i, j], power_per_vm[i]] for i in range(M) if running_vms[t, i, j] > 0]
                    dv = sorted(dv, key=lambda x: x[1]*x[2], reverse=True)
                    while over > 0:
                        i, vms, unit_power = dv.pop(0)
                        evict_vms = min(vms, int(math.ceil(over / unit_power)))
                        running_vms[t, i, j] -= evict_vms
                        over -= evict_vms*unit_power

            # greedily fill rest of energy
            required_vm_to_progress = vm_to_progress - np.sum(running_vms[t, :, :], axis = 1)
            dv = [[i, required_vm_to_progress[i], power_per_vm[i]] for i in range(M) if required_vm_to_progress[i] > 0 and i not in completed_vm]
            dv = sorted(dv, key=lambda x: x[1]*x[2], reverse=True)

            for j in range(N):
                if power[j] < self.energy[j,t]:
                    rest = self.energy[j,t] - power[j]
                    while rest > 0 and len(dv) > 0:
                        i, vms, unit_power = dv[0]
                        fill_vms = min(vms, int(math.floor(rest / unit_power)))
                        if fill_vms == 0:
                            break
                        if fill_vms == vms:
                            dv.pop(0)
                        else:
                            dv[0][1] = vms - fill_vms
                        running_vms[t, i, j] += fill_vms
                        rest -= fill_vms*unit_power

            can_progress = np.array([1 - min(1, x) for x in vm_to_progress - np.sum(running_vms[t, :, :], axis = 1)])
            alloc_overhead = np.zeros(M)
            recompute_overhead = np.zeros(M)
            if t > 0:
                diff_vms = running_vms[t, :, :] - running_vms[t-1, :, :]
                alloc_overhead = np.array([max(diff_vms[i,j] for j in range(N)) * self.apps[i].alloc_overhead for i in range(M)])
                recompute_overhead = np.array([min(1,max(diff_vms[i,j] for j in range(N))) * self.apps[i].recomputation for i in range(M)])

            progress -= can_progress * TIMESTEP - alloc_overhead - recompute_overhead
            power = np.matmul(power_per_vm, running_vms[t, :, :])

            for j in range(N):
                assert(power[j] <= self.energy[j, t])

            completed_vm += [i for i in range(M) if progress[i] <= 0 and i not in completed_vm]
                        
            # log_msg(self.energy[:, t])
            # log_msg(power)
            # log_msg(can_progress)
            # log_msg(t, progress)
            # log_msg(completed_vm)

            if len(completed_vm) == len(self.apps):
                break

        log_msg("Greedy Completion Time", t, t*TIMESTEP)
        return t 

                        
            





def pow_to_cores(new_power, max_power, max_cores):
    return min(math.floor(max_cores * new_power / max_power), max_cores)

def plot_energy(energy, label):
    fig, ax = plt.subplots()
    plt.plot(energy['Time'], energy['Value'])
    ax.set_xlabel('Datetime')
    ax.set_ylabel('Power (MW)')
    fig.autofmt_xdate()
    plt.grid()
    fig.savefig(f'figs/wind-over-time-{label}.pdf')

def use_simple_example():
    energy = np.array([[2,2,2,1,1,1,1,1,1], [3,3,3,3,3,3,3,3,3]])
    app = App(name="cofii-e4s", num_vms=4, init_placement=[2,2], cores_per_vm=1, completion_time=15*6, migration=1, recomputation=0)
    apps = [app]
    return energy, apps, 2, 8, True

def create_workloads(num_per_app):
    app_1 = App(name="cofii-e4s", num_vms=16, vm_type=0, cores_per_vm=4, completion_time=90, migration=2, recomputation=0)
    # obtain profiling numbers for flux
    app_2 = App(name="db", num_vms=3, vm_type=0, cores_per_vm=4, completion_time=90, migration=2, recomputation=0)
    app_3 = App(name="large-dnn-nc6", num_vms=1, vm_type=1, cores_per_vm=6, completion_time=150, migration=7, recomputation=3)

    template = [app_1, app_2, app_3]
    apps = []

    _id = 0
    for i in range(3):
        for j in range(num_per_app[i]):
            new_app = copy.deepcopy(template[i])
            new_app.id = _id
            apps.append(new_app)
            _id += 1
    return apps
        
    


if __name__ == "__main__":
    # global vars
    DATA_PATH = '/home/jhsun/software/virtual-battery/data'
    RENEWABLE_PATH = os.path.join(DATA_PATH, 'renewable/')
    SIMULATION_OUTPUT_PATH = os.path.join(DATA_PATH, 'simulator-validation/simulation_output')
    EPS = 1e-6
    sites = 4
    max_power = 2624.0        
    init = False

    if len(sys.argv) > 2:
        policy = sys.argv[1]
        scale = int(sys.argv[2])
    else:
        policy = "mip"
        scale = 5

    log_msg(policy, scale)

    max_cores = scale*15
    step = scale*10
    num_per_app = [scale]*3

    # energy traces
    renewable_trace = pd.read_csv(os.path.join(RENEWABLE_PATH, 'smartgrid_wind_20191201-20191231.csv'), parse_dates=['EffectiveTime'])
    renewable_trace['Time'] = pd.to_datetime(renewable_trace['EffectiveTime'].dt.floor('s'), utc=True)
    renewable_trace = renewable_trace.sort_values(by=['Time'])
    start_date = dt.datetime(2019, 12, 1).replace(tzinfo=tz.tzutc())
    powers = []
    for i in range(sites):
        start = start_date + dt.timedelta(days=i*7)
        end = start + dt.timedelta(days=7)
        trace = renewable_trace[(end > renewable_trace['Time']) & (renewable_trace['Time'] >= start)].copy()
        trace['Time'] = trace['Time'].apply(lambda time : time - dt.timedelta(days=i*7))
        trace['Value'] = trace['Value'].apply(lambda value : pow_to_cores(value, max_power, max_cores))
        powers.append(trace['Value'].to_numpy())
        plot_energy(trace, i)

    energy = np.vstack(powers)
    energy = energy.astype(int)
    energy = energy[0:sites,200:]
    # energy = np.repeat(energy, 15, axis=1)
    # with np.printoptions(threshold=np.inf):    
    #     log_msg(energy)

    # log_msg(np.sum(energy,axis=0))
    # init_energy = energy[:,0].copy()
    # min_cores = np.sum(energy,axis=0).min()
    # init_energy = init_energy*min_cores//sum(init_energy)

    # the time unit is one minute
    apps = create_workloads(num_per_app)

    # energy, apps, sites, step, init = use_simple_example()

    if policy == "mip":
        policy = BATCH_MIP("batch-mip", energy, apps, sites)
        policy.migrate(step=step, init=init, pre_migration=False)
    elif policy == "pre":
        policy = BATCH_MIP("batch-mip-pre", energy, apps, sites)
        policy.migrate(step=step, init=init, pre_migration=True)

    policy = GREEDY("greedy", energy, apps, sites)
    policy.migrate(step=step, init=init)

 



