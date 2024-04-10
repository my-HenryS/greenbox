import numpy as np
from numpy.lib.function_base import i0, kaiser

import datetime as dt
import pandas as pd
import os
from dateutil import tz
import math
import random
import matplotlib.pyplot as plt
import copy
import sys

from macros import *
from factory import *

GRB_ENV.setParam("OutputFlag",0)
GRB_ENV.start()

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
    # OPT_TIME = 60.0
    # OPT_GOAL = 0.002
    
    def __init__(self):
        # super(BATCH_MIP, self).__init__(*args, **kwargs)
        # self.step = step
        pass

    @staticmethod
    def migrate(start_time, step, timestep, energy, predicted_energy, apps, regions, app_profiles, OPT_TIME, OPT_GOAL, init_vms=None, init_progress=None, completed_vm = [], pre_migration=False, hint=None, policy="mip", subgraph_id = 0, blocking_time = None):
    
        # log_msg(f"Creating model")
        model = gp.Model("batch", env=GRB_ENV)
        if policy == "mip": 
            logfilename = f"logs/mip-model-{subgraph_id}.log"
        elif policy == "mip-app": 
            logfilename = f"logs/mip-app-model-{subgraph_id}.log"
        if start_time == 0:
            logfile = open(logfilename, "w").close()
        model.Params.LogToConsole = 0
        model.Params.OutputFlag = 1
        model.Params.LogFile = logfilename
        
        T, M, N, K = step, len(apps), regions, 3

        # vars 
        vms = model.addVars(T, M, N, vtype=GRB.INTEGER, lb=0)
        powered_vms = model.addVars(T, M, N, vtype=GRB.INTEGER, lb=0)
        sufficient = model.addVars(T, M, vtype=GRB.BINARY)
        progress = model.addVars(T, M, vtype=GRB.CONTINUOUS)
        complete = model.addVars(T, M, vtype=GRB.BINARY)
        last_iteration_residual = [None for i in range(M)]
        total_complete = model.addVar(vtype=GRB.CONTINUOUS) 
        avg_complete = model.addVar(vtype=GRB.CONTINUOUS) 
        t_complete = [None for i in range(M)]
        powered_core = None
        running_core = None # [[None for n in range(N)] for t in range(T)] 
        total_core = None
        avg_progress = None
        overhead = model.addVars(T, M, vtype=GRB.CONTINUOUS)
        real_overhead = model.addVars(T, M, vtype=GRB.CONTINUOUS)
        allocation_overhead = model.addVars(T, M, vtype=GRB.CONTINUOUS)
        recomputation_overhead = model.addVars(T, M, vtype=GRB.CONTINUOUS)
        latency_multiplier = model.addVars(T, M, vtype=GRB.CONTINUOUS)
        latency_overhead = model.addVars(T, M, vtype=GRB.CONTINUOUS)
        vms_to_boot = model.addVars(T, M, N, vtype=GRB.INTEGER)
        vms_to_boot_by_type = model.addVars(T, K, N, vtype=GRB.INTEGER)
        # aux
        aux_boot = model.addVars(T, M, N, vtype=GRB.CONTINUOUS)
        aux0 = model.addVars(T, M, vtype=GRB.INTEGER)
        aux1 = model.addVars(T, M, vtype=GRB.INTEGER)
        aux2 = model.addVars(T, M, vtype=GRB.INTEGER)
        aux3 = model.addVars(T, M, vtype=GRB.INTEGER)
        aux4 = model.addVars(T, M, vtype=GRB.INTEGER)
        aux5 = model.addVars(T, M, vtype=GRB.BINARY)
        aux6 = model.addVars(T, M, vtype=GRB.CONTINUOUS)
        aux7 = model.addVars(T, M, vtype=GRB.INTEGER)
        has_progress = model.addVars(T, M, vtype=GRB.BINARY)
        aux8 = model.addVars(T, M, vtype=GRB.BINARY)
        aux9 = model.addVars(T, M, vtype=GRB.BINARY)
        # for boolean cast
        CONST_A = 100
        CONST_B = 0.001

            
        if hint is not None:
            # T_H, M_H, N_H = hint.shape
            # for t in range(1):
            #     print("HINT_VM", np.array([[hint[t, i, j] for i in range(M)] for j in range (N)]))
            if len(hint.shape) == 3:
                T_H, M_H, N_H = hint.shape
                for t in range(T_H):
                    for i in range(M_H):
                        for j in range(N_H):
                            vms[t, i, j].start = int(hint[t, i, j])
                            
            elif len(hint.shape) == 2:
                M_H, N_H = hint.shape
                for i in range(M_H):
                    for j in range(N_H):
                        vms[0, i, j].start = int(hint[i, j])
                    
            model.update()

        # (1) initial placement 
        if init_vms is not None:
            # log_msg(f"initial placement")
            # we do not place init vms into powered vms variable
            # model.addConstrs(powered_vms[0, i, j] == init_vms[i, j] for j in range(N) for i in range(M))
            #for j in range(N):
            #   assert(sum([init_vms[i, j] for i in range(M)]) <= energy[j, start_time])
            # print(energy.shape, start_time, start_time+10, energy[:, start_time:start_time+10], init_vms, [app.cores_per_vm for app in apps])
            init_progress = np.array(init_progress)
            # for _ in init_progress:
            #     if _ != 0 and _ <= CONST_B:
            #         print(_)
            init_progress[init_progress <= CONST_B] = 0
            pass
        else:
            init_progress = np.zeros(M)

        # (2) progress and completion
        # log_msg(f"sufficient VM constraint")
        model.addConstrs(gp.quicksum([vms[t, i, j] for j in range(N)])  >= (apps[i].num_vms * sufficient[t, i]) for i in range(M) for t in range(T))
        
        # model.addConstrs(sufficient[0, i] == 1 for i in range(M) if init_progress[i] == 0)

        # model.addConstrs(gp.quicksum([powered_vms[t, i, j] for j in range(N)])  <= apps[i].num_vms for i in range(M) for t in range(T))
    

        # log_msg(f"progress constraint")
        #model.addConstrs(aux0[t, i] == sufficient[t, i] - overhead[t, i] * self.apps[i].alloc_overhead / timestep for i in range(M) for t in range(T))
        
        # log_msg(f"accumulative execution time constraints")
        # model.addConstrs(gp.quicksum(progress[t, i] for t in range(T)) >= apps[i].completion_time / timestep for i in range(M))
        model.addConstrs(init_progress[i] * timestep + gp.quicksum(progress[t, i] for t in range(T)) * timestep <= apps[i].completion_time for i in range(M) if i not in completed_vm)

        # t = timestamp, i = application_index, epoch_time = time for each checkpoint epoch
        # timestep = 15 mins
        # lookahead time; decision granularity; trace granularity;
        # A - N*B >=0
        # which matters more? (1) num of migration or (2) recomputation or data migration time

        # log_msg(f"tasks completion constraints")
        model.addConstrs(init_progress[i] * timestep + gp.quicksum(progress[t0, i] for t0 in set(range(t+1))) * timestep >= apps[i].completion_time * complete[t, i] for t in range(T) for i in range(M) if i not in completed_vm)
        model.addConstrs(gp.quicksum(complete[t, i] for t in range(T)) <= 1 for i in range(M) if i not in completed_vm)
        # make sure no progress after complete
        # model.addConstrs(gp.quicksum(complete[t0, i] for t0 in set(range(t+1))) <= 1 - progress[t, i] for t in range(T) for i in range(M))
        
        # log_msg(f"max job completion constraints")
        # non_completion_penalty = [apps[i].completion_time / timestep - init_progress[i] - gp.quicksum(progress[t, i] for t in range(T)) for i in range(M)]
        t_complete = [(1-gp.quicksum(complete[t, i] for t in range(T)))*(T+10)+gp.quicksum(complete[t, i]*(t+1) for t in range(T)) for i in range(M)]
        # model.addConstrs(t_complete[i] <= total_complete for i in range(M))
        # last_iteration_residual = [init_progress[i] + gp.quicksum(progress[t, i] for t in range(T)) - apps[i].completion_time / timestep for i in range(M)]
        last_iteration_residual = [gp.quicksum((progress[t, i]+real_overhead[t, i])*complete[t, i] for t in range(T)) for i in range(M)]
        #model.addConstrs((1-progress[t, i])*complete[t, i] <= 1-1e-4 for i in range(M) for t in range(T))
        # model.addConstrs(last_iteration_residual[i] >= 0.00001 for i in range(M))
        # model.addConstrs(t_complete[i] - last_iteration_residual[i] <= total_complete for i in range(M))
        model.addConstr(avg_complete == gp.quicksum(t_complete[i] - 1 + last_iteration_residual[i] for i in range(M) if i not in completed_vm))
            
        # (3) running cores (to disable pre-migration, we set running_cores = powered_cores)
        # log_msg(f"required running cores") 
        running_core = np.array([[gp.quicksum(vms[t, i, j] * apps[i].cores_per_vm for i in range(M)) for j in range(N)] for t in range(T)])
        powered_core = np.array([[gp.quicksum(powered_vms[t, i, j] * apps[i].cores_per_vm for i in range(M)) for j in range(N)] for t in range(T)])

        for i in range(M):
            if i in completed_vm:
                model.addConstrs(powered_vms[t, i, j] == 0 for j in range(N) for t in range(1,T))

        if pre_migration:
            model.addConstrs(vms[t, i, j] <= powered_vms[t, i, j] for i in range(M) for j in range(N) for t in range(T))
        else:
            model.addConstrs(vms[t, i, j] == powered_vms[t, i, j] for i in range(M) for j in range(N) for t in range(T))
        
        # (4) powered cores
        # log_msg(f"powered cores & vm alloc")    
        for t in range(T):
            for j in range(N):
                # model.addConstr(powered_core[t, j] <= energy[j, start_time+t])
                model.addConstr(powered_core[t, j] <= predicted_energy[j, start_time+t])
                # if policy == "mip":
                #     model.addConstr(powered_core[t, j] <= energy[j, start_time+t])
                # elif policy == "mip-app":
                #     model.addConstr(powered_core[t, j] <= energy[j, start_time])
        
        # (5) overhead
        for t in range(T):
            for j in range(N):
                if False: #init_vms is not None:
                    model.addConstrs(aux_boot[t, i, j] == powered_vms[t, i, j] - init_vms[i, j] for i in range(M))
                    model.addConstrs(vms_to_boot[t, i, j] == gp.max_(0, aux_boot[t, i, j]) for i in range(M))
                    # model.addConstrs(vms_to_boot[t, i, j] >= powered_vms[t, i, j] - powered_vms[t-1, i, j] for i in range(M))
                    # model.addConstrs(vms_to_boot[t, i, j] >= 0 for i in range(M))
                else:
                    if t == 0:
                        # model.addConstrs(vms_to_boot_by_type[t, k, j] >= 0 for k in range(K))
                        if init_vms is not None:
                            model.addConstrs(vms_to_boot[t, i, j] >= powered_vms[t, i, j] - init_vms[i, j] for i in range(M))
                            model.addConstrs(vms_to_boot[t, i, j] >= 0 for i in range(M))
                        else:
                            model.addConstrs(vms_to_boot[t, i, j] == 0 for i in range(M))
                    else:
                        # for k in range(K):
                        #     model.addConstr(aux2[t, k] == gp.quicksum(vms[t, i, j] for i in range(M) if apps[i].vm_type == k) - gp.quicksum(powered_vms[t-1, i, j] for i in range(M) if apps[i].vm_type == k))

                        # model.addConstrs(vms_to_boot_by_type[t, k, j] >= aux2[t, k] for k in range(K))
                        # model.addConstrs(vms_to_boot_by_type[t, k, j] >= 0 for k in range(K))
                        # model.addConstrs(gp.quicksum(vms_to_boot[t, i, j] for i in range(M) if apps[i].vm_type == k) == vms_to_boot_by_type[t, k, j] for k in range(K))
                        for i in range(M):
                            # model.addConstr(aux_boot[t, i, j] == powered_vms[t, i, j] - powered_vms[t-1, i, j])
                            # model.addConstr(vms_to_boot[t, i, j] == gp.max_(0, aux_boot[t, i, j]))
                            model.addConstr(vms_to_boot[t, i, j] >= powered_vms[t, i, j] - powered_vms[t-1, i, j])
                            model.addConstr(vms_to_boot[t, i, j] >= 0)
                        # model.addConstrs(vms_to_boot[t, i, j] >= 0 for i in range(M))
                    

            for i in range(M):
                model.addConstr(aux0[t, i] == gp.max_([vms_to_boot[t, i, j] for j in range(N)]))
                model.addConstr(aux1[t, i] == gp.min_(1, aux0[t, i]))
                model.addConstr(aux2[t, i] == gp.max_([powered_vms[t, i, j] for j in range(N)]))
                model.addConstr(aux3[t, i] == gp.max_([vms[t, i, j] for j in range(N)]))
                model.addConstr(aux2[t, i] <= apps[i].num_vms)
                
                model.addConstr(aux4[t, i] == gp.max_(apps[i].num_vms-1, aux3[t, i]))
                model.addConstr(aux5[t, i] == apps[i].num_vms - aux4[t, i])

                ## https://cs.stackexchange.com/questions/51025/cast-to-boolean-for-integer-linear-programming
                # IF 100 >= sum(progress) >= 0.001 THEN has_progress = 1 and aux8 = 0
                # IF -0.001 >= sum(progress) > -100.001 THEN has_progress = 0 and aux8 = 1
                # IF sum(progress) == 0 THEN has_progress = 0 and aux8 = 0
                # CONST_A = 100
                # CONST_B = 0.0001
                model.addConstr(init_progress[i] + gp.quicksum(progress[t0, i] for t0 in set(range(t))) <= CONST_A * has_progress[t, i])
                model.addConstr(init_progress[i] + gp.quicksum(progress[t0, i] for t0 in set(range(t))) <= -CONST_B * has_progress[t, i] + (CONST_A + CONST_B)*(1-aux8[t, i]))
                model.addConstr(init_progress[i] + gp.quicksum(progress[t0, i] for t0 in set(range(t))) >= CONST_B * has_progress[t, i] - (CONST_A + CONST_B)*aux8[t, i])
                # model.addConstr(has_progress[t, i] == 1)
                                
                
                # model.addConstr(recomputation_overhead[t, i] * timestep * 2 == has_progress[t, i] * aux1[t, i] * apps[i].recomputation)
                model.addConstr(allocation_overhead[t, i] * timestep == has_progress[t, i] * aux0[t, i] * apps[i].alloc_overhead)
                
                # alternative calculation of recompute
                model.addConstr(aux6[t, i] == (init_progress[i] + gp.quicksum(progress[t0, i] for t0 in set(range(t)))) - aux7[t, i] * (apps[i].recomputation / timestep))
                model.addConstr(aux6[t, i] >= 0)
                # no need to use has start here; if the total progress is zero then aux6 is also zero
                # model.addConstr(aux9[t, i] == has_progress[t, i] * aux1[t, i])
                # model.addConstr(recomputation_overhead[t, i] * timestep == aux9[t, i] * aux6[t, i] * apps[i].recomputation)
                model.addConstr(recomputation_overhead[t, i] * timestep == aux1[t, i] * aux6[t, i] * apps[i].recomputation)
                
                if t == 0 and blocking_time is not None:
                    model.addConstr(real_overhead[t, i] == blocking_time[i] / timestep + allocation_overhead[t, i])
                else:
                    model.addConstr(real_overhead[t, i] == allocation_overhead[t, i])
                # model.addConstr(real_overhead[t, i] <= 1)
                model.addConstr(overhead[t, i] == gp.min_(real_overhead[t, i], sufficient[t, i]))
                model.addConstr(latency_multiplier[t, i] == (1 - aux5[t, i]) + aux5[t, i] * apps[i].latency)
                model.addConstr(latency_overhead[t, i] >= (1 - latency_multiplier[t, i]) * (sufficient[t, i] - overhead[t, i]))
                model.addConstr(latency_overhead[t, i] >= 0)
                model.addConstr(progress[t, i] <= sufficient[t, i] - recomputation_overhead[t, i] - overhead[t, i] - latency_overhead[t, i])



        # (6) objective
        total_core = gp.quicksum(powered_core[t, j] for j in range(N) for t in range(T))
        avg_progress = gp.quicksum(progress[t, i] for t in range(T) for i in range(M)  if i not in completed_vm)
        avg_overhead = gp.quicksum(recomputation_overhead[t, i] + overhead[t, i] + latency_overhead[t, i] for t in range(T) for i in range(M)  if i not in completed_vm)
        all_complete = gp.quicksum(complete[t, i] for t in range(T) for i in range(M))
        next_complete = gp.quicksum(complete[t, i] for t in range(T) for i in range(M) if i not in completed_vm)
        
        
        max_cumulative_blackout = model.addVar(vtype=GRB.CONTINUOUS) 
        blackouts = model.addVars(M, vtype=GRB.CONTINUOUS) 
        total_blackout = model.addVar(vtype=GRB.CONTINUOUS) 
        model.addConstr(total_blackout == gp.quicksum(has_progress[t, i]*(1-sufficient[t, i]) for t in range(T) for i in range(M) if i not in completed_vm and app_profiles[i].blackout >= 2))
        model.addConstrs(blackouts[i] == gp.quicksum(has_progress[t, i]*(1-sufficient[t, i]) for t in range(T)) + app_profiles[i].blackout for i in range(M))
        # model.addConstr(max_cumulative_blackout == gp.max_([(gp.quicksum(blackouts[t, i]for t in range(T)) + app_profiles[i].blackout) for i in range(M) if i not in completed_vm]))
        if len(completed_vm) < M:
            model.addConstr(max_cumulative_blackout == gp.max_([blackouts[i] for i in range(M) if i not in completed_vm]))
        else:
            model.addConstr(max_cumulative_blackout == 0)
        

        # log_msg(f"Optimizing model")
        # model.setObjective(total_complete*timestep + allocation_overhead + total_core / 50, GRB.MINIMIZE)  
        # model.setObjective(total_complete + total_core, GRB.MINIMIZE)  
        if policy == "mip-app": 
            # model.setObjectiveN(max_cumulative_blackout, 0, priority=2, weight=1.0) 
            # model.setObjectiveN(avg_progress - total_blackout, 1, priority=1, weight=-1.0) 
            model.setObjective(avg_progress - total_blackout - max_cumulative_blackout, GRB.MAXIMIZE) 
        elif policy == "mip": 
            model.setObjective(avg_complete / M - np.average([apps[i].completion_time / timestep - init_progress[i] for i in range(M)]), GRB.MINIMIZE) 
            # model.setObjective(avg_complete / M, GRB.MINIMIZE) 
        
        # model.setObjective(avg_complete / M, GRB.MINIMIZE)  
        # model.setObjective(avg_progress, GRB.MAXIMIZE)  
        # model.setObjective(next_complete, GRB.MAXIMIZE)  
        # model.setObjective(total_core, GRB.MINIMIZE) 
        # model.ModelSense = GRB.MAXIMIZE 
        # model.setObjectiveN(avg_complete / M, 1, 0)
        # model.setObjectiveN(avg_complete / M, 1, 0)
        # model.setObjectiveN(avg_progress, 0, 1)
        model.Params.TIME_LIMIT = OPT_TIME
        model.Params.MIPGap = OPT_GOAL
        # model.Params.Threads = 16
        # model.Params.NonConvex= 2
        #model.Params.Presolve = 0
        model.Params.MIPFocus=2
        # model.Params.NodefileStart = 10
        # model.Params.Cuts=0
        model.optimize()

        ## results
        #log_msg(f"Objective: {model.objVal}")
        #log_msg(f"Total cores: {total_core.getValue()}")
        #log_msg(f"Completion time: {total_complete.x*timestep}")

        # log_msg(avg_progress.getValue(), total_blackout.x, max_cumulative_blackout.x) 
        # try:
        #     model.printQuality()
        # except gp.GurobiError:
        #     return None

        if start_time == 0:
            mode = "w+"
        else:
            mode = "a+"

        tp = 1
        print_vm = False
        # if subgraph_id == 21 and start_time == 0:
        #     tp = T
        #     print_vm = True
        if policy == "mip":
            logfile = f"logs/mip_{subgraph_id}.log"
        elif policy == "mip-app":
            logfile = f"logs/mip_app_{subgraph_id}.log"
        with open(logfile, mode) as log_f:
            for t in range(tp):
                print("MIP_G: ", subgraph_id, file=log_f)
                print("MIP_T: ", start_time+t, file=log_f)
                try:
                    progress[t, i].x
                except:
                    log_msg("Subgraph", subgraph_id, "did not finish")
                    np.set_printoptions(precision=6)
                    print(init_progress)
                    print(blocking_time)
                    print(init_vms)
                    exit()
                print("MIP_P: ",np.array([progress[t, i].x  for i in range(M)]), file=log_f)
                print("HAS_P: ",np.array([has_progress[t, i].x  for i in range(M)]), file=log_f)
                # print(np.array([progress[t, i].x  for i in range(M)]))
                print("MIP_O: ",np.array([overhead[t, i].x  for i in range(M)]), file=log_f)
                print("MIP_A: ",np.array([allocation_overhead[t, i].x  for i in range(M)]), file=log_f)
                print("MIP_R: ",np.array([recomputation_overhead[t, i].x  for i in range(M)]), file=log_f)
                print("MIP_LM: ",np.array([latency_multiplier[t, i].x  for i in range(M)]), file=log_f)
                print("MIP_L: ",np.array([latency_overhead[t, i].x for i in range(M)]), file=log_f)
                print("MIP_DIFF: ",np.array([aux0[t, i].x  for i in range(M)]), file=log_f)
                print("MIP_ENERGY: ", energy[:, start_time+t], file=log_f)
                print("MIP_POWER: ", np.array([powered_core[t, j].getValue() for j in range(N)]), file=log_f)
                if print_vm:
                    print("MIP_VM", np.array([[powered_vms[t, i, j].x for j in range(N)] for i in range (M)]), file=log_f)
        
        sys.stdout.flush()
            # print("Completed VMs", [t_complete[i].getValue() - 1 + last_iteration_residual[i].getValue() for i in range(M)], file=log_f)
        # print("MIP_VM_DIFF", np.array([[vms_to_boot[0, i, j].x for i in range(M)] for j in range (N)]))
        # for t in range(T):
        #     print("MIP_VM_DIFF", np.array([[vms_to_boot[t, i, j].x for i in range(M)] for j in range (N)]))
        # for t in range(T):
        #     print("MIP_VM", np.array([[powered_vms[t, i, j].x for i in range(M)] for j in range (N)]))
        # if init_vms is not None:
        #     print("MIP_REAL_VM_DIFF", np.array([[powered_vms[0, i, j].x - init_vms[i, j] for i in range(M)] for j in range (N)]))
        #     print("MIP_INIT_VMS", init_vms)
            
        # for t in range(T):
        #     print("MIP_REAL_P: ",[latency_overhead[t, 7].x * (sufficient[t, 7].x - overhead[t, 7].x)])
        #     print("MIP_REAL_P: ",[progress[t, 7].x])
        # print("MIP_REC: ",[recomputation_overhead[0, i].x  for i in range(M)])
        # if init_vms is not None:
        #     return np.array([[round(powered_vms[1, i, j].x) for j in range(N)] for i in range(M)], dtype=int)
        # else:
        #     # print([sum([progress[t, i].x for t in range(T)]) for i in range(M)])
        #     # print([last_iteration_residual[i].getValue()  for i in range(M)])
        #     # print(avg_complete.x / M)
        #     # print(avg_overhead.getValue() / M)
        return np.array([[[round(powered_vms[t, i, j].x) for j in range(N)] for i in range(M)] for t in range(T)], dtype=int)
       

class AppProfile():
    def __init__(self, app, T):
        self.app = app
        self.finished = False
        self.clear(T)

    def clear(self, T):
        self.queuing, self.blackout, self.overhead = 0, 0, 0
        self.exec, self.start, self.end, self.time = -1, -1, -1, -1
        self.overhead_breakdown = np.zeros(3)
        self.migration_timestamp = []
        self.migration_distribution = [np.zeros(5) for _ in range(T)]
        self.per_step_progress = [0 for _ in range(T)]
        
    def normalize(self, factor):
        self.queuing /= factor
        self.blackout /= factor
        self.overhead /= factor
        self.exec /= factor
        self.start /= factor
        self.end /= factor
        self.time /= factor
        self.overhead_breakdown /= factor
        self.migration_timestamp = [t/factor for t in self.migration_timestamp]
        self.migration_distribution = [dist/factor for dist in self.migration_distribution]

    def __str__(self):
        return f"{self.queuing}, {self.overhead}, {self.exec}, {self.start}, {self.end}, {self.time}"



class GlobalProfile():
    def __init__(self, T):
        self.migration_distribution = [_ for _ in range(T)]
        
    def normalize(self, factor):
        pass
    
    def __str__(self):
        return str(self.migration_distribution)
    
@ray.remote
class VBSimulator(Policy):
    def __init__(self, policy, energy, apps, regions, subgraph_id, opt_time, opt_goal, step=-1, timestep=60, misprediction_ratio = 0.0):
        self.policy = policy
        self.energy = energy
        self.apps = apps
        self.regions = regions
        self.step = step
        self.subgraph_id = subgraph_id
        
        MAX_STEP = self.step
        self.TIMESTEP = timestep

        self.T, self.M, self.N = MAX_STEP, len(self.apps), self.regions
        self.running_vms = np.zeros((self.T, self.M, self.N))
        self.power_per_vm = np.array([app.cores_per_vm for app in self.apps])
        self.vm_to_progress = np.array([app.num_vms for app in self.apps])
        self.progress = np.array([app.completion_time for app in self.apps], dtype=np.float64)
        self.blocking_time = np.array([app.start * self.TIMESTEP for app in self.apps], dtype=np.float64) 
        # self.blocking_time = np.zeros(len(self.apps))
        self.time_to_start = np.array([app.start for app in self.apps], dtype=np.float64)
        self.completed_vm = dict()

        self.app_profiles = np.array([AppProfile(self.apps[i], self.T) for i in range(self.M)])
        self.global_profile = GlobalProfile(self.T)
        self.utilization = np.zeros(self.T)

        self.t = 0

        self.energy_contributed = 0
        
        self.prev_ret = None
        
        # TODO: this is important to the queuing delay, as MIP will NOT schedule any VMs to an app if it cannot finish 
        if self.policy == "mip":
            self.lookahead = 16 # self.T // 2
        elif self.policy == "mip-app":
            self.lookahead = 1
        else:
            self.lookahead = 1
        
        self.lookahead_t = 0

        self.opt_time = opt_time
        self.opt_goal = opt_goal
        
        self.predicted_energy_per_timestamp = dict()
        self.misprediction_ratio = misprediction_ratio
        if self.policy == "mip-app":
            for t in range(self.T):
                self.predicted_energy_per_timestamp[t] = np.tile(np.array([self.energy[:, t]]).transpose(), (1, self.T+1))
        elif self.policy == "mip":
            random_scale = np.random.uniform(1-misprediction_ratio,1+misprediction_ratio,self.T)
            mispredicted_energy = self.energy.copy()
            for t in range(self.T):
                mispredicted_energy[:, t] *= random_scale[t]
            for t in range(self.T):
                self.predicted_energy_per_timestamp[t] = mispredicted_energy.copy()
                self.predicted_energy_per_timestamp[t][:, t] = self.energy[:, t]


    def greedy_decision(self, fair_sched=False):
        T, M, N, t = self.T, self.M, self.N, self.t

        if t == 0:
            expected_running_vms = np.zeros((M, N))
        else:
            expected_running_vms = np.copy(self.running_vms[t-1, :, :])
            for i in self.completed_vm:
                expected_running_vms[i, :] = np.zeros((N))

        # Step one: Migration
        migrated_apps = set()
        if fair_sched:
            power = np.matmul(self.power_per_vm, expected_running_vms[:, :])
            under_util_sites = []
            over_util_sites = []
            for j in range(N):
                power = np.matmul(self.power_per_vm, expected_running_vms[:, :])
                if power[j] > self.energy[j,t]:
                    over_util_sites.append((j, power[j] - self.energy[j,t]))
                else:
                    under_util_sites.append((j, power[j] - self.energy[j,t]))
                
            for over_site, over_power in over_util_sites:
                for under_site, under_power in under_util_sites:
                    for i in range(M):
                        if i in self.completed_vm:
                            continue
                        num_vms = 1
                        if under_power + self.power_per_vm[i] * num_vms >= 0:
                            break
                        if over_power <= 0:
                            break
                        if expected_running_vms[i, over_site] < num_vms:
                            continue
                        expected_running_vms[i, over_site] -= num_vms
                        expected_running_vms[i, under_site] += num_vms
                        under_power += self.power_per_vm[i] * num_vms
                        over_power -= self.power_per_vm[i] * num_vms
                        migrated_apps.add(i)
                        # log_msg(f"App {i} from site {over_site} to site {under_site}")

                if over_power <= 0:
                    break      
        
        # Step two: Eviction
        evicted_apps = set()
        evicted_sites = set()
        old_expected_running_vms = expected_running_vms.copy()
        old_power_per_vm = self.power_per_vm.copy()
        old_power = np.matmul(self.power_per_vm, expected_running_vms[:, :])
        log = []
        for j in range(N):
            power = np.matmul(self.power_per_vm, expected_running_vms[:, :])
            if power[j] > self.energy[j,t]:
                over = power[j] - self.energy[j,t]
                evicted_sites.add(j)
                dv = [[i, expected_running_vms[i, j], self.power_per_vm[i], int(i not in migrated_apps)] for i in range(M) if expected_running_vms[i, j] > 0]
                # dv = sorted(dv, key=lambda x: x[1]*x[2]*x[3], reverse=False)
                start = random.randint(0, len(dv)-1)
                while over > 0:
                    i, vms, unit_power, none_migrated = dv.pop(start)
                    # evict_vms = min(vms, int(math.ceil(over / unit_power)))
                    evict_vms = vms
                    # remove redundant VMs for evicted apps at all sites
                    expected_running_vms[i, :] = 0
                    over -= evict_vms*unit_power
                    evicted_apps.add(i)
                    log.append((j, i, vms, unit_power, none_migrated))
                    if start >= len(dv):
                        start = 0

        power = np.matmul(self.power_per_vm, expected_running_vms[:, :])
        for j in range(N):
            try:
                assert((expected_running_vms >= 0).all())
                assert(power[j] <= self.energy[j, t])
            except:
                print(log)
                print(old_power, old_power_per_vm)
                print(old_expected_running_vms)
                print(evicted_apps, evicted_sites)
                print(self.energy[:, t], self.power_per_vm, expected_running_vms[:, :])
                exit()

        # Step three: Schedule VMs
        power = np.matmul(self.power_per_vm, expected_running_vms[:, :])
        # greedily fill rest of energy
        for j in range(N):
            if j in evicted_sites:
                continue
            if power[j] < self.energy[j,t]:
                rest = self.energy[j,t] - power[j]
                # decision vector
                required_vm_to_progress = self.vm_to_progress - np.sum(expected_running_vms[:, :], axis = 1)
                # dv = [[i, required_vm_to_progress[i], self.power_per_vm[i]] for i in range(M) if required_vm_to_progress[i] > 0 and i not in self.completed_vm and self.time_to_start[i] <= t]
                # dv = sorted(dv, key=lambda x: x[1]*x[2], reverse=False)
                # random.shuffle(dv)
                # while rest > 0 and len(dv) > 0:
                if rest <= 0:
                    continue
                # TODO: check if required_vm_to_progress is less than 0 ()
                for i in range(M):
                    if i in self.completed_vm:
                        continue
                    vms, unit_power = required_vm_to_progress[i], self.power_per_vm[i]
                    fill_vms = min(vms, int(math.floor(rest / unit_power)))
                    # assert(fill_vms >= 0)
                    if fill_vms <= 0:
                        continue
                    expected_running_vms[i, j] += fill_vms
                    rest -= fill_vms*unit_power
                    assert(rest >= 0)
                    if rest < min(self.power_per_vm):
                        break
            
        power = np.matmul(self.power_per_vm, expected_running_vms[:, :])
        for j in range(N):
            try:
                assert((expected_running_vms >= 0).all())
                assert(power[j] <= self.energy[j, t])
            except:
                print(log)
                print(old_power, old_power_per_vm)
                print(old_expected_running_vms)
                print(evicted_apps, evicted_sites)
                print(self.energy[:, t], self.power_per_vm, expected_running_vms[:, :])
                exit()

        return expected_running_vms

    def mip_decision(self, hint=None, backlog=False, new_apps=True):
        
        if backlog:
            if self.t == 0:
                ret = BATCH_MIP.migrate(self.t, min(self.lookahead, self.step - self.t), self.TIMESTEP, self.energy, self.predicted_energy_per_timestamp[self.t], self.apps, self.regions, self.app_profiles, self.opt_time, self.opt_goal, init_vms=None, pre_migration=True, hint=hint, subgraph_id = self.subgraph_id)
                self.running_vms[:, :, :] = ret
                
                if np.isnan(np.sum(self.running_vms[:, :, :])):
                    return False
            
        else:
            # print(new_apps or self.lookahead_t <= self.t)
            if new_apps or self.lookahead_t <= self.t or self.policy == "mip-app" or self.misprediction_ratio != 0.0:
                if self.t == 0:
                    ret = BATCH_MIP.migrate(self.t, min(self.lookahead, self.step - self.t), self.TIMESTEP, self.energy, self.predicted_energy_per_timestamp[self.t], self.apps, self.regions, self.app_profiles, self.opt_time, self.opt_goal, init_vms=None, pre_migration=True, hint=hint, policy=self.policy, subgraph_id = self.subgraph_id, blocking_time = self.blocking_time)

                    # self.running_vms[:10, :, :] = ret[:10, :, :]
                else:
                    # print([(self.apps[i].completion_time - self.progress[i]) / self.TIMESTEP for i in range(len(self.apps))])
                    # print(list(self.completed_vm.keys()))
                    # hint = self.prev_ret[1, :, :]
                    ret = BATCH_MIP.migrate(self.t, min(self.lookahead, self.step - self.t), self.TIMESTEP, self.energy, self.predicted_energy_per_timestamp[self.t], self.apps, self.regions, self.app_profiles,self.opt_time, self.opt_goal, init_vms=self.running_vms[self.t-1, :, :], init_progress = [(self.apps[i].completion_time - self.progress[i]) / self.TIMESTEP for i in range(len(self.apps))], completed_vm = list(self.completed_vm.keys()), pre_migration=True, hint=hint, policy=self.policy, subgraph_id = self.subgraph_id, blocking_time = self.blocking_time)
                
                assert(ret is not None)
                self.prev_ret = ret
                
                self.lookahead_t = min(self.step, self.t+self.lookahead)

                self.running_vms[self.t:self.t+self.lookahead, :, :] = ret[0:self.lookahead, :, :]
            else:
                self.running_vms = self.running_vms
            
        for i in self.completed_vm:
            self.running_vms[self.t, i, :] = np.zeros((self.N))
        
        assert((self.running_vms >= 0).all())

        return True


    def migrate(self, new_apps=None, hint=None, cloned=False):
        has_new_apps = False
        if new_apps and len(new_apps) > 0:
            has_new_apps = True
            self.apps += new_apps
            self.M += len(new_apps)

            new_running_vms = np.zeros((self.T, len(new_apps), self.N))
            self.running_vms = np.concatenate((self.running_vms, new_running_vms), axis=1)

            power_per_new_vm = np.array([app.cores_per_vm for app in new_apps])
            self.power_per_vm = np.concatenate((self.power_per_vm, power_per_new_vm), axis=0)
            new_vm_to_progress = np.array([app.num_vms for app in new_apps])
            self.vm_to_progress = np.concatenate((self.vm_to_progress, new_vm_to_progress), axis=0)
            new_vm_progress = np.array([app.completion_time for app in new_apps])
            self.progress = np.concatenate((self.progress, new_vm_progress), axis=0)
            new_time_to_start = np.array([app.start for app in new_apps], dtype=np.float64)
            self.time_to_start = np.concatenate((self.time_to_start, new_time_to_start), axis=0)
            new_breakdown = np.array([AppProfile(new_apps[i], self.T) for i in range(len(new_apps))])
            self.app_profiles = np.concatenate((self.app_profiles, new_breakdown), axis=0)
            
            new_blocking_time = np.array([(app.start - self.t) * self.TIMESTEP for app in new_apps], dtype=np.float64)
            # new_blocking_time = np.zeros(len(new_apps))
            self.blocking_time = np.concatenate((self.blocking_time, new_blocking_time), axis=0)

            # print(self.running_vms.shape, self.power_per_vm.shape, power_per_new_vm.shape, len(new_apps))

        t_schedule = 0
        if len(self.apps) == 0:
            self.t += 1
            return self, t_schedule


        T, M, N, t = self.T, self.M, self.N, self.t
        if t == 0:
            self.running_vms[0, :, :] = np.zeros((M, N))

        if not cloned:
           # log_msg(f"Step {t}")
            pass
        #     log_msg(list(self.completed_vm.keys()))
        
        if self.policy == "greedy":
            expected_running_vms = self.greedy_decision()
            self.running_vms[self.t, :, :] = expected_running_vms
        elif self.policy == "greedy-fair":
            expected_running_vms = self.greedy_decision(fair_sched=True)
            self.running_vms[self.t, :, :] = expected_running_vms
        elif self.policy == "mip" or self.policy == "mip-app":
            cloned_scheduler = copy.deepcopy(self)
            cloned_scheduler.policy = "greedy"
            cloned_scheduler.energy = self.predicted_energy_per_timestamp[self.t]
            # if self.policy == "mip-app":
            #     cloned_scheduler.energy = np.tile(np.array([cloned_scheduler.energy[:, self.t]]).transpose(), (1, self.T+1))
            lookahead = min(self.lookahead, self.step - self.t)
            for i in range(lookahead):
                cloned_scheduler.migrate(cloned=True)
            hint = cloned_scheduler.running_vms[self.t:self.t+lookahead, :, :]
            # hint = self.greedy_decision()
            # hint = None
            t_start = time.time()
            success = self.mip_decision(hint, backlog=False, new_apps=has_new_apps)
            t_schedule = time.time() - t_start
            if not success:
                log_msg("Fall back to greedy")
                self.greedy_decision()

        can_progress = np.array([1 - max(0, min(1, x)) for x in self.vm_to_progress - np.sum(self.running_vms[t, :, :], axis = 1)])
        # print(self.vm_to_progress)
        # print(np.sum(self.running_vms[t, :, :], axis = 1))
        # print(can_progress)

        alloc_overhead = np.zeros(M)
        blackout_overhead = np.zeros(M)
        queuing_overhead = np.zeros(M)
        recompute_overhead = np.zeros(M)
        latency_overhead = np.zeros(M)
        latency_multiplier = np.zeros(M)
        diff_vms = np.zeros((M,N))
        if t > 0:
            diff_vms = self.running_vms[t, :, :] - self.running_vms[t-1, :, :]
            diff_vms[diff_vms < 0] = 0
            alloc_overhead = []
            # recompute_overhead = np.array([min(1,max(diff_vms[i,j] for j in range(N))) * self.apps[i].recomputation for i in range(M)])
            recompute_overhead = []
            for i in range(M):
                if self.progress[i] < self.apps[i].completion_time:
                    alloc_overhead.append(max(diff_vms[i,j] for j in range(N)) * self.apps[i].alloc_overhead)
                else:
                    alloc_overhead.append(0)

                if self.apps[i].recomputation and self.progress[i] < self.apps[i].completion_time:
                    recompute_overhead.append(min(1,max(diff_vms[i,j] for j in range(N))) * ((self.apps[i].completion_time - self.progress[i]) % self.apps[i].recomputation))
                    # if min(1,max(diff_vms[i,j] for j in range(N))):
                    #     print("recompute", self.apps[i].name, self.apps[i].recomputation / 2)
                    # recompute_overhead.append(min(1,max(diff_vms[i,j] for j in range(N))) * self.apps[i].recomputation / 2)
                else:
                    recompute_overhead.append(0)
                    
                
                if i not in self.completed_vm:
                    if max(diff_vms[i, :]) > 0 and self.progress[i] < self.apps[i].completion_time:
                        self.app_profiles[i].migration_timestamp.append(t)
        
        latency_multiplier = []
        for i in range(M):
            if not self.apps[i].num_vms >= max(self.running_vms[t,i,:]):
                print(t, i, self.apps[i].num_vms, self.running_vms[t,i,:])
            assert(self.apps[i].num_vms >= max(self.running_vms[t,i,:]))
            has_latency = max(0, min(1, self.apps[i].num_vms-max(self.running_vms[t,i,:]))) 
            if has_latency:
                latency_multiplier.append(self.apps[i].latency)
            else:
                latency_multiplier.append(1)

        recompute_overhead = np.array(recompute_overhead)
        latency_multiplier = np.array(latency_multiplier)
        alloc_overhead = np.array(alloc_overhead)
        
        propagated_blocking_time = self.blocking_time.copy()
        self.blocking_time += alloc_overhead
        self.progress += recompute_overhead
        time_block_this_step = np.minimum(self.blocking_time, self.TIMESTEP)
        blocking_overhead = can_progress * time_block_this_step
        alloc_overhead = can_progress * alloc_overhead
        time_avail_compute = np.minimum(self.progress / latency_multiplier, can_progress * self.TIMESTEP - blocking_overhead)
        latency_overhead = time_avail_compute * (1 - latency_multiplier)
        time_avail_compute_with_latency = time_avail_compute * latency_multiplier
        self.progress -= time_avail_compute_with_latency
        
        residual = time_avail_compute + blocking_overhead
        # all potential available computation time
        non_blocking_time_without_alloc = np.maximum(self.TIMESTEP - propagated_blocking_time, 0)
        for i in range(M):
            if self.progress[i] == self.apps[i].completion_time:
                queuing_overhead[i] = (1 - can_progress[i]) * non_blocking_time_without_alloc[i]
            else:
                blackout_overhead[i] = (1 - can_progress[i]) * non_blocking_time_without_alloc[i]

        for i in range(M):
            self.app_profiles[i].per_step_progress[t] = time_avail_compute_with_latency[i] - recompute_overhead[i]
        
        
        # make_progress = can_progress * self.TIMESTEP - alloc_overhead - recompute_overhead
        # for i in range(M):
            # if make_progress[i] < 0:
            #    print(can_progress * self.TIMESTEP, alloc_overhead, recompute_overhead)
            # assert(make_progress[i] >= 0)
        # make_progress = np.minimum(self.progress / latency_multiplier, make_progress)
        # latency_overhead = np.clip(make_progress * (1 - latency_multiplier), 0, None)
        # residual = make_progress - latency_overhead + alloc_overhead + recompute_overhead
        # self.progress -= make_progress - latency_overhead 
        
        self.blocking_time -= self.TIMESTEP
        self.blocking_time = np.maximum(self.blocking_time, 0)
        
        if not cloned:
            if self.policy == "mip":
                log_name = f"logs/mip_sim_{self.subgraph_id}.log"
            elif self.policy == "mip-app":
                log_name = f"logs/mip_app_sim_{self.subgraph_id}.log"
            else:
                log_name = f"/home/js39/software/virtual-battery/logs/sim_{self.subgraph_id}.log"
            if t == 0:
                mode = "w+"
            else:
                mode = "a+"
            with open(log_name, mode) as log_f:
                print("SIM_G: ", self.subgraph_id, file=log_f)
                print("SIM_T: ", t, file=log_f)
                print("SIM_P: ", format_numpy((time_avail_compute_with_latency - recompute_overhead) / self.TIMESTEP), file=log_f)
                print("SIM_A: ", format_numpy(alloc_overhead / self.TIMESTEP), file=log_f)
                print("SIM_R: ", format_numpy(recompute_overhead / self.TIMESTEP), file=log_f)
                print("SIM_L: ", format_numpy(latency_overhead / self.TIMESTEP), file=log_f)
                print("SIM_DIFF", format_numpy([max(diff_vms[i,:]) for i in range(M)]), file=log_f)
                print("SIM_REST_P: ", format_numpy(self.progress / self.TIMESTEP), file=log_f)
                # print("SIM_VM", self.running_vms[t, :, :], file=log_f)
                # print("SIM_VM_DIFF: ", diff_vms, file=log_f)

                log_f.flush()
        
        # if t == 2 and self.policy == "mip":
        #     exit()

        # self.progress -= can_progress * self.TIMESTEP - alloc_overhead - recompute_overhead - can_progress * latency_overhead

        # for i in range(len(self.apps)):
        #     if latency_overhead[i] < 0:
        #         print(self.apps[i].num_vms, self.running_vms[t,i,:])
        # log_msg(t, can_progress * TIMESTEP - alloc_overhead - recompute_overhead - can_progress * latency_overhead)
        # log_msg(t, progress)

        # log_msg(progress, alloc_overhead, recompute_overhead, latency_overhead)
        power = np.matmul(self.power_per_vm, self.running_vms[t, :, :])
        self.utilization[t] = np.sum(power)

        for j in range(N):
            # print(j, t, power[j], self.energy[j, t], self.power_per_vm, self.running_vms[t, :, j])
            assert(power[j] <= self.energy[j, t])
        
        self.energy_contributed += sum([self.apps[i].total_cores * can_progress[i] for i in range(M)])

        # log breakdown
        for i in range(M):
            if self.app_profiles[i].start == -1:
                self.app_profiles[i].start = self.time_to_start[i]
                self.app_profiles[i].exec = self.apps[i].completion_time / self.TIMESTEP
            
            if i not in self.completed_vm:
                self.app_profiles[i].queuing += queuing_overhead[i]  / self.TIMESTEP
                self.app_profiles[i].blackout += blackout_overhead[i]  / self.TIMESTEP
                self.app_profiles[i].overhead += (alloc_overhead[i] + recompute_overhead[i] + latency_overhead[i]) / self.TIMESTEP
                self.app_profiles[i].overhead_breakdown += np.array((alloc_overhead[i], recompute_overhead[i], latency_overhead[i])) / self.TIMESTEP
                self.app_profiles[i].migration_distribution[self.t] = np.vstack((alloc_overhead[i], recompute_overhead[i], latency_overhead[i], blackout_overhead[i], queuing_overhead[i])) / self.TIMESTEP
                
                
        self.global_profile.migration_distribution[self.t] = np.vstack((alloc_overhead, recompute_overhead, latency_overhead, blackout_overhead, queuing_overhead)) / self.TIMESTEP

        for i in range(M):
            if self.progress[i] <= 1e-9 and i not in self.completed_vm:
                # self.completed_vm[i] = t + self.progress[i] / self.TIMESTEP - self.time_to_start[i] + 1
                self.completed_vm[i] = t + residual[i] / self.TIMESTEP - self.time_to_start[i] 
                self.app_profiles[i].end = t + residual[i] / self.TIMESTEP
                self.app_profiles[i].time = self.completed_vm[i]
                self.app_profiles[i].finished = True
                # log_msg(t, progress[i] / TIMESTEP, time_to_start[i])
                    
        # log_msg(self.energy[:, t])
        # log_msg(self.running_vms)
        # log_msg(can_progress)
        # log_msg(t)
        # log_msg(self.completed_vm)

        self.t += 1

        return self, t_schedule

    def status(self):
        return self.completed_vm

    # for the greedy model
    # @property
    def remaining_energy(self, target_util=1.0, t=None):
        power_per_active_vms = self.power_per_vm.copy()
        for i in range(self.M):
            if i in self.completed_vm:
                power_per_active_vms[i] = 0
        power = np.matmul(power_per_active_vms, self.running_vms[self.t - 1, :, :])
        for j in range(self.N):
            assert(power[j] >= 0)
        if t is None:
            t = self.t
        remaining_power_per_site = self.energy[:, t] * target_util - power

        # evenly distribute the power of hibernated vms onto sites
        potential_power = sum(power_per_active_vms[i] * self.vm_to_progress[i] for i in range(self.M))
        hibernated_power = potential_power - sum(power)
        if hibernated_power > 0:
            total_positive_power = sum(_ for _ in remaining_power_per_site if _ > 0)
            # print(t, self.subgraph_id, hibernated_power, remaining_power_per_site, total_positive_power)
            for j in range(self.N):
                if remaining_power_per_site[j] > 0:
                    share_of_hibernated_power = hibernated_power * remaining_power_per_site[j] / total_positive_power
                    remaining_power_per_site[j] -= share_of_hibernated_power
        # print(t, self.subgraph_id, remaining_power_per_site)
        return remaining_power_per_site

    # for MIP model
    # @property
    def predicted_remaining_energy(self, target_util=1.0):   

        # cloned_scheduler = copy.deepcopy(self)
        # cloned_scheduler.policy = "greedy"
        # lookahead = min(self.lookahead, self.step - self.t)
        # for i in range(lookahead):
        #     cloned_scheduler.migrate(cloned=True)
        
        # predicted_remaining_energy = np.zeros((self.N, self.T))
        # for t in range(self.t, self.t+lookahead):
        #     power = np.matmul(self.power_per_vm, cloned_scheduler.running_vms[t, :, :])
        #     for j in range(self.N):
        #         assert(power[j] >= 0)
        #     remaining_power_per_site = self.energy[:, t] * target_util - power 
        #     predicted_remaining_energy[:, t] = remaining_power_per_site

        # return predicted_remaining_energy   

        predicted_remaining_energy = np.zeros((self.N, self.T))
        for t in range(self.t, self.T):
            predicted_remaining_energy[:, t] = self.remaining_energy(target_util, t)

        return predicted_remaining_energy

    def finish(self):
        if len(self.apps) == 0:
            return 24, 0, [], 0, self.app_profiles, self.global_profile, self.utilization, self.running_vms

        # log_msg("Greedy Completion Time", t, t*TIMESTEP)

        non_completion_penalty = 1 # 0.25 * self.T
        naturally_completed = len(self.completed_vm)
        # if len(self.completed_vm) != len(self.apps):
        #     for i in range(len(self.apps)):
        #         if i not in self.completed_vm:
        #             self.completed_vm[i] = (self.T + non_completion_penalty + self.progress[i] / self.TIMESTEP) - self.time_to_start[i]
        #             self.app_profiles[i].end = self.T + non_completion_penalty + self.progress[i] / self.TIMESTEP
        #             self.app_profiles[i].time = self.completed_vm[i]
        #             if self.progress[i] == self.apps[i].completion_time:
        #                 self.app_profiles[i].queuing += non_completion_penalty
        #             else:
        #                 self.app_profiles[i].blackout += non_completion_penalty

        # print(len(self.apps))
        for i in range(len(self.apps)):  
            if i not in self.completed_vm:
                continue  
            try:
                assert(abs((self.app_profiles[i].end - self.app_profiles[i].start) - (self.app_profiles[i].queuing + self.app_profiles[i].blackout + self.app_profiles[i].overhead + self.app_profiles[i].exec)) <= 1e-5)
            except:
                    print(i, self.app_profiles[i].start, self.app_profiles[i].end, self.app_profiles[i].queuing, self.app_profiles[i].blackout, self.app_profiles[i].overhead, self.app_profiles[i].exec)
                    print(self.app_profiles[i].migration_distribution)
                    print(self.app_profiles[i].migration_timestamp)
                    print(self.app_profiles[i].per_step_progress)
                    exit()
            
        # print(np.average([self.app_profiles[i].end - self.app_profiles[i].start for i in range(len(self.apps))]), np.average([self.app_profiles[i].queuing for i in range(len(self.apps))]), np.average([self.app_profiles[i].blackout for i in range(len(self.apps))]), np.average([self.app_profiles[i].overhead for i in range(len(self.apps))]), np.average([self.app_profiles[i].exec for i in range(len(self.apps))]))

        # normalize everything into hours
        for i in range(len(self.apps)):
            if i not in self.completed_vm:
                continue
            self.completed_vm[i] /= (60 // self.TIMESTEP)
            self.app_profiles[i].normalize(60 // self.TIMESTEP)
            
        if len(self.completed_vm) > 0:
            avg_t = np.average(list(self.completed_vm.values()))    
        else:
            avg_t = 0.0    
        
        # log_msg(self.energy_contributed, np.sum(self.energy))
        # with open("logs/sim.log", "a+") as log_f:
        #     print([self.completed_vm[i] for i in range(len(self.apps))], file=log_f)

        # overheads = np.average([self.app_profiles[i].overhead for i in range(self.M)])
        # queuing = np.average([self.app_profiles[i].queuing for i in range(self.M)])
        # execution = np.average([self.app_profiles[i].exec for i in range(self.M)])

        # log_msg("Greedy Average Completion Time", avg_t, avg_t*TIMESTEP)
        return self.t, avg_t, list(self.completed_vm.values()), naturally_completed, self.app_profiles, self.global_profile, self.utilization, self.running_vms


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
    


if __name__ == "__main__":
    # global vars
    DATA_PATH = '/home/jhsun/software/virtual-battery/data'
    RENEWABLE_PATH = os.path.join(DATA_PATH, 'renewable/')
    SIMULATION_OUTPUT_PATH = os.path.join(DATA_PATH, 'simulator-validation/simulation_output')
    EPS = 1e-6
    sites = 2
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
    num_per_app = [5,5,10]#[scale]*3

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
    # energy = np.repeat(energy, 15, axis=1)
    # with np.printoptions(threshold=np.inf):    
    #     log_msg(energy)

    # log_msg(np.sum(energy,axis=0))
    # init_energy = energy[:,0].copy()
    # min_cores = np.sum(energy,axis=0).min()
    # init_energy = init_energy*min_cores//sum(init_energy)

    # the time unit is one minute
    # apps = create_workloads(num_per_app)

    # energy, apps, sites, step, init = use_simple_example()

    # if policy == "mip":
    #     policy = BATCH_MIP("batch-mip", energy, apps, sites)
    #     policy.migrate(step=step, init=init, pre_migration=False)
    # elif policy == "pre":
    #     policy = BATCH_MIP("batch-mip-pre", energy, apps, sites)
    #     policy.migrate(step=step, init=init, pre_migration=True)

    template = App(name="cofii-e4s", num_vms=4, vm_type=0, cores_per_vm=4, completion_time=10, migration=2, recomputation=0)
    apps = []
    for i in range(5):
        new_app = copy.deepcopy(template)
        new_app.id = i
        apps.append(new_app)

    policy = GREEDY("greedy", energy, apps, sites)

    energy = energy[0:sites,200:]

    results = policy.migrate(step=20, timestep=1)
    # print(results)
    schedule = np.reshape(results[-1], (20,-1)).transpose()

    for row in schedule:
        print(" ".join(map(str, map(int, row))))

 