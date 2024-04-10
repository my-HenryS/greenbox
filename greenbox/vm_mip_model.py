import gurobipy as gp
from gurobipy import GRB
import numpy as np
from macros import *
from factory import VM, VMProfile

MIGRATE_IN_CONST_PER_GB = 0.08 # Watt per GB under 1 Gbps network
MIGRATE_OUT_CONST_PER_GB = 0.08
EPSILON = 1e-3
GRB_ENV.setParam("OutputFlag",0)
GRB_ENV.start()

class NEW_MIP():
    def __init__(self):
        pass

    @staticmethod
    def migrate(cur_timestamp, timestamps, interval, energy, predicted_energy_copy, vms, num_rMDCs, SLO_AVAIL = 0.99, MIP_OPT_TIME = 10000, MIP_OPT_GOAL = 0.01, init_states=None, hints=None, policy="mip", subgraph_id = 0):
        TIME, NUM_VM, NUM_SITE = timestamps, len(vms), num_rMDCs
        NUM_LOW_PRIO_VM = len([vm for vm in vms if vm.priority == 1])
        predicted_energy = np.copy(predicted_energy_copy)
        
        np.set_printoptions(threshold=sys.maxsize)
        print(f"curr_timestamp:{cur_timestamp}, time range:{timestamps}, predicted energy:{predicted_energy[:, cur_timestamp:cur_timestamp+timestamps]}")
        print(f"num of vms: {NUM_VM}, vms:{vms}")  
        # print(hints[0].shape)
        # print(vms[0].start_time / interval, vms[0].lifetime / interval)
        # print(f"init_states:{init_states}")
        
        # for n in range(NUM_SITE):
        #     predicted_energy[n, cur_timestamp] = energy[n, cur_timestamp] 

        model = gp.Model()
        logfile = f"logs/mip-model-{subgraph_id}.log"
        f = open(logfile, "a").close()
        output_file = "logs/mip-output-{subgraph_id}.log"
        model.Params.LogToConsole = 0
        model.Params.OutputFlag = 1
        model.Params.LogFile = logfile

        # input
        per_vm_slo = False
        no_migration = False

        # vars
        # TODO: separate running vm and placement
        placement = model.addVars(TIME+1, NUM_VM, NUM_SITE, vtype=GRB.BINARY)
        booted_vms = model.addVars(TIME, NUM_VM, NUM_SITE, vtype=GRB.BINARY)
        migration = model.addVars(TIME, NUM_VM, NUM_SITE, NUM_SITE, vtype=GRB.BINARY)
        complete = model.addVars(TIME+1, NUM_VM, vtype=GRB.BINARY)
        progress = model.addVars(TIME, NUM_VM, vtype=GRB.CONTINUOUS)
        exec_time = model.addVars(TIME, NUM_VM, vtype=GRB.CONTINUOUS)
        compl_time = model.addVars(NUM_VM, vtype=GRB.CONTINUOUS)
        has_completed = model.addVars(NUM_VM, vtype=GRB.BINARY)
        nr_power = model.addVars(TIME, NUM_SITE, vtype=GRB.CONTINUOUS, lb=0)
        r_power = model.addVars(TIME, NUM_SITE, vtype=GRB.CONTINUOUS, lb=0)
        avail_per_vm = model.addVars(NUM_VM, vtype=GRB.CONTINUOUS, lb=0, ub=1)
        vm_overhead = model.addVars(TIME, NUM_SITE, vtype=GRB.CONTINUOUS)
        migrate_in_overhead = model.addVars(TIME, NUM_SITE, vtype=GRB.CONTINUOUS)
        migrate_out_overhead = model.addVars(TIME, NUM_SITE, vtype=GRB.CONTINUOUS)
        # vm_lifetime = model.addVars(TIME, NUM_VM, vtype=GRB.CONTINUOUS)

        # tmp vars
        tmpVar0 = model.addVars(TIME, NUM_VM, NUM_SITE, vtype=GRB.INTEGER)
        tmpVar1 = model.addVars(TIME, NUM_VM, NUM_SITE, vtype=GRB.BINARY)
        tmpVar2 = model.addVars(NUM_VM, vtype=GRB.CONTINUOUS)
        tmpVar3 = model.addVars(NUM_VM, vtype=GRB.CONTINUOUS)
        tmpVar4 = model.addVars(TIME, NUM_SITE, vtype=GRB.CONTINUOUS)
        tmpVar5 = model.addVars(TIME, NUM_SITE, vtype=GRB.CONTINUOUS)
        sum_avail_current_vms = model.addVar(vtype=GRB.CONTINUOUS)
        
        # added hints
        if hints is not None:
            placement_hint_t = min(TIME+1, hints[0].shape[0])
            booted_vms_hint_t = min(TIME, hints[1].shape[0])
            for t in range(placement_hint_t):
                for i in range(NUM_VM):
                    for j in range(NUM_SITE):
                        placement[t, i, j].start = int(hints[0][t, i, j])
                        if t < booted_vms_hint_t:
                            booted_vms[t, i, j].start =  int(hints[1][t, i, j])
            model.update()

        # initial vm placement at t = 0
        if init_states is None:
            model.addConstrs(placement[0, m, n] == 0 for n in range(NUM_SITE) for m in range(NUM_VM))
            init_progress = np.zeros(NUM_VM)
            global_exec, global_lifetime, global_avail, num_complete_vm = 0, 0, 0, 0
        else:
            init_placement, init_progress = init_states["init_placement"], init_states["init_progress"]
            model.addConstrs(placement[0, m, n] == init_placement[m, n] for n in range(NUM_SITE) for m in range(NUM_VM))
            global_exec, global_lifetime, global_avail, num_complete_vm = init_states["global_exec"], init_states["global_lifetime"], init_states["global_avail"], init_states["num_complete_vm"]
        

        # objective
        nr_total_power = gp.quicksum(nr_power[t, n] for t in range(TIME) for n in range(NUM_SITE))
        model.addConstrs(tmpVar4[t, n] == vm_overhead[t, n] + migrate_in_overhead[t, n] + migrate_out_overhead[t, n] for t in range(TIME) for n in range(NUM_SITE))
        model.addConstrs(tmpVar5[t,n] == predicted_energy[n, t+cur_timestamp] for t in range(TIME) for n in range(NUM_SITE))
        model.addConstrs(r_power[t, n] == gp.min_(tmpVar4[t, n], tmpVar5[t,n]) for t in range(TIME) for n in range(NUM_SITE))
        r_total_power = gp.quicksum(r_power[t, n] for t in range(TIME) for n in range(NUM_SITE))
        total_carbon = nr_total_power * 0.7 + r_total_power * 0.011
        if policy == "mip":
            model.setObjective(total_carbon, GRB.MINIMIZE)
            # model.setObjective(nr_total_power, GRB.MINIMIZE)
            # model.setObjectiveN(nr_total_power, 0, 1)
            # model.setObjectiveN(gp.quicksum(1- avail_per_vm[m] for m in range(NUM_VM) if vms[m].priority == 1), 1, 0)
        if policy == "mip-app":
            model.setObjective(total_carbon, GRB.MINIMIZE)
            # model.setObjective(nr_total_power, GRB.MINIMIZE)
            # model.setObjectiveN(nr_total_power, 0, 1)
            # model.setObjectiveN(gp.quicksum(1- avail_per_vm[m] for m in range(NUM_VM) if vms[m].priority == 1), 1, 0)
            # model.setObjectiveN(nr_total_power, 0, 1)
            # model.setObjectiveN(-sum_avail_current_vms, 1, 0)

        # avail constraints
        if not per_vm_slo:
            ''' here we use the average survival rate of VMs as SLO '''
            model.addConstrs(tmpVar2[m] == compl_time[m] * avail_per_vm[m] for m in range(NUM_VM))
            model.addConstrs(tmpVar3[m] == (cur_timestamp + timestamps - vms[m].start_time / interval) * avail_per_vm[m] for m in range(NUM_VM))
            model.addConstrs((has_completed[m] == 1) >> (tmpVar2[m] == vms[m].lifetime / interval) for m in range(NUM_VM))
            # model.addConstrs((has_completed[m] == 1) >> (tmpVar2[m] == vms[m].pred_lifetime / interval) for m in range(NUM_VM))
            model.addConstrs((has_completed[m] == 0) >> (tmpVar3[m] == exec_time[TIME-1, m]) for m in range(NUM_VM))
            model.addConstr(sum_avail_current_vms == gp.quicksum(avail_per_vm[m] for m in range(NUM_VM) if vms[m].priority == 1))
            model.addConstr(sum_avail_current_vms + global_avail * num_complete_vm >= SLO_AVAIL * (NUM_LOW_PRIO_VM + num_complete_vm))
            model.addConstrs(avail_per_vm[m] == 1 for m in range(NUM_VM) if vms[m].priority == 0)
            # model.addConstrs(avail_per_vm[m] >= SLO_AVAIL for m in range(NUM_VM) if vms[m].priority == 1)
        else:
            ''' here we use the individual survival rate of VMs as SLO '''
            raise NotImplementedError
            model.addConstrs(avail_per_vm[m] >= SLO_AVAIL for m in range(NUM_VM))


        # power constraints: VM Power Consumption + Migration Overhead as Source + Migration Overhead as Destination ≤ Renewable Power Supply + Non-Renewable Power Supply
        model.addConstrs(vm_overhead[t, n] == gp.quicksum(booted_vms[t, m, n] * vms[m].maxpower[cur_timestamp+t] for m in range(NUM_VM)) for t in range(TIME) for n in range(NUM_SITE))
        model.addConstrs(migrate_in_overhead[t, n] == gp.quicksum(MIGRATE_IN_CONST_PER_GB * migration[t, m, n1, n] * vms[m].memory for n1 in range(NUM_SITE) for m in range(NUM_VM)) for t in range(TIME) for n in range(NUM_SITE))
        model.addConstrs(migrate_out_overhead[t, n] == gp.quicksum(MIGRATE_OUT_CONST_PER_GB * migration[t, m, n, n2] * vms[m].memory for n2 in range(NUM_SITE) for m in range(NUM_VM)) for t in range(TIME) for n in range(NUM_SITE))
        model.addConstrs(vm_overhead[t, n] + migrate_in_overhead[t, n] + migrate_out_overhead[t, n] <= predicted_energy[n, cur_timestamp+t] + nr_power[t, n] for t in range(TIME) for n in range(NUM_SITE))
        # model.addConstrs(vm_overhead[t, n] == gp.quicksum(booted_vms[t, m, n] * vms[m].avg_power for m in range(NUM_VM)) for t in range(TIME) for n in range(NUM_SITE))
        # model.addConstrs(migrate_in_overhead[t, n] == gp.quicksum(MIGRATE_IN_CONST_PER_GB * migration[t, m, n1, n] * vms[m].memory for n1 in range(NUM_SITE) for m in range(NUM_VM)) for t in range(TIME) for n in range(NUM_SITE))
        # model.addConstrs(migrate_out_overhead[t, n] == gp.quicksum(MIGRATE_OUT_CONST_PER_GB * migration[t, m, n, n2] * vms[m].memory for n2 in range(NUM_SITE) for m in range(NUM_VM)) for t in range(TIME) for n in range(NUM_SITE))
        # model.addConstrs(vm_overhead[t, n] + migrate_in_overhead[t, n] + migrate_out_overhead[t, n] <= predicted_energy[n, cur_timestamp+t] + nr_power[t, n] for t in range(TIME) for n in range(NUM_SITE))

        # vms and migrate constraints: using placement[t, m, n] to bound migration[t, m, n1, n2]
        model.addConstrs(gp.quicksum(placement[t, m, n] for n in range(NUM_SITE)) <= 1 for m in range(NUM_VM) for t in range(1, TIME+1))
        model.addConstrs(placement[t, m, n] >= booted_vms[t, m, n] for n in range(NUM_SITE) for m in range(NUM_VM) for t in range(TIME))
        model.addConstrs((migration[t, m, n, n2] == 1) >> (placement[t, m, n] == 1) for m in range(NUM_VM) for t in range(TIME) for n in range(NUM_SITE) for n2 in range(NUM_SITE))
        model.addConstrs((migration[t, m, n, n2] == 1) >> (placement[t+1, m, n2] == 1) for m in range(NUM_VM) for t in range(TIME) for n in range(NUM_SITE) for n2 in range(NUM_SITE))

        # FIXME: here Gurobi has some numerical bugs to prevent it becoming an equality constraint, but changing it to inequality causes no problem.
        model.addConstrs((tmpVar0[t, m, n] >= placement[t, m, n] - placement[t+1, m, n]) for m in range(NUM_VM) for t in range(TIME) for n in range(NUM_SITE))
        model.addConstrs((tmpVar1[t, m, n] == gp.min_(tmpVar0[t, m, n], 1)) for m in range(NUM_VM) for t in range(TIME) for n in range(NUM_SITE))
        model.addConstrs((tmpVar1[t, m, n] == 1) >> (placement[t+1, m, n2] == migration[t, m, n, n2]) for m in range(NUM_VM) for t in range(TIME) for n in range(NUM_SITE) for n2 in range(NUM_SITE))
        if no_migration:
            model.addConstrs(migration[t, m, n, n2] == 0 for n2 in range(NUM_SITE) for n in range(NUM_SITE) for m in range(NUM_VM) for t in range(TIME)) 
        model.addConstrs(migration[t, m, n, n] == 0 for n in range(NUM_SITE) for m in range(NUM_VM) for t in range(TIME))

        # vm execution constaints: using progress[t, m] to bound exec_time[t, m]
        model.addConstrs(gp.quicksum(booted_vms[t, m, n] for n in range(NUM_SITE)) >= progress[t, m] for m in range(NUM_VM) for t in range(TIME))
        model.addConstrs(gp.quicksum(progress[t0, m] for t0 in range(t+1)) + init_progress[m] / interval == exec_time[t, m] for m in range(NUM_VM) for t in range(TIME))
        # model.addConstrs(exec_time[t, m] <= vms[m].lifetime / interval for t in range(TIME) for m in range(NUM_VM))
        # model.addConstrs(exec_time[t, m] <= vms[m].pred_lifetime / interval for t in range(1, TIME) for m in range(NUM_VM))
        # model.addConstrs(exec_time[0, m] <= vms[m].lifetime / interval for m in range(NUM_VM))
        model.addConstrs(exec_time[t, m] <= vms[m].pred_lifetime / interval for t in range(TIME) for m in range(NUM_VM))
        # model.addConstrs(exec_time[t, m] <= vm_lifetime[t, m] / interval for t in range(TIME) for m in range(NUM_VM))

        # vm lifetime constraint: pred_lifetime should be larger than the current lifetime
        # model.addConstrs(vm_lifetime[t, m] >= vms[m].pred_lifetime for t in range(1, TIME) for m in range(NUM_VM))
        # model.addConstrs(vm_lifetime[0, m] <= vm_lifetime[t, m] for t in range(1, TIME) for m in range(NUM_VM))
        # model.addConstrs(vm_lifetime[0, m] == vms[m].lifetime for m in range(NUM_VM))
        # model.addConstrs(vm_lifetime[t, m] == vm_lifetime[t+1, m] for t in range(1, TIME-1) for m in range(NUM_VM))

        # complete constraints: bound complete[t, m] and compl_time[m]
        # model.addConstrs((complete[t, m] == 1) >> (exec_time[t, m] >= vms[m].lifetime / interval) for t in range(TIME) for m in range(NUM_VM))
        # model.addConstrs((complete[t, m] == 1) >> (exec_time[t, m] >= vms[m].pred_lifetime / interval) for t in range(1, TIME) for m in range(NUM_VM))
        # model.addConstrs((complete[0, m] == 1) >> (exec_time[0, m] >= vms[m].lifetime / interval) for m in range(NUM_VM))
        model.addConstrs((complete[t, m] == 1) >> (exec_time[t, m] >= vms[m].pred_lifetime / interval) for t in range(TIME) for m in range(NUM_VM))
        # model.addConstrs((complete[t, m] == 1) >> (exec_time[t, m] >= vm_lifetime[t, m] / interval) for t in range(TIME) for m in range(NUM_VM))
        model.addConstrs((complete[t, m] == 1) >> (exec_time[t, m] >= exec_time[t-1, m] + 1) for t in range(1,TIME) for m in range(NUM_VM))

        model.addConstrs(gp.quicksum(complete[t, m] for t in range(TIME+1)) == 1 for m in range(NUM_VM))
        model.addConstrs(gp.quicksum(complete[t, m] * (t + 1) for t in range(TIME+1)) + cur_timestamp - vms[m].start_time / interval == compl_time[m] for m in range(NUM_VM))
        model.addConstrs(gp.quicksum(complete[t, m] for t in range(TIME)) == has_completed[m] for m in range(NUM_VM))

        model.Params.TIME_LIMIT = MIP_OPT_TIME
        model.Params.MIPGap = MIP_OPT_GOAL
        model.Params.MIPFocus = 2
        model.Params.NonConvex = 2
        model.Params.Presolve = 0
        model.optimize()
        
        try:
            placement_array = np.round(np.array([[[placement[t, m, n].x for n in range(NUM_SITE)] for m in range(NUM_VM)] for t in range(TIME+1)])).astype(int)
            booted_vm_array = np.round(np.array([[[booted_vms[t, m, n].x for n in range(NUM_SITE)] for m in range(NUM_VM)] for t in range(TIME)])).astype(int)
            nr_power_array = np.array([[nr_power[t, n].x for n in range(NUM_SITE)] for t in range(TIME)], dtype=float)
            r_power_array = np.array([[min(vm_overhead[0, n].x + migrate_in_overhead[0, n].x + migrate_out_overhead[0, n].x, predicted_energy[n, cur_timestamp]) for n in range(NUM_SITE)]], dtype=float)
            migration_array = np.round(np.array([[[[migration[t, m, n1, n2].x for n2 in range(NUM_SITE) ] for n1 in range(NUM_SITE)] for m in range(NUM_VM)] for t in range(TIME)])).astype(int)
            # mispredicted_vms = np.array([1 if exec_time[0, m].x >= vms[m].pred_lifetime / interval and complete[0, m].x == 0 else 0 for m in range(NUM_VM)])
        except:
            return None, None, None, None, None

        # print("VM Placement", placement_array.transpose())
        # print("BOOTED VMs", booted_vm_array.transpose())
        # print("Exec time", np.array([[exec_time[t, m].x for t in range(TIME)] for m in range(NUM_VM)]))
        # print("Complete", np.array([[complete[t, m].x for t in range(TIME+1)] for m in range(NUM_VM)]))
        # print("Progress", np.array([[progress[t, m].x for t in range(TIME)] for m in range(NUM_VM)]))
        # print("NR Power", np.array(nr_power_array))
        # print("Avail Per VM", np.array([avail_per_vm[m].x for m in range(NUM_VM)]))
        # print("Complete Time", np.array([compl_time[m].x for m in range(NUM_VM)]))
        # print("Has Completed", np.array([has_completed[m].x for m in range(NUM_VM)]))
        # print('Remaining Energy', np.array([[predicted_energy[n, cur_timestamp+t] + nr_power[t, n].x -  vm_overhead[t, n].x + migrate_in_overhead[t, n].x + migrate_out_overhead[t, n].x for n in range(NUM_SITE)] for t in range(TIME)]))
        # print('Migration Energy', np.sum(np.array([[ migrate_in_overhead[t, n].x + migrate_out_overhead[t, n].x for n in range(NUM_SITE)] for t in range(TIME)])))
        # print('Migration Energy', np.array([[ migrate_in_overhead[t, n].x + migrate_out_overhead[t, n].x for n in range(NUM_SITE)] for t in range(TIME)]))
        # print('Mispredicted vms', mispredicted_vms)
        # print("vm lifetime", np.array([[vm_lifetime[t, m].x for t in range(TIME)] for m in range(NUM_VM)]))
        # print(f"Objective: {model.objVal}")
        
        # return placement_array, booted_vm_array, migration_array, nr_power_array, mispredicted_vms
        return placement_array, booted_vm_array, migration_array, nr_power_array, r_power_array

    


class GlobalProfile():
    def __init__(self, T):
        self.migration_distribution = [[] for _ in range(T)]
        self.migrated_vms = [0 for _ in range(T)]
        self.nr_energy_used = [0 for _ in range(T)]
        self.r_energy_used = [0 for _ in range(T)]
        self.low_priority_nr_energy_used = [0 for _ in range(T)]
        self.low_priority_r_energy_used = [0 for _ in range(T)]
    
    def __str__(self):
        return f"Total Migrations: {sum(self.migrated_vms)}, Total NR Energy: {sum(self.nr_energy_used)}, Total R Energy: {sum(self.r_energy_used),}, Low Priority NR Energy: {sum(self.low_priority_nr_energy_used)}, Low Priority R Energy: {sum(self.low_priority_r_energy_used)}"

@ray.remote
class GreenBoxSim():
    # def __init__(self, policy, energy, predicted_energy, vms, num_rMDCs, opt_time, opt_goal, slo_avail, max_timestamps=-1, interval=60, subgraph_id=-1):
    def __init__(self, policy, energy, predicted_energy, vms, num_rMDCs, opt_time, opt_goal, slo_avail, max_timestamps=-1, interval=60, subgraph_id=-1, lahead=5):
        self.policy = policy
        self.energy = energy
        self.predicted_energy = predicted_energy
        self.vms = vms
        self.num_rMDCs = num_rMDCs
        self.max_timestamps = max_timestamps
        self.subgraph_id = subgraph_id
        self.interval = interval

        self.T, self.M, self.N = self.max_timestamps + 1, len(self.vms), self.num_rMDCs
        self.vm_placement = np.zeros((self.T+1, self.M, self.N))
        self.running_vms = np.zeros((self.T, self.M, self.N))
        self.migration = np.zeros((self.T, self.M, self.N, self.N))
        self.nr_energy = np.zeros((self.T, self.N))
        self.r_energy = np.zeros((self.T, self.N))
        self.completed_vms = []
        self.vm_profiles = [VMProfile(self.vms[m], self.T) for m in range(self.M)]
        self.global_profile = GlobalProfile(self.T)
        self.utilization = np.zeros(self.T)

        self.timestamp = 0
                
        # TODO: this is important to the queuing delay, as MIP will NOT schedule any VMs to an app if it cannot finish 
        if self.policy == "mip":
            # self.lookahead = 5 # self.T // 2
            self.lahead = lahead
            self.lookahead = lahead # self.T // 2
        elif self.policy == "mip-app":
            self.lookahead = 2
        elif self.policy == "greedy":
            self.lookahead = 1
        else:
            self.lookahead = 1
        
        self.lookahead_t = 0

        self.opt_time = opt_time
        self.opt_goal = opt_goal
        self.slo_avail = slo_avail
        

    def greedy_decision(self, mode="NR"):
        T, M, N, t = self.T, self.M, self.N, self.timestamp
        self.lookahead_t = min(self.max_timestamps, self.timestamp+self.lookahead)
        vm_placement_array = np.zeros((2, self.M, self.N))
        booted_vm_array = np.zeros((1, self.M, self.N))
        migration_array = np.zeros((1, self.M, self.N, self.N))
        nr_energy_array = np.zeros((1, self.N))
        r_energy_array = np.zeros((1, self.N))

        if t == 0:
            vm_placement_array[0, :, :] = np.zeros((M, N))
            booted_vm_array = vm_placement_array[0:1, :, :].copy()
        else:
            vm_placement_array[0, :, :] = self.vm_placement[t, :, :]
            booted_vm_array = vm_placement_array[0:1, :, :].copy()
            
        booted_vm_ids = [m for m in range(self.M) if sum(booted_vm_array[0, m, :]) == 1]
        new_vm_ids = [m for m in range(self.M) if sum(vm_placement_array[0, m, :]) == 0]
        # print(booted_vm_ids, new_vm_ids)
        
        low_prio_avail = [vm.profile.avail() for vm in self.completed_vms if vm.priority == 1]
        if len(low_prio_avail) > 0:
            global_avail = np.average(low_prio_avail)
        else:
            global_avail = 0
        num_complete_low_prio_vm = len(low_prio_avail)
        num_low_prio_vm = len([vm for vm in self.vms if vm.priority == 1])
        # global_avail = np.sum([vm.profile.avail() for vm in self.completed_vms])
        total_target_avail = self.slo_avail * (num_complete_low_prio_vm + num_low_prio_vm)
        vm_target_avail = total_target_avail - global_avail
        projected_vm_avail_distribution = dict()
        for m, vm in enumerate(self.vms):
            if vm.priority == 0:
                continue
            if m in booted_vm_ids:
                avail = vm.profile.predict_avail(extra_exec=self.interval, extra_overhead=0)
            else:
                if self.timestamp < vm.start_time / self.interval:
                    avail = 1
                else:
                    avail = vm.profile.predict_avail(extra_exec=0, extra_overhead=self.interval)
            projected_vm_avail_distribution[m] = avail
        
        if mode == "NR":
            vm_placement_array[1, :, :] = vm_placement_array[0, :, :]
            # 1. Schedule all new VMs for the next timestamp, starting from the site with the most remaining energy
            remaining_new_vms = len(new_vm_ids)
            for n in range(self.N):
                remaining_energy = self.energy[n, self.timestamp] - sum(vm_placement_array[0, m, n] * self.vms[m].maxpower[self.timestamp] for m in range(self.M))
                while remaining_energy > 0 and remaining_new_vms > 0:
                    new_vm_id = new_vm_ids.pop(0)
                    vm_placement_array[1, new_vm_id, n] = 1
                    remaining_new_vms -= 1
                    remaining_energy -= self.vms[new_vm_id].maxpower[t]
            while remaining_new_vms > 0:
                new_vm_id = new_vm_ids.pop(0)
                vm_placement_array[1, new_vm_id, remaining_new_vms % self.N - 1] = 1
                remaining_new_vms -= 1
            # remaining_new_vms = len(new_vm_ids)
            # for n in range(self.N):
            #     remaining_energy = self.energy[n, self.timestamp] - sum(vm_placement_array[0, m, n] * self.vms[m].avg_power for m in range(self.M))
            #     while remaining_energy > 0 and remaining_new_vms > 0:
            #         new_vm_id = new_vm_ids.pop(0)
            #         vm_placement_array[1, new_vm_id, n] = 1
            #         remaining_new_vms -= 1
            #         remaining_energy -= self.vms[new_vm_id].avg_power
            # while remaining_new_vms > 0:
            #     new_vm_id = new_vm_ids.pop(0)
            #     vm_placement_array[1, new_vm_id, remaining_new_vms % self.N - 1] = 1
            #     remaining_new_vms -= 1

            # 2. Migrate as many VMs as we can from the sites with insufficient power to the sites with exccessive power 
            if True:
                has_remaining_energy = [self.energy[n, self.timestamp] - sum(booted_vm_array[0, m, n] * self.vms[m].maxpower[self.timestamp] for m in range(self.M)) for n in range(self.N)]
                sites_insufficient_energy = {n:has_remaining_energy[n] for n in range(self.N) if has_remaining_energy[n] <= 0}
                sites_excessive_energy = {n:has_remaining_energy[n] for n in range(self.N) if has_remaining_energy[n] > 0}
                print(sites_excessive_energy)
                booted_vm_ids_in_negative_power_sites = [victim_vm_id for victim_vm_id in booted_vm_ids if np.where(booted_vm_array[0, victim_vm_id, :] == 1)[0] in list(sites_insufficient_energy.keys()) and self.vms[victim_vm_id].priority == 1]
                failure_count, failure_threshold = 0, 10
                while len(sites_insufficient_energy) > 0 and failure_count <= failure_threshold and len(booted_vm_ids_in_negative_power_sites) > 0:
                    # decide a vm to migrate and compute the power difference
                    migrated_vm_id = random.choice(booted_vm_ids_in_negative_power_sites)
                    vm_power = self.vms[migrated_vm_id].maxpower[self.timestamp]
                    migration_out_power = self.vms[migrated_vm_id].memory * MIGRATE_OUT_CONST_PER_GB
                    migration_in_power = self.vms[migrated_vm_id].memory * MIGRATE_IN_CONST_PER_GB
                    # For the NEXT timestamp, we have source_site_power -= vm_power - migration_power
                    # AND target_site_power += vm_power + migration_power
                    # decide a site to migrate onto
                    source_site = np.where(booted_vm_array[0, migrated_vm_id, :] == 1)[0][0]
                    candidate_sites = [s for s, e in sites_excessive_energy.items() if e >= vm_power + migration_in_power]
                    # a counter for the number of consecutive failed identification of a candidate site; reset the counter for any success
                    if len(candidate_sites) == 0:
                        failure_count += 1
                        continue
                    else:
                        failure_count = 0
                    target_site = random.choice(candidate_sites)
                    vm_placement_array[1, migrated_vm_id, target_site] = 1 
                    vm_placement_array[1, migrated_vm_id, source_site] = 0 

                    # let's compute everything again at the end
                    # has_remaining_energy[source_site] += vm_power - migration_out_power
                    has_remaining_energy[target_site] -= vm_power - migration_out_power
                    ssites_insufficient_energy = {n:has_remaining_energy[n] for n in range(self.N) if has_remaining_energy[n] <= 0}
                    sites_excessive_energy = {n:has_remaining_energy[n] for n in range(self.N) if has_remaining_energy[n] > 0}
                    booted_vm_ids_in_negative_power_sites = [victim_vm_id for victim_vm_id in booted_vm_ids if np.where(booted_vm_array[0, victim_vm_id, :] == 1)[0] in list(sites_insufficient_energy.keys()) and self.vms[victim_vm_id].priority == 1]

            # 3. Randomly shutdown VMs until it's below the vm_target_avail
            has_remaining_energy = [self.energy[n, self.timestamp] - sum(booted_vm_array[0, m, n] * self.vms[m].maxpower[self.timestamp] for m in range(self.M)) > 0 for n in range(self.N)]
            sites_wo_remaining_energy = [n for n in range(self.N) if not has_remaining_energy[n]]
            booted_vm_ids_in_negative_power_sites = [victim_vm_id for victim_vm_id in booted_vm_ids if np.where(booted_vm_array[0, victim_vm_id, :] == 1)[0] in sites_wo_remaining_energy and self.vms[victim_vm_id].priority == 1]
            while len(booted_vm_ids_in_negative_power_sites) > 0:
                victim_vm_id = random.choice(booted_vm_ids_in_negative_power_sites)
                projected_vm_avail_distribution[victim_vm_id] = vm.profile.predict_avail(extra_exec=0, extra_overhead=self.interval)
                if sum(projected_vm_avail_distribution.values()) <= vm_target_avail:
                    projected_vm_avail_distribution[victim_vm_id] = vm.profile.predict_avail(extra_exec=self.interval, extra_overhead=0)
                    break
                booted_vm_array[0, victim_vm_id, :] = np.zeros(self.N)   
                has_remaining_energy = [self.energy[n, self.timestamp] - sum(booted_vm_array[0, m, n] * self.vms[m].maxpower[self.timestamp] for m in range(self.M)) > 0 for n in range(self.N)]
                sites_wo_remaining_energy = [n for n in range(self.N) if not has_remaining_energy[n]]
                booted_vm_ids_in_negative_power_sites = [victim_vm_id for victim_vm_id in booted_vm_ids if np.where(booted_vm_array[0, victim_vm_id, :] == 1)[0] in sites_wo_remaining_energy and self.vms[victim_vm_id].priority == 1]
                # booted_vm_ids_in_negative_power_sites.remove(victim_vm_id)
            # has_remaining_energy = [self.energy[n, self.timestamp] - sum(booted_vm_array[0, m, n] * self.vms[m].avg_power for m in range(self.M)) > 0 for n in range(self.N)]
            # sites_wo_remaining_energy = [n for n in range(self.N) if not has_remaining_energy[n]]
            # booted_vm_ids_in_negative_power_sites = [victim_vm_id for victim_vm_id in booted_vm_ids if np.where(booted_vm_array[0, victim_vm_id, :] == 1)[0] in sites_wo_remaining_energy and self.vms[victim_vm_id].priority == 1]
            # while len(booted_vm_ids_in_negative_power_sites) > 0:
            #     victim_vm_id = random.choice(booted_vm_ids_in_negative_power_sites)
            #     projected_vm_avail_distribution[victim_vm_id] = vm.profile.predict_avail(extra_exec=0, extra_overhead=self.interval)
            #     # print(projected_vm_avail_distribution, sum(projected_vm_avail_distribution), vm_target_avail, self.vm_profiles)
            #     # if sum(projected_vm_avail_distribution.values()) < vm_target_avail:
            #     if sum(projected_vm_avail_distribution.values()) <= vm_target_avail:
            #         projected_vm_avail_distribution[victim_vm_id] = vm.profile.predict_avail(extra_exec=self.interval, extra_overhead=0)
            #         break
            #     booted_vm_array[0, victim_vm_id, :] = np.zeros(self.N)   
            #     has_remaining_energy = [self.energy[n, self.timestamp] - sum(booted_vm_array[0, m, n] * self.vms[m].avg_power for m in range(self.M)) > 0 for n in range(self.N)]
            #     sites_wo_remaining_energy = [n for n in range(self.N) if not has_remaining_energy[n]]
            #     booted_vm_ids_in_negative_power_sites = [victim_vm_id for victim_vm_id in booted_vm_ids if np.where(booted_vm_array[0, victim_vm_id, :] == 1)[0] in sites_wo_remaining_energy and self.vms[victim_vm_id].priority == 1]
            #     # booted_vm_ids_in_negative_power_sites.remove(victim_vm_id)


            # 4. calculate NR energy used at each site   
            for n in range(self.N):
                demand = sum(booted_vm_array[0, m, n] * self.vms[m].maxpower[self.timestamp] for m in range(self.M))
                supply = self.energy[n, self.timestamp] 
                remaining_energy = supply - demand
                nr_energy_array[0, n] = max(0, -remaining_energy)
                r_energy_array[0,n] = min(demand, supply)
            # for n in range(self.N):
            #     remaining_energy = self.energy[n, self.timestamp] - sum(booted_vm_array[0, m, n] * self.vms[m].avg_power for m in range(self.M)) 
            #     nr_energy_array[0, n] = max(0, -remaining_energy)     

        # return vm_placement_array, booted_vm_array, migration_array, nr_energy_array, []
        return vm_placement_array, booted_vm_array, migration_array, nr_energy_array, r_energy_array


    def mip_decision(self, hints=None, backlog=False, has_new_apps=True):

        
        # if not has_new_apps and self.lookahead_t > self.timestamp and self.lookahead > 3:
            # return None
        
        self.lookahead_t = min(self.max_timestamps, self.timestamp+self.lookahead)
        self.lookahead = self.lookahead_t - self.timestamp
        if len(self.completed_vms) > 0:
            low_prio_avail = [vm.profile.avail() for vm in self.completed_vms if vm.priority == 1]
            if len(low_prio_avail) > 0:
                global_avail = np.average(low_prio_avail)
            else:
                global_avail = 0
            num_complete_vm = len(low_prio_avail)
        else:
            global_avail = 0
            num_complete_vm = 0
        
        init_states = {"init_placement": self.vm_placement[self.timestamp, :, :], "init_progress": [self.vm_profiles[m].exec for m in range(self.M)], "global_exec": sum([vm.profile.exec for vm in self.completed_vms+self.vms]), "global_lifetime": sum([vm.profile.lifetime for vm in self.completed_vms+self.vms]), "num_complete_vm": num_complete_vm, "global_avail": global_avail}
        
        ret = NEW_MIP.migrate(self.timestamp, self.lookahead, self.interval, self.energy, self.predicted_energy, self.vms, self.num_rMDCs, SLO_AVAIL = self.slo_avail, MIP_OPT_TIME = self.opt_time, MIP_OPT_GOAL = self.opt_goal, init_states=init_states, hints=hints, policy=self.policy, subgraph_id = self.subgraph_id)
        
        # assert(ret is not None)
        
        return ret
        



    # @input new_vms: all newly scheduled VMs
    # @input cloned: whether we are in the "clone" mode to pre-compute a less optimized schedule with greedy policy
    # @return None
    def migrate(self, new_vms=None, cloned=False, get_some_hints=False):
        # handle incoming vms
        if new_vms and len(new_vms) > 0:
            self.vms += new_vms
            self.M += len(new_vms)
            self.vm_placement = np.pad(self.vm_placement, ((0,0),(0,len(new_vms)),(0,0)), mode='constant')
            self.running_vms = np.pad(self.running_vms, ((0,0),(0,len(new_vms)),(0,0)), mode='constant')
            self.migration = np.pad(self.migration, ((0,0),(0,len(new_vms)),(0,0),(0,0)), mode='constant')
            self.vm_profiles += [VMProfile(new_vms[i], self.T) for i in range(len(new_vms))]
        
        if len(self.vms) == 0:
            self.timestamp += 1
            print("this subgraph has No VMs")
            return
        
        # back to greedy flag
        back_to_greedy = False

        # now we make decisions given the specified scheduling policy
        update_global_vars = False
        if self.policy == "greedy":
            # vm_placement, running_vms, migration, nr_energy, mispredicted_vms = self.greedy_decision()
            vm_placement, running_vms, migration, nr_energy, r_energy = self.greedy_decision()
            update_global_vars = True
            
        elif self.policy == "mip" or self.policy == "mip-app":
            # first "clone" a greedy scheduler to help us make some decisions
            cloned_scheduler = copy.deepcopy(self)
            cloned_scheduler.policy = "greedy"
            cloned_scheduler.energy = self.predicted_energy
            cloned_scheduler.lookahead = 1
            lookahead = min(self.lookahead, self.max_timestamps + 1 - self.timestamp)
            for i in range(lookahead):
                cloned_scheduler.migrate(cloned=True)
            placement_hint = cloned_scheduler.vm_placement[self.timestamp:self.timestamp+lookahead, :, :]
            vm_hint = cloned_scheduler.running_vms[self.timestamp:self.timestamp+lookahead, :, :]
            migr_hint = cloned_scheduler.migration[self.timestamp:self.timestamp+lookahead, :, :, :]
            nr_hint = cloned_scheduler.nr_energy[self.timestamp:self.timestamp+lookahead, :]
            r_hint = cloned_scheduler.r_energy[self.timestamp:self.timestamp+lookahead, :]   
            if get_some_hints: 
                hints = [placement_hint, vm_hint, migr_hint, nr_hint]
            else:
                hints = None
            # then let's use MIP to make decision!
            ret = self.mip_decision(hints=hints, backlog=False, has_new_apps=(new_vms and len(new_vms) > 0))
            if ret is None:
                update_global_vars = False
            else:
                # vm_placement, running_vms, migration, nr_energy, mispredicted_vms = ret
                vm_placement, running_vms, migration, nr_energy, r_energy = ret
                update_global_vars = True
            
                # if failed (in very rare cases), we fall back to use greedy
                if running_vms is None:
                    log_msg("MIP failed; Fall back to greedy")
                    self.lookahead = 2
                    ret = self.mip_decision(hints=hints, backlog=False, has_new_apps=(new_vms and len(new_vms) > 0))
                    vm_placement, running_vms, migration, nr_energy, r_energy = ret
                    if running_vms is None:
                        self.lookahead = 1
                        vm_placement, running_vms, migration, nr_energy, r_energy = self.greedy_decision()
                    back_to_greedy = True
        
        ''' Migration '''
        migration_source_total_mem = defaultdict(int)
        migration_dest_total_mem = defaultdict(int)
        migrated_vms = defaultdict(lambda: None)
        if self.timestamp > 0:
            for m in range(self.M):
                if np.max(migration[0, m, :, :]) > 0:
                    source_rMDC, dest_rMDC = tuple(np.argwhere(migration[0, m, :, :] == 1)[0])
                    assert(vm_placement[0, m, source_rMDC] == 1)
                    assert(vm_placement[1, m, dest_rMDC] == 1)
                    migration_source_total_mem[source_rMDC] += self.vms[m].memory
                    migration_dest_total_mem[dest_rMDC] += self.vms[m].memory
                    migrated_vms[m] = (source_rMDC, dest_rMDC)
        self.global_profile.migrated_vms[self.timestamp] = len(migrated_vms)
        
        ''' Power '''
        vm_power = np.matmul(np.array([self.vms[m].maxpower[self.timestamp] for m in range(self.M)]), running_vms[0, :, :])
        low_priority_vm_power = np.matmul(np.array([self.vms[m].maxpower[self.timestamp] for m in range(self.M) if self.vms[m].priority == 1]), [running_vms[0, m, :] for m in range(self.M) if self.vms[m].priority == 1])
        # log_msg(f"vm power: {vm_power}, low priority vm power:{low_priority_vm_power}")
        migration_power_as_source, migration_power_as_dest = np.zeros(self.N), np.zeros(self.N)
        for n in range(self.N):
            migration_power_as_source[n] = MIGRATE_IN_CONST_PER_GB * migration_source_total_mem[n]
            migration_power_as_dest[n] =  MIGRATE_OUT_CONST_PER_GB * migration_dest_total_mem[n]
        # vm_power = np.matmul(np.array([self.vms[m].avg_power for m in range(self.M)]), running_vms[0, :, :])
        # migration_power_as_source, migration_power_as_dest = np.zeros(self.N), np.zeros(self.N)
        # for n in range(self.N):
        #     migration_power_as_source[n] = MIGRATE_IN_CONST_PER_GB * migration_source_total_mem[n]
        #     migration_power_as_dest[n] =  MIGRATE_OUT_CONST_PER_GB * migration_dest_total_mem[n]
        
        power_consumption = vm_power + migration_power_as_source + migration_power_as_dest
        # TODO: add misprediction handling here
        # log_msg(f"vm power on each site:{vm_power}, migration power as source:{migration_power_as_source}, migration power as dest:{migration_power_as_dest}, power on this site: {self.energy[:, self.timestamp].transpose()}, predicted power on this site: {self.predicted_energy[:, self.timestamp].transpose()}, non-renewable power:{self.nr_energy[self.timestamp, :]}")
        for n in range(self.N):
            if (power_consumption[n] > self.energy[n, self.timestamp] + nr_energy[0, n] + EPSILON):
                nr_energy[0, n] = power_consumption[n] - self.energy[n, self.timestamp]
                log_msg(f"power misprediction handler: site {n}: power_consumption {power_consumption[n]}, energy {self.energy[n, self.timestamp]}, extra nr energy needed {nr_energy[0, n]}")
            assert(power_consumption[n] <= self.energy[n, self.timestamp] + nr_energy[0, n] + EPSILON)
        self.global_profile.nr_energy_used[self.timestamp] = np.sum(nr_energy[0, :])     
        self.global_profile.r_energy_used[self.timestamp] = np.sum(r_energy[0, :])
        if np.sum(vm_power) > 0.:
            self.global_profile.low_priority_nr_energy_used[self.timestamp] = np.sum(nr_energy[0, :]) * np.sum(low_priority_vm_power) / np.sum(vm_power)
            self.global_profile.low_priority_r_energy_used[self.timestamp] = np.sum(r_energy[0, :]) * np.sum(low_priority_vm_power) / np.sum(vm_power)
        else:
            self.global_profile.low_priority_nr_energy_used[self.timestamp] = 0.
            self.global_profile.low_priority_r_energy_used[self.timestamp] = 0.
            

        ''' Compute '''
        progress = np.sum(running_vms[0, :, :] * self.interval, axis=1)

        ''' Update VM Profiles '''
        for m in range(self.M):
            if self.vms[m].start_time <= self.timestamp * self.interval:
                self.vm_profiles[m].exec += progress[m]
                self.vm_profiles[m].queuing += 0
                self.vm_profiles[m].overhead += 0
                self.vm_profiles[m].interrupt += self.interval - progress[m]
            self.vm_profiles[m].migration_log[self.timestamp] = [migrated_vms[m]]
                
        self.global_profile.migration_distribution[self.timestamp] =[(migration_source_total_mem, migration_dest_total_mem)]

        ''' Wrap UP Completed VMs '''
        # TODO: remove finished VMs from global vars
        if not cloned:
            for m in reversed(range(self.M)):
                # Lifetime misprediction handling
                if self.vm_profiles[m].lifetime_mispredict():
                    log_msg(f"vm lifetime misprediction handling")
                    self.vms[m].pred_lifetime = max(self.vms[m].pred_lifetime + 3, 48)
                    

                if self.vm_profiles[m].has_finished():
                    self.vms[m].profile.finished = True
                    self.vms[m].profile.end = (self.timestamp + 1) * self.interval
                    self.completed_vms.append(self.vms[m])
                    self.vms[m].pred_lifetime = self.vms[m].pred_origin
                    self.vms.pop(m)
                    self.vm_profiles.pop(m)

                    vm_placement = np.delete(vm_placement, m, axis=1)
                    running_vms = np.delete(running_vms, m, axis=1)
                    migration = np.delete(migration, m, axis=1)
                    self.vm_placement = np.delete(self.vm_placement, m, axis=1)
                    self.running_vms = np.delete(self.running_vms, m, axis=1)
                    self.migration = np.delete(self.migration, m, axis=1)
        
                

        # set the global variables of the simulator with the scheduler decisions 
        if update_global_vars:
            self.vm_placement[self.timestamp:self.timestamp+self.lookahead+1, :, :] = vm_placement[0:self.lookahead+1, :, :]
            self.running_vms[self.timestamp:self.timestamp+self.lookahead, :, :] = running_vms[0:self.lookahead, :, :]
            self.migration[self.timestamp:self.timestamp+self.lookahead, :, :, :] = migration[0:self.lookahead, :, :, :]
            self.nr_energy[self.timestamp:self.timestamp+self.lookahead, :] = nr_energy[0:self.lookahead, :]
            if back_to_greedy:
                self.lookahead = 4
             

        self.M = len(self.vms)
        log_msg(self.global_profile)

        # dump some logs
        # if not cloned:
        #     if self.policy == "mip":
        #         log_name = f"logs/mip_sim_{self.subgraph_id}.log"
        #     elif self.policy == "mip-app":
        #         log_name = f"logs/mip_app_sim_{self.subgraph_id}.log"
        #     else:
        #         log_name = f"/home/js39/software/virtual-battery/logs/sim_{self.subgraph_id}.log"
        #     if self.timestamp == 0:
        #         mode = "w+"
        #     else:
        #         mode = "a+"
        #     with open(log_name, mode) as log_f:
        #         print("SIM_G: ", self.subgraph_id, file=log_f)
        #         print("SIM_T: ", self.timestamp, file=log_f)
        #         print("SIM_P: ", format_numpy((time_avail_compute)), file=log_f)
        #         print("SIM_DIFF", format_numpy([max(diff_vms[i,:]) for i in range(self.M)]), file=log_f)
        #         print("SIM_REST_P: ", format_numpy(self.progress / self.interval), file=log_f)
        #         log_f.flush()
                
        self.timestamp += 1

        return

    # for the greedy model
    # @property
    def remaining_energy(self, target_util=1.0, t=None):
        power_per_active_vms = [vm.avg_power for vm in self.vms]
        vm_to_progress = [(vmp.lifetime - vmp.exec) / self.interval for vmp in self.vm_profiles]
        power = np.matmul(power_per_active_vms, self.vm_placement[self.timestamp, :, :])
        for j in range(self.N):
            assert(power[j] >= 0)
        if t is None:
            t = self.timestamp
        # remaining_power_per_site = self.energy[:, t] * target_util - power
        remaining_power_per_site = self.predicted_energy[:, t] * target_util - power
        return remaining_power_per_site

        # evenly distribute the power of hibernated vms onto sites
        potential_power = sum(power_per_active_vms[i] * vm_to_progress[i] for i in range(self.M))
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
        predicted_remaining_energy = np.zeros((self.N, self.T))
        for t in range(self.timestamp, self.T):
            predicted_remaining_energy[:, t] = self.remaining_energy(target_util, t)

        return predicted_remaining_energy

    def finish(self):
        if len(self.vms) + len(self.completed_vms) == 0:
            return 24, 0, [], [], self.vm_profiles, self.global_profile, self.utilization, self.running_vms

        naturally_completed = len(self.completed_vms)
        
        for vm in self.vms:
            vm.profile.end = self.timestamp * self.interval
            
        if len(self.completed_vms) > 0:
            avg_t = np.average(list([vm.profile.end - vm.profile.start for vm in self.completed_vms]))  
            dist_t = list([vm.profile.end - vm.profile.start for vm in self.completed_vms])  
        else:
            avg_t = 0.0    
            dist_t = []

        all_profiles = list(filter(lambda x:x.start+x.lifetime<=self.timestamp*self.interval, [vm.profile for vm in self.completed_vms+self.vms]))
        # print(all_profiles)
        return self.timestamp, avg_t, dist_t, self.completed_vms, all_profiles, self.global_profile, self.utilization, self.running_vms




if __name__ == "__main__":
    traces = pd.DataFrame()
    traces['A'] = [0, 100, 100, 0, 0, 20, 20, 0, 0]
    traces['B'] = [0, 0, 0, 50, 0, 0, 0, 0, 0]
    traces['C'] = [0, 0, 0, 50, 60, 100, 100, 100, 100]
    traces = np.transpose(traces.to_numpy())
    
    VM_A = VM("a", cores=16, memory=128, lifetime=6*HOURS, start_time=1*HOURS)
    vms = []
    for _ in range(5):
        vm = copy.deepcopy(VM_A)
        vm.id = _
        vms.append(vm)

    
    sim = GreenBoxSim("mip", traces, traces, vms, 3, opt_time=1200, opt_goal=0.01, slo_avail=0.95, max_timestamps=9, interval=60)
    
    for _ in range(9):
        sim.migrate(get_some_hints=False)        
        
        
    # traces = pd.DataFrame()
    # traces['A'] = [0, 0, 0]
    # traces['B'] = [0, 0, 0]
    # traces['C'] = [0, 100, 100]
    # traces = np.transpose(traces.to_numpy())
    
    # VM_A = VM("a", cores=16, memory=128, lifetime=2*HOURS, avg_power=18, peak_power=30, start_time=1*HOURS)
    # vms = [copy.deepcopy(VM_A) for _ in range(1)]
    # sim = GreenBoxSim("mip", traces, traces, vms, 3, 0, opt_time=1200, opt_goal=0.01, slo_avail=0.95, max_timestamps=3, interval=60)
        
    # for _ in range(3):
    #     sim.migrate(get_some_hints=False)
    
        
    
    # model = NEW_MIP()
    # model.migrate(0, 7, 60, traces, traces, vms, 3, SLO_AVAIL= 0.99, MIP_OPT_TIME=1200, MIP_OPT_GOAL=0.01, init_vms=None, init_progress=None, init_interrupt=None, hint=None, policy="mip", subgraph_id=0)
    
