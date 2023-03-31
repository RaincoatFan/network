###############################################################################
# Python code of AHA.                                                         #
# The python code is based on the following papers:                           #
# W. Zhao, L. Wang and S. Mirjalili, Artificial hummingbird algorithm: A      #
# new bio-inspired optimizer with its engineering applications, Computer      #
# Methods in Applied Mechanics and Engineering (2021) 114194, https:          #
# //doi.org/10.1016/j.cma.2021.114194.                                        #
###############################################################################
import numpy as np
from torch import randperm
from matplotlib.pyplot import *
from pylab import *
from net import generation

def ben_functions(x, function_index):
    # Sphere
    if function_index == 1:
        s = generation(x)
    return s


def space_bound(X, Up, Low):
    dim = len(X)
    S = (X > Up) + (X < Low)
    res = (np.random.rand(dim) * (np.array(Up) - np.array(Low)) + np.array(Low)) * S + X * (~S)
    return res

def initializationSineBack(npop,dim,ub,lb,fun_index):
    Boundary_no = dim
    pop_pos = np.zeros((2 * npop, dim))
    for i in range(npop):
        for j in range(dim):
            x0 = np.random.uniform(low=0.0, high=1.0, size=None)
            R = math.sin(math.pi * x0)
            pop_pos[i, j] = math.floor(R * (ub[j] - lb[j]) + lb[j])
            pop_pos[npop + i, j] = math.floor(ub[j] + lb[j] - pop_pos[i, j])
    pop_fit = np.zeros(2 * npop)
    for i in range(2 * npop):
        print('初始',i)
        pop_fit[i] = ben_functions(pop_pos[i, :], fun_index)
    index = np.argsort(pop_fit)
    pop_pos = pop_pos[index[:npop]]
    return pop_pos

def levy(dim):
    beta = 3 / 2
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
                math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(dim) * sigma;
    v = np.random.randn(dim);
    step = u / abs(v) ** (1 / beta);
    o = step;
    return o

def AHA(fun_index, max_it, npop):
    # 改
    lb, ub, dim = [0],[88.9],4611
    if len(lb) == 1:
        lb = lb * dim
        ub = ub * dim
    pop_pos = np.zeros((npop, dim))
#新的初始种群化方法（基于sine混沌映射和反向策略初始化种群）
    # for i in range(dim):
    #     pop_pos[:, i] = np.random.rand(npop) * (ub[i] - lb[i]) + lb[i]
    pop_pos = initializationSineBack(npop,dim,ub,lb,fun_index)

    pop_fit = np.zeros(npop)
    for i in range(npop):
        pop_fit[i] = ben_functions(pop_pos[i, :], fun_index)

    best_f = float('inf')
    best_x = []
    for i in range(npop):
        if pop_fit[i] <= best_f:
            best_f = pop_fit[i]
            best_x = pop_pos[i, :]
    his_best_fit = np.zeros(max_it)
    visit_table = np.zeros((npop, npop))
    diag_ind = np.diag_indices(npop)
    visit_table[diag_ind] = float('nan')
    for it in range(max_it):
        print('迭代',it)
        # Direction
        visit_table[diag_ind] = float('-inf')
        for i in range(npop):
            direct_vector = np.zeros((npop, dim))
            r = np.random.rand()
            # Diagonal flight
            if r < 1 / 3:
                rand_dim = randperm(dim)
                if dim >= 3:
                    rand_num = np.ceil(np.random.rand() * (dim - 2))
                else:
                    rand_num = np.ceil(np.random.rand() * (dim - 1))

                direct_vector[i, rand_dim[:int(rand_num)]] = 1
            # Omnidirectional flight
            elif r > 2 / 3:
                direct_vector[i, :] = 1
            else:
                # Axial flight
                rand_num = ceil(np.random.rand() * (dim - 1))
                direct_vector[i, int(rand_num)] = 1
            # Guided foraging
            if np.random.rand() < 0.5:
                MaxUnvisitedTime = max(visit_table[i, :])
                TargetFoodIndex = visit_table[i, :].argmax()
                MUT_Index = np.where(visit_table[i, :] == MaxUnvisitedTime)
                if len(MUT_Index[0]) > 1:
                    Ind = pop_fit[MUT_Index].argmin()
                    TargetFoodIndex = MUT_Index[0][Ind]
#加入levy飞行///np.random.randn()->levy(dim)
                newPopPos = np.floor(pop_pos[TargetFoodIndex, :] + levy(dim) * direct_vector[i, :] * (
                        pop_pos[i, :] - pop_pos[TargetFoodIndex, :]))
                newPopPos = np.floor(space_bound(newPopPos, ub, lb))
                newPopFit = ben_functions(newPopPos, fun_index)
                if newPopFit < pop_fit[i]:
                    pop_fit[i] = newPopFit
                    pop_pos[i, :] = newPopPos
                    visit_table[i, :] += 1
                    visit_table[i, TargetFoodIndex] = 0
                    visit_table[:, i] = np.max(visit_table, axis=1) + 1
                    visit_table[i, i] = float('-inf')
                else:
                    visit_table[i, :] += 1
                    visit_table[i, TargetFoodIndex] = 0
            else:
                # Territorial foraging
                newPopPos = np.floor(pop_pos[i, :] + np.random.randn() * direct_vector[i, :] * pop_pos[i, :])
                newPopPos = np.floor(space_bound(newPopPos, ub, lb))
                newPopFit = ben_functions(newPopPos, fun_index)
                if newPopFit < pop_fit[i]:
                    pop_fit[i] = newPopFit
                    pop_pos[i, :] = newPopPos
                    visit_table[i, :] += 1
                    visit_table[:, i] = np.max(visit_table, axis=1) + 1
                    visit_table[i, i] = float('-inf')
                else:
                    visit_table[i, :] += 1
        visit_table[diag_ind] = float('nan')
        # Migration foraging
        #引入灰狼优化思想向头狼靠近
        if np.mod(it, npop) == 0:
            visit_table[diag_ind] = float('-inf')
            MigrationIndex = pop_fit.argmax()

            MaxUnvisitedTime = max(visit_table[npop-1, :])
            TargetFoodIndex = visit_table[npop-1, :].argmax()
            MUT_Index = np.where(visit_table[npop-1, :] == MaxUnvisitedTime)
            if len(MUT_Index[0]) > 1:
                Ind = pop_fit[MUT_Index].argmin()
                TargetFoodIndex = MUT_Index[0][Ind]
            # 加入levy飞行///np.random.randn()->levy(dim)
            pop_pos[MigrationIndex, :] = pop_pos[TargetFoodIndex, :] + levy(dim) * direct_vector[npop-1, :] * (
                    pop_pos[MigrationIndex, :] - pop_pos[TargetFoodIndex, :])
            pop_pos[MigrationIndex, :] = np.floor(space_bound(pop_pos[MigrationIndex, :], ub, lb))
            #pop_pos[MigrationIndex, :] = np.random.rand(dim) * (np.array(ub) - np.array(lb)) + np.array(lb)

            visit_table[MigrationIndex, :] += 1
            visit_table[:, MigrationIndex] = np.max(visit_table, axis=1) + 1
            visit_table[MigrationIndex, MigrationIndex] = float('-inf')
            pop_fit[MigrationIndex] = ben_functions(pop_pos[MigrationIndex, :], fun_index)
            visit_table[diag_ind] = float('nan')
        for i in range(npop):
            if pop_fit[i] < best_f:
                best_f = pop_fit[i]
                best_x = pop_pos[i, :]
        his_best_fit[it] = best_f
    return best_x, best_f, his_best_fit


def main():
    # fun_index = 1: Sphere
    # fun_index = 2: Rosenbrock
    # fun_index = 3: Schwefel
    # fun_index = 4: Eggholder
    # fun_index = 5: Branin
    # fun_index = 6: Penalized
    fun_index = 1
    max_it = 50
    npop = 20
    best_x, best_f, his_best_fit = AHA(fun_index, max_it, npop)
    if best_f > 0:
        yscale('log')
        plot(arange(1, max_it + 1), his_best_fit, 'r')
    else:
        plot(arange(1, max_it + 1), his_best_fit, 'r')
    xlim([0, max_it + 1])
    xlabel('Iterations')
    ylabel('Fitness')
    title('F' + str(fun_index))
    show()
    print('The best solution is: ', best_x)
    print('The fitness is: ', best_f)



if __name__ == '__main__':
    main()
