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


def fun_range(fun_index):
    d = 30
    if fun_index == 1:
        l = [-100]
        u = [100]
    elif fun_index == 2:
        l = [-30]
        u = [30]
    elif fun_index == 3:
        l = [-500]
        u = [500]
    elif fun_index == 4:
        l = [-512]
        u = [512]
        d = 2
    elif fun_index == 5:
        l = [-5, 0]
        u = [10, 15]
        d = 2
    elif fun_index == 6:
        l = [-50]
        u = [50]
    return l, u, d


def ben_functions(x, function_index):
    # Sphere
    if function_index == 1:
        s = sum(np.square(x))
    # Rosenbrock
    elif function_index == 2:
        s = sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
    # Schwefel
    elif function_index == 3:
        s = sum(x * np.sin(np.sqrt(abs(x))))
    # Eggholder
    elif function_index == 4:
        s = -(x[1] + 47.0) * np.sin(np.sqrt(abs(x[0] / 2 + (x[1] + 47.0)))) - x[0] * np.sin(
            np.sqrt(abs(x[0] - (x[1] + 47.0))))
    # Branin
    elif function_index == 5:
        s = ((x[1] - 5.1 / (4 * np.pi ** 2) * x[0] ** 2 + 5 / np.pi * x[0] - 6) ** 2 + 10 * (
                1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10)
    # Penalized
    elif function_index == 6:
        a = 10
        k = 100
        m = 4
        Dim = len(x)
        s = (pi / Dim) * (10 * ((np.sin(pi * (1 + (x[0] + 1) / 4))) ** 2) + sum(
            (((x[:-1] + 1) / 4) ** 2) * (1 + 10 * ((np.sin(pi * (1 + (x[1:] + 1) / 4)))) ** 2)) + (
                                  (x[Dim - 1] + 1) / 4) ** 2) + sum(
            k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a)))
    return s


def space_bound(X, Up, Low):
    dim = len(X)
    S = (X > Up) + (X < Low)
    res = (np.random.rand(dim) * (np.array(Up) - np.array(Low)) + np.array(Low)) * S + X * (~S)
    return res


def AHA(fun_index, max_it, npop):
    lb, ub, dim = fun_range(fun_index)
    if len(lb) == 1:
        lb = lb * dim
        ub = ub * dim
    pop_pos = np.zeros((npop, dim))
    for i in range(dim):
        pop_pos[:, i] = np.random.rand(npop) * (ub[i] - lb[i]) + lb[i]
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
                newPopPos = pop_pos[TargetFoodIndex, :] + np.random.randn() * direct_vector[i, :] * (
                        pop_pos[i, :] - pop_pos[TargetFoodIndex, :])
                newPopPos = space_bound(newPopPos, ub, lb)
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
                newPopPos = pop_pos[i, :] + np.random.randn() * direct_vector[i, :] * pop_pos[i, :]
                newPopPos = space_bound(newPopPos, ub, lb)
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
        if np.mod(it, 2 * npop) == 0:
            visit_table[diag_ind] = float('-inf')
            MigrationIndex = pop_fit.argmax()
            pop_pos[MigrationIndex, :] = np.random.rand(dim) * (np.array(ub) - np.array(lb)) + np.array(lb)
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
    max_it = 1000
    npop = 50
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

