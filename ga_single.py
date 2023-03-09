from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot as plt
import math, random


class Population:
    # 种群的设计
    def __init__(self, interval, size, chrom_size, cp, mp, gen_max):
        self.individuals = []  # 个体集合
        self.fitness = []  # 个体适应度集合
        self.selector_probability = []  # 个体选择概率集合
        self.new_individuals = []  # 新一代个体集合

        self.interval = interval  # 变量区间
        self.elitist = {'chromsome': [0, 0], 'fitness': 0, 'age': 0}  # 最佳个体信息
        self.size = size  # 种群所包含的个体数
        self.chromosome_size = chrom_size  # 个体的染色体长度
        self.crossover_probability = cp  # 个体之间的交叉概率
        self.mutation_probability = mp  # 个体之间的变异概率
        self.generation_max = gen_max  # 种群进化的最大世代数
        self.age = 0  # 种群当前所处世代

        # 随机产生初始个体集，并将新一代个体、适应度、选择概率等集合以0值进行初始化
        v = 2**self.chromosome_size - 1  # 染色体长度决定了二进制位数，对应十进制的最大值为v
        for i in range(self.size):
            self.individuals.append([random.randint(0,v), random.randint(0, v)])
            self.new_individuals.append([0, 0])
            self.fitness.append(0)
            self.selector_probability.append(0)

    # 基于轮盘赌的选择
    def decode(self, interval, chromosome):
        '''
        将一个染色体chromosome映射为区间interval之内的数值
        :param self:
        :param interval:
        :param chromosome:
        :return:
        '''
        d = interval[1] - interval[0]
        n = float(2**self.chromosome_size-1)
        return (interval[0]+chromosome*d/n)

    def fitness_func(self, interval, chrom1, chrom2):
        '''
        适应度函数，可以根据个体的两个染色体计算出该个体的适应度
        :param self:
        :param chrom1:
        :param chrom2:
        :return:
        '''
        (x, y) = (self.decode(interval, chrom1), self.decode(interval, chrom2))
        n = lambda x, y: math.sin(math.sqrt(x**2+y**2))**2 - 0.5
        d = lambda x, y: (1+0.001*(x**2+y**2)**2)
        func = lambda x, y: 0.5 - n(x, y)/d(x, y)
        return func(x, y)

    def evaluate(self):
        '''
        用于评估种群中的个体集合self.individuals 中各个个体的适应度
        :param self:
        :return:
        '''
        sp = self.selector_probability
        for i in range(self.size):
            self.fitness[i] = self.fitness_func(self.interval, self.individuals[i][0], self.individuals[i][1])  # 每个个体适应度
        ft_sum = sum(self.fitness)
        for i in range(self.size):
            sp[i] = self.fitness[i] / float(ft_sum)  # 得到各个个体的生存概率
        for i in range(1,  self.size):
            sp[i] = sp[i] + sp[i-1]  # 将个体的生存概率进行叠加，从而计算出各个个体的选择概率

    # 轮盘赌博（选择）
    def select(self):
        (t, i) = (random.random(), 0)
        for p in self.selector_probability:
            if p > t:
                break
            i += 1
        return i

    # 交叉
    def crossover(self, chrom1, chrom2):
        p = random.random()
        n = 2 ** self.chromosome_size - 1
        if chrom1 != chrom2 and p < self.crossover_probability:
            t = random.randint(1, self.chromosome_size-1)  # 随机选择一点（单点交叉）
            mask = n << t  # 左移运算符
            (r1, r2) = (chrom1&mask, chrom2&mask)  # &是按与运算符
            mask = n >> (self.chromosome_size - t)
            (l1, l2) = (chrom1&mask, chrom2&mask)
            (chrom1, chrom2) = (r1+l2, r2+l1)
        return (chrom1, chrom2)

    # 变异
    def mutate(self, chrom):
        p = random.random()
        if p < self.mutation_probability:
            t = random.randint(1, self.chromosome_size)
            mask1 = 1 << (t-1)
            mask2 = chrom & mask1
            if mask2 > 0:
                chrom = chrom&(~mask2)  # ~按位取反运算：对数据的每个二进制位取反
            else:
                chrom = chrom ^ mask1  # ^按位异或运算：当两对应的二进位相异时，结果为1
        return chrom

    # 保留最佳个体
    def reproduct_elitist(self):
        j = -1
        for i in range(self.size):
            if self.elitist['fitness'] < self.fitness[i]:
                j = i
                self.elitist['fitness'] = self.fitness[i]
        if (j >= 0):
            self.elitist['chromsome'][0] = self.individuals[j][0]
            self.elitist['chromsome'][1] = self.individuals[j][1]
            self.age += 1
            self.elitist['age'] = self.age

    # 进化
    def evolve(self):
        indvs = self.individuals
        new_indvs = self.new_individuals
        self.evaluate()  # 计算个体适应度和选择概率
        i=0
        while True:
            # 选择两个个体
            idv1 = self.select()
            idv2 = self.select()
            # 交叉
            (idv1_x, idv1_y) = (indvs[idv1][0], indvs[idv1][1])
            (idv2_x, idv2_y) = (indvs[idv2][0], indvs[idv2][1])
            (idv1_x, idv2_x) = self.crossover(idv1_x, idv2_x)  # 对个体1和个体2的x染色体进行交叉
            (idv1_y, idv2_y) = self.crossover(idv1_y, idv2_y)
            # 变异
            (idv1_x, idv1_y) = (self.mutate(idv1_x), self.mutate(idv1_y))
            (idv2_x, idv2_y) = (self.mutate(idv2_x), self.mutate(idv2_y))

            (new_indvs[i][0], new_indvs[i][1]) = (idv1_x, idv1_y)
            (new_indvs[i+1][0], new_indvs[i+1][1]) = (idv2_x, idv2_y)
            i += 2
            if i >= self.size:
                break
        self.reproduct_elitist()

        for i in range(self.size):
            self.individuals[i][0] = self.new_individuals[i][0]
            self.individuals[i][1] = self.new_individuals[i][1]

    def run(self):
        '''
        根据种群最大进化世代数设定循环，调用evolve函数进行进化计算，并输出种群的每一代个体适应度最大值
        :param self:
        :return:
        '''
        for i in range(self.generation_max):
            self.evolve()
            # print(i, max(self.fitness), sum(self.fitness)/self.size, min(self.fitness))
            print(self.elitist)

    def pplot(self):
        fig = plt.figure(figsize=(10, 6))
        ax = Axes3D(fig)
        x = np.arange(-10, 10, 0.1)
        y = np.arange(-10, 10, 0.1)
        X, Y = np.meshgrid(x, y)  # 生成网格点坐标矩阵
        Z = 0.5 - (np.sin(np.sqrt(X ** 2 + Y ** 2)) ** 2 - 0.5) / (1 + 0.001 * (x ** 2 + y ** 2) ** 2)
        plt.xlabel('x')
        plt.ylabel('y')
        ax.set_zlim([-1, 5])
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
        plt.show()


if __name__ == '__main__':
    # 个体数量400，染色体长度25， 交叉概率0.8， 变异概率0.1， 最大世代数100
    pop = Population([-10, 10], 200, 24, 0.8, 0.1, 100)
    pop.run()
    # pop.pplot()
