# -*- coding: utf-8 -*-
# @Time    : 2021/9/3 20:39
# @Author  : naptmn
# @File    : GAoptim.py
# @Software: PyCharm
## 2. GA优化算法
from rec import estimate
import tqdm
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import random
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
class GA(object):
    ###2.1 初始化
    def __init__(self, population_size, chromosome_num, chromosome_length, max_value, iter_num, pc, pm):
        '''初始化参数
        input:population_size(int):种群数
              chromosome_num(int):染色体数，对应需要寻优的参数个数
              chromosome_length(int):染色体的基因长度
              max_value(float):作用于二进制基因转化为染色体十进制数值
              iter_num(int):迭代次数
              pc(float):交叉概率阈值(0<pc<1)
              pm(float):变异概率阈值(0<pm<1)
        '''
        self.population_size = population_size
        self.choromosome_length = chromosome_length
        self.chromosome_num = chromosome_num
        self.iter_num = iter_num
        self.max_value = max_value
        self.pc = pc  ##一般取值0.4~0.99
        self.pm = pm  ##一般取值0.0001~0.1

    def species_origin(self):
        '''初始化种群、染色体、基因
        input:self(object):定义的类参数
        output:population(list):种群
        '''
        population = []
        ## 分别初始化两个染色体
        for i in range(self.chromosome_num):
            tmp1 = []  ##暂存器1，用于暂存一个染色体的全部可能二进制基因取值
            for j in range(self.population_size):
                tmp2 = []  ##暂存器2，用于暂存一个染色体的基因的每一位二进制取值
                for l in range(self.choromosome_length):
                    tmp2.append(random.randint(0, 1))
                tmp1.append(tmp2)
            population.append(tmp1)
        return population

    ###2.2 计算适应度函数值
    def translation(self, population):
        '''将染色体的二进制基因转换为十进制取值
        input:self(object):定义的类参数
              population(list):种群
        output:population_decimalism(list):种群每个染色体取值的十进制数
        '''
        # 2 ^ 4 = 16
        population_decimalism = []
        for i in range(len(population)):
            tmp = []  ##暂存器，用于暂存一个染色体的全部可能十进制取值
            for j in range(len(population[0])):
                total = 0.0
                for l in range(len(population[0][0])):
                    total += population[i][j][l] * (math.pow(2, l))
                # 将其映射到【1-10】
                total = total+1 #+1 防止其为0
                # print('before',total)
                # total = (total/15)*9+1
                # print('after',total)
                #print(total)
                tmp.append(total)
            population_decimalism.append(tmp)
        return population_decimalism

    def fitness(self, population):
        '''计算每一组染色体对应的适应度函数值
        input:self(object):定义的类参数
              population(list):种群
        output:fitness_value(list):每一组染色体对应的适应度函数值
        '''
        fitness = []
        population_decimalism = self.translation(population)
        # 这里解码后为（4,100）的列表
        for i in range(len(population_decimalism[0])):
            # print('第', i, '个：')
            sum = population_decimalism[0][i] + population_decimalism[1][i] + population_decimalism[2][i] + population_decimalism[3][i]
            population_decimalism[0][i] = population_decimalism[0][i] / sum
            population_decimalism[1][i] = population_decimalism[1][i] / sum
            population_decimalism[2][i] = population_decimalism[2][i] / sum
            population_decimalism[3][i] = population_decimalism[3][i] / sum
            a = population_decimalism[0][i]
            b = population_decimalism[1][i]
            c = population_decimalism[2][i]
            d = population_decimalism[3][i]
            fitness.append(estimate(a,b,c,d))
        return fitness

    ###2.3 选择操作
    def sum_value(self, fitness_value):
        '''适应度求和
        input:self(object):定义的类参数
              fitness_value(list):每组染色体对应的适应度函数值
        output:total(float):适应度函数值之和
        '''
        total = 0.0
        for i in range(len(fitness_value)):
            total += fitness_value[i]
        return total

    def cumsum(self, fitness1):
        '''计算适应度函数值累加列表
        input:self(object):定义的类参数
              fitness1(list):适应度函数值列表
        output:适应度函数值累加列表
        '''
        ##计算适应度函数值累加列表
        for i in range(len(fitness1) - 1, -1, -1):  # range(start,stop,[step]) # 倒计数
            total = 0.0
            j = 0
            while (j <= i):
                total += fitness1[j]
                j += 1
            fitness1[i] = total

    def selection(self, population, fitness_value):
        '''选择操作
        input:self(object):定义的类参数
              population(list):当前种群
              fitness_value(list):每一组染色体对应的适应度函数值
        '''
        new_fitness = []  ## 用于存储适应度函归一化数值
        total_fitness = self.sum_value(fitness_value)  ## 适应度函数值之和
        for i in range(len(fitness_value)):
            new_fitness.append(fitness_value[i] / total_fitness)

        self.cumsum(new_fitness)

        ms = []  ##用于存档随机数
        pop_len = len(population[0])  ##种群数

        for i in range(pop_len):
            ms.append(random.randint(0, 1))
        ms.sort()  ## 随机数从小到大排列

        ##存储每个染色体的取值指针
        fitin = 0
        newin = 0

        new_population = population

        ## 轮盘赌方式选择染色体
        while newin < pop_len & fitin < pop_len:
            if (ms[newin] < new_fitness[fitin]):
                for j in range(len(population)):
                    new_population[j][newin] = population[j][fitin]
                newin += 1
            else:
                fitin += 1

        population = new_population

    ### 2.4 交叉操作
    def crossover(self, population):
        '''交叉操作
        input:self(object):定义的类参数
              population(list):当前种群
        '''
        pop_len = len(population[0])

        for i in range(len(population)):
            for j in range(pop_len - 1):
                if (random.random() < self.pc):
                    cpoint = random.randint(0, len(population[i][j]))  ## 随机选择基因中的交叉点
                    ###实现相邻的染色体基因取值的交叉
                    tmp1 = []
                    tmp2 = []
                    # 将tmp1作为暂存器，暂时存放第i个染色体第j个取值中的前0到cpoint个基因，
                    # 然后再把第i个染色体第j+1个取值中的后面的基因，补充到tem1后面
                    tmp1.extend(population[i][j][0:cpoint])
                    tmp1.extend(population[i][j + 1][cpoint:len(population[i][j])])
                    # 将tmp2作为暂存器，暂时存放第i个染色体第j+1个取值中的前0到cpoint个基因，
                    # 然后再把第i个染色体第j个取值中的后面的基因，补充到tem2后面
                    tmp2.extend(population[i][j + 1][0:cpoint])
                    tmp2.extend(population[i][j][cpoint:len(population[i][j])])
                    # 将交叉后的染色体取值放入新的种群中
                    population[i][j] = tmp1
                    population[i][j + 1] = tmp2

    ### 2.5 变异操作
    def mutation(self, population):
        '''变异操作
        input:self(object):定义的类参数
              population(list):当前种群
        '''
        pop_len = len(population[0])  # 种群数
        Gene_len = len(population[0][0])  # 基因长度
        for i in range(len(population)):
            for j in range(pop_len):
                if (random.random() < self.pm):
                    mpoint = random.randint(0, Gene_len - 1)  ##基因变异位点
                    ##将第mpoint个基因点随机变异，变为0或者1
                    if (population[i][j][mpoint] == 1):
                        population[i][j][mpoint] = 0
                    else:
                        population[i][j][mpoint] = 1

    ### 2.6 找出当前种群中最好的适应度和对应的参数值
    def best(self, population_decimalism, fitness_value):
        '''找出最好的适应度和对应的参数值
        input:self(object):定义的类参数
              population(list):当前种群
              fitness_value:当前适应度函数值列表
        output:[bestparameters,bestfitness]:最优参数和最优适应度函数值
        '''
        #print('here',population_decimalism)
        #sleep(1000)
        pop_len = len(population_decimalism[0])
        bestparameters = []  ##用于存储当前种群最优适应度函数值对应的参数
        bestfitness = 0.0  ##用于存储当前种群最优适应度函数值
        #print('fitness::',fitness_value)
        for i in range(0, pop_len):
            tmp = []
            if (fitness_value[i] > bestfitness):
                bestfitness = fitness_value[i]
                for j in range(len(population_decimalism)):
                    # tmp.append(
                    #     abs(population_decimalism[j][i] * self.max_value / (math.pow(2, self.choromosome_length) - 10)))
                    tmp.append(population_decimalism[j][i])
                    bestparameters = tmp

        return bestparameters, bestfitness

    ### 2.7 画出适应度函数值变化图
    def plot(self, results):
        '''画图
        '''
        X = []
        Y = []
        for i in range(self.iter_num):
            X.append(i + 1)
            Y.append(results[i])
        plt.plot(X, Y)
        plt.xlabel('迭代轮数', size=15)
        plt.ylabel('适应度', size=15)
        #plt.title('遗传算法参数优化')
        plt.show()

    ### 2.8 主函数
    def main(self):
        results = []
        parameters = []
        best_fitness = 0.0
        best_parameters = []
        ## 初始化种群
        population = self.species_origin()
        ## 迭代参数寻优
        for i in tqdm.tqdm(range(self.iter_num)):
            ##计算适应函数数值列表
            fitness_value = self.fitness(population)
            ## 计算当前种群每个染色体的10进制取值
            population_decimalism = self.translation(population)
            ## 寻找当前种群最好的参数值和最优适应度函数值
            current_parameters, current_fitness = self.best(population_decimalism, fitness_value)
            ## 与之前的最优适应度函数值比较，如果更优秀则替换最优适应度函数值和对应的参数
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_parameters = current_parameters
            print('iteration is :', i, ';Best parameters:', best_parameters, ';Best fitness', best_fitness)
            results.append(best_fitness)
            parameters.append(best_parameters)

            ## 种群更新
            ## 选择
            self.selection(population, fitness_value)
            ## 交叉
            self.crossover(population)
            ## 变异
            self.mutation(population)
        results.sort()
        self.plot(results)
        print('Final parameters are :', parameters[-1])
        a = parameters[-1][0] / sum(parameters[-1])
        b = parameters[-1][1] / sum(parameters[-1])
        c = parameters[-1][2] / sum(parameters[-1])
        d = parameters[-1][3] / sum(parameters[-1])
        print('a=',a,'b=',b,'c=',c,'d=',d)
if __name__ == '__main__':
    seed_torch(620664)
    ga = GA(population_size=30, chromosome_num=4, chromosome_length=4, max_value=10, iter_num=40, pc=0.5, pm=0.001)
    ga.main()