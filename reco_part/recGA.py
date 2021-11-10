# -*- coding: UTF-8 -*-
from time import sleep

from getmark import getmark
from getrank import getrank
import pandas as pd
import numpy as np
import random
import tqdm
import os
import torch
import math
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
def nortest(x):
    # 这里为测试部分（GA调参）
    u = 517  # 均值μ
    sig = 56  # 标准差δ
    y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
    return y_sig
def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
def classify(year,mark):
    # 初步筛选学校的函数 输入为学生高考年份与分数 返回为学校列表
    # 这里df中的数据均为15-20年真实位次数据  没有21年的！
    df = pd.read_csv('./data/recList.csv', encoding='ansi')
    rank_s = getrank(mark,year)
    school = [] # 筛选的学校
    school_rank = [] # 筛选学校的排名
    school_batch = [] # 记住学校的批次  有些学校存在多批次的情况
    count = 0 #筛选学校的数目
    # 这里先拿 15-19的做为已知 20的作为验证年份 之后需要改为 15-20作为已知 21作为验证
    N = int(25*100*nortest(mark)) + 25 #调优 21为未知 不使用
    #print('入选高校数量为', N)
    for i in range(len(df)):
        year = 2015
        for j in range(5):#GA时改为5  使用时改为6
            #print(year)
            if df[str(year)][i]>rank_s:
                count = count+1
                school.append(df['name'][i])
                school_rank.append(int(df['2021rank'][i]))  #这里直接添加行号即可  因为他是按照排名顺序来的 不是的！！
                school_batch.append(df['batch'][i])
                break
            year = year+1
        if count == N: #同时要找到高校才能停止
            break
    return school,school_rank,school_batch
def claPara(school,year_s,mark,school_rank,school_batch):
    # 根据筛选出的学校 考生年份 考生分数 计算概率的相关参数
    real_mark = [] # 提供学校预测年份的实际分数线 注意 是真实分数 并非预测 ！！！！
    # 以下年份仅供测试  之后需要改为以15-20作为已知 21作为预测！！
    p = [] # 往年（这里是15-19年）能够考上位次所占百分比
    f = [] # 预测得到的分与考生分数的百分比
    r = [] # 软科综合实力排名
    g = [] # 高校投档位次与目标排名的距离离散度
    # 计算出着四个参数后进行归一化处理  之后进行参数优化
    year = ['2015', '2016', '2017', '2018', '2019']  # 后期需要加上2020 GA时把2020去掉  使用时把2020加上
    # 要改成预测的！！！   注意预测的是在一个单独文件里面
    yearP = '2020'  # 后期改为2021 GA时改为2020  使用时改为2021
    rank_s = getrank(mark,year_s)
    # df 中包括21-15中的
    df = pd.read_csv('./data/recList.csv', encoding='ansi')
    for i in range(len(school)):
        for j in range(len(df)):
            if df['name'][j] == school[i] and df['batch'][j]==school_batch[i]:
                # 放入真实分数线
                real_mark.append(getmark(df[yearP][j],2020))# 后期改为2021 GA时改为2020  使用时改为2021
                # 计算p
                count = 0
                for k in year:
                    if df[k][j]>rank_s:
                        count = count+1
                count = count/len(year)
                p.append(count)
                # 计算f
                f_c = (mark-getmark(df['bp2020'][j],2020))/(750-mark) #GA时改为 bp2020 与 2020 使用时改成bp 2021
                f.append(f_c)
                # 计算g
                g_c = 0
                for k in year:
                    g_c = g_c + ((rank_s-df[k][j])/rank_s)**2
                g_c = g_c ** 0.5
                g_c = g_c / len(year)
                g.append(g_c)
    # 计算r 按照之前的排名即可
    r = school_rank
    return p, f, r, g, real_mark

def normalization(p,f,r,g):
    maxP = max(p)
    maxF = max(f)
    maxR = max(r)
    maxG = max(g)
    minP = min(p)
    minF = min(f)
    minR = min(r)
    minG = min(g)
    for i in range(len(p)):
        if maxP == minP: # 有一种极端情况  即 所有院校都相同
            p[i] = p[i]
        f[i] = (f[i] - minF) / (maxF - minF)
        r[i] = (r[i] - minR) / (maxR - minR)
        g[i] = (g[i] - minG) / (maxG - minG)
    return p,f,r,g
def calResult(mark,real_mark,result):
    chong = 0
    wen = 0
    bao = 0
    chong_sum = 0
    wen_sum = 0
    bao_sum = 0
    for i in range(len(result)):
        if result[i]>=0.8: #保
            if real_mark[i]<=mark:
                bao = bao + 1
            bao_sum = bao_sum + 1
        elif result[i]<=0.6: #冲
            if real_mark[i]<=mark:
                chong = chong + 1
            chong_sum = chong_sum + 1
        else:
            if real_mark[i]<=mark:
                wen = wen+1
            wen_sum = wen_sum+1
    # 防止极端情况下出现除零错误
    if chong_sum!=0:
        chongP = chong/chong_sum
    else:
        chongP = 0
    if wen_sum!=0:
        wenP = wen/wen_sum
    else:
        wenP = 0
    if bao_sum!=0:
        baoP = bao/bao_sum
    else:
        baoP = 0
    return chongP,wenP,baoP
def estimate(a,b,c,d):
    mark = [475,500,525,550,575,600,625,650]
    fit = 0
    for i in mark:
        bao,var = recGA(i,a,b,c,d)
        fit_temp = 0.8*bao+15*var
        fit = fit + fit_temp
    return fit/8
    #print(fit)

def recGA(mark, a, b, c, d):
    yearTest = 2020 #GA时为2020  使用时为2021
    markTest = mark
    school, school_rank, school_batch = classify(yearTest, markTest)
    # real_mark需要改
    p, f, r, g, real_mark = claPara(school, yearTest, markTest, school_rank, school_batch)
    # print(p,f,r,g)
    p, f, r, g = normalization(p, f, r, g)
    result = []
    for i in range(len(p)):
        temp = p[i] * a + f[i] * b + r[i] * c + g[i] * d
        result.append(temp)
    chong, wen, bao = calResult(markTest, real_mark, result)
    # 这里返回的是冲稳保三个录取率 以及命中率之间的方差 result保存了50所学校的录取概率 school中保存了50所学校  这里面
    return bao, np.var(result)
if __name__ =='__main__':
    seed_torch(620664) # 固定随机数种子
