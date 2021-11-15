
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
def nor(x):
    # 这里为使用部分 测试时请换为下面的函数nortest
    u = 540  # 均值μ
    sig = 55  # 标准差δ
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
    N = int(25*100*nor(mark)) + 25 #使用部分 实际使用请把下方函数注释掉
    #N = int(25*100*nortest(mark)) + 25 #调优 21为未知 不使用
    print('入选高校数量为', N)
    for i in range(len(df)):
        year = 2015
        for j in range(6):#GA时改为5  使用时改为6
            #print(year)
            if df[str(year)][i] >= rank_s:
                count = count+1
                school.append(df['name'][i])
                school_rank.append(int(df['2021rank'][i]))
                school_batch.append(df['batch'][i])
                break
            year = year+1
        if count == N: #同时要找到高校才能停止
            break
    return school,school_rank,school_batch
def claPara(school,year_s,mark,school_rank,school_batch):
    # 根据筛选出的学校 考生年份 考生分数 计算概率的相关参数
    real_mark = [] # 提供学校预测年份的实际分数线 注意 是真实分数 并非预测 ！！！！！
    p = [] # 往年（这里是15-20年）能够考上位次所占百分比
    f = [] # 预测得到的分与考生分数的百分比
    r = [] # 软科综合实力排名
    g = [] # 高校投档位次与目标排名的距离离散度
    year = ['2015', '2016', '2017', '2018', '2019','2020']
    # 要改成预测的！！！   注意预测的是在一个单独文件里面
    yearP = '2021'  # 后期改为2021 GA时改为2020  使用时改为2021
    rank_s = getrank(mark,year_s)
    # df 中包括21-15中的
    df = pd.read_csv('./data/recList.csv', encoding='ansi')
    for i in range(len(school)):
        for j in range(len(df)):
            if df['name'][j] == school[i] and df['batch'][j]==school_batch[i]:
                # 放入真实分数线
                real_mark.append(getmark(df[yearP][j],2021))# 后期改为2021 GA时改为2020  使用时改为2021
                # 计算p
                count = 0
                for k in year:
                    if df[k][j] >= rank_s:
                        count = count+1
                count = count/len(year)
                p.append(count)
                # 计算f
                f_c = (mark-getmark(df['bp'][j],2021))/(750-mark) #使用时改成 bp 2021
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
    maxF = max(f)
    maxR = max(r)
    maxG = max(g)
    minF = min(f)
    minR = min(r)
    minG = min(g)
    for i in range(len(p)):
        f[i] = (f[i] - minF) / (maxF - minF)
        r[i] = (r[i] - minR) / (maxR - minR)
        g[i] = (g[i] - minG) / (maxG - minG)
    return p,f,r,g
def randomTest():
    Test_num = 50
    # 每个分段模拟50位考生
    a = 0.73
    b = 0.13
    c = 0.09
    d = 0.05
    mark_650 = []
    mark_600 = []
    mark_550 = []
    mark_500 = []
    mark_450 = []
    for i in range(Test_num):
        mark_650.append(random.randint(650,700))
        mark_600.append(random.randint(600,650))
        mark_550.append(random.randint(550,600))
        mark_500.append(random.randint(500,550))
        mark_450.append(random.randint(450,500))
    sum_chong_650 = 0
    sum_wen_650 = 0
    sum_bao_650 = 0
    sum_chong_600 = 0
    sum_wen_600 = 0
    sum_bao_600 = 0
    sum_chong_550 = 0
    sum_wen_550 = 0
    sum_bao_550 = 0
    sum_chong_500 = 0
    sum_wen_500 = 0
    sum_bao_500 = 0
    sum_chong_450 = 0
    sum_wen_450 = 0
    sum_bao_450 = 0
    for i in tqdm.tqdm(range(Test_num)):
        chong,wen,bao,_,_,_,_ = recAll(mark_650[i],a,b,c,d)
        sum_chong_650 = sum_chong_650+chong
        sum_wen_650 = sum_wen_650 + wen
        sum_bao_650 = sum_bao_650 + bao
        chong, wen, bao,_,_,_,_ = recAll(mark_600[i], a, b, c, d)
        sum_chong_600 = sum_chong_600 + chong
        sum_wen_600 = sum_wen_600 + wen
        sum_bao_600 = sum_bao_600 + bao
        chong, wen, bao,_,_,_,_ = recAll(mark_550[i], a, b, c, d)
        sum_chong_550 = sum_chong_550 + chong
        sum_wen_550 = sum_wen_550 + wen
        sum_bao_550 = sum_bao_550 + bao
        chong, wen, bao,_,_,_,_ = recAll(mark_500[i], a, b, c, d)
        sum_chong_500 = sum_chong_500 + chong
        sum_wen_500 = sum_wen_500 + wen
        sum_bao_500 = sum_bao_500 + bao
        chong, wen, bao,_,_,_,_ = recAll(mark_450[i], a, b, c, d)
        sum_chong_450 = sum_chong_450 + chong
        sum_wen_450 = sum_wen_450 + wen
        sum_bao_450 = sum_bao_450 + bao
    res650 = [sum_chong_650 / Test_num, sum_wen_650 / Test_num, sum_bao_650 / Test_num]
    res600 = [sum_chong_600 / Test_num, sum_wen_600 / Test_num, sum_bao_600 / Test_num]
    res550 = [sum_chong_550 / Test_num, sum_wen_550 / Test_num, sum_bao_550 / Test_num]
    res500 = [sum_chong_500 / Test_num, sum_wen_500 / Test_num, sum_bao_500 / Test_num]
    res450 = [sum_chong_450 / Test_num, sum_wen_450 / Test_num, sum_bao_450 / Test_num]
    return res650,res600,res550,res500,res450
def plot():
    label = ["650-700", "600-650", "550-600", "500-550", "450-500"]
    res650, res600, res550, res500, res450 = randomTest()
    chong = [res650[0],res600[0],res550[0],res500[0],res450[0]]
    wen = [res650[1],res600[1],res550[1],res500[1],res450[1]]
    bao = [res650[2],res600[2],res550[2],res500[2],res450[2]]
    fig, axes = plt.subplots(1, 1, figsize=(8, 4))
    axes.plot(label, chong, linestyle='-',label="冲", color="#845EC2", marker='x', linewidth=3)
    axes.plot(label, wen, linestyle='-',label="稳", color="#D7E8F0", marker='o', linewidth=3)
    axes.plot(label, bao, linestyle='-', label="保", color="#2E839E", marker='v', linewidth=3)
    axes.set_ylabel("录取率")
    axes.set_xlabel("分数区间")
    axes.legend()
    plt.savefig(fname='marktest.svg', format='svg')
    plt.show()
def calResult(mark,real_mark,result):
    chong = 0
    wen = 0
    bao = 0
    chong_sum = 0
    wen_sum = 0
    bao_sum = 0
    for i in range(len(result)):
        if result[i]>0.8: #保
            if real_mark[i]<=mark:
                bao = bao + 1
            bao_sum = bao_sum + 1
        elif result[i]<0.6: #冲
            if real_mark[i]<=mark:
                chong = chong + 1
            chong_sum = chong_sum + 1
        elif result[i]>=0.6 and result[i]<=0.8:#稳
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
def recAll(mark,a,b,c,d):
    yearTest = 2021 #GA时改为2020 使用时为2021
    markTest = mark
    school , school_rank,school_batch= classify(yearTest, markTest)
    # real_mark需要改
    p, f, r, g, real_mark = claPara(school, yearTest, markTest,school_rank,school_batch)
    # print(p,f,r,g)
    p, f, r, g = normalization(p, f, r, g)
    result = []
    for i in range(len(p)):
        temp = p[i] * a + f[i] * b + r[i] * c + g[i] * d
        result.append(temp)
    chong, wen, bao = calResult(markTest, real_mark, result)
    # 这里返回的是冲稳保三个录取率 以及命中率之间的方差 result保存了50所学校的录取概率 school中保存了50所学校  这里面
    return chong,wen,bao,school,result,school_batch,r
def final(school,result,batch,r):
    # school为学校名字  result为录取概率 batch为所在批次 r为排名归一化后结果
    barDown = 0.6
    barUp = 0.8
    chong = []
    wen = []
    bao = []
    chong_batch = []
    wen_batch = []
    bao_batch = []
    recmark = [] # 推荐度
    # 计算推荐度
    for k in range(len(result)):
        recmark.append(result[k]*(1-r[k]))
    # 定义推荐度 p*(1-r)  每个类别中取推荐度前两位  修复部分情况下无法推荐满6所高校的情况  并且输出高校所在批次
    dict_r = list(sorted(zip(recmark,school,result,batch),reverse=True))
    #print(dict_r)
    #sleep(1000)
    all_school = [] #维护一个总推荐
    chonglen=2
    wenlen=0
    baolen=4
    # 计算一下每个类中的个数 方便调整
    chong_num = []
    wen_num = []
    bao_num = []
    for i in range(len(dict_r)):
        if dict_r[i][2]>barUp:
            bao_num.append(dict_r[i][1])
            #continue
        if dict_r[i][2]<barDown:
            chong_num.append(dict_r[i][1])
            #continue
        if dict_r[i][2]>=barDown and dict_r[i][2]<=barUp:
            wen_num.append(dict_r[i][1])
    print('冲:',len(chong_num),'稳:',len(wen_num),'保:',len(bao_num))
    for i in range(len(dict_r)):
        if dict_r[i][2]>barUp and len(bao)<baolen:
            bao.append(dict_r[i][1])
            bao_batch.append(dict_r[i][3])
            all_school.append(dict_r[i])
        if dict_r[i][2]<barDown and len(chong)<chonglen:
            chong.append(dict_r[i][1])
            chong_batch.append(dict_r[i][3])
            all_school.append(dict_r[i])
        if dict_r[i][2]>=barDown and dict_r[i][2]<= barUp and len(wen)<wenlen:
            wen.append(dict_r[i][1])
            wen_batch.append(dict_r[i][3])
            all_school.append(dict_r[i])
    chong = list(zip(chong,chong_batch))
    wen = list(zip(wen,wen_batch))
    bao = list(zip(bao,bao_batch))
    return chong,wen,bao
if __name__ =='__main__':
    seed_torch(1005) # 固定随机数种子
    # plot()
    # sleep(1000)
    a = 0.73
    b = 0.13
    c = 0.09
    d = 0.05
    _,_,_,school,result,batch,r = recAll(618,a,b,c,d)
    chong,wen,bao = final(school,result,batch,r)
    print(chong,wen,bao)