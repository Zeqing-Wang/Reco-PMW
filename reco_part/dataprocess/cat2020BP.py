# -*- coding: utf-8 -*-
# @Time    : 2021/11/1 23:49
# @Author  : naptmn
# @File    : cat2020BP.py
# @Software: PyCharm
#将2020年BP结果与reclist进行拼接
import numpy as np
import pandas as pd
if __name__ =='__main__':
    df20 = pd.read_csv('./data/BP2020.csv',encoding='gbk')
    dfrec = pd.read_csv('./data/recList.csv',encoding='ansi')
    for i in range(len(dfrec)):
        for j in range(len(df20)):
            print(dfrec['name'][i],df20['name'][j])
            if dfrec['name'][i]==df20['name'][j]:
                print('11')
                dfrec['bp2020'][i]=df20['rank'][j]
                break
    dfrec['bp2020'].dtype='int32'
    print(dfrec['bp2020'])
    dfrec.to_csv('recList.csv')