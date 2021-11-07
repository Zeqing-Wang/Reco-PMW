# -*- coding: utf-8 -*-
# @Time    : 2021/10/5 20:24
# @Author  : naptmn
# @File    : wash2021Only.py
# @Software: PyCharm
import pandas as pd
import numpy as np
if __name__=='__main__':
    df = pd.read_csv('./data/recList.csv',encoding='ansi')
    count = 0
    index = []
    # 判断21年与20年是否相同就行  挑出只有21年数据的
    for i in range(len(df)):
        # 不可能连续三年相同
        if df['2021'][i] == df['2020'][i] and df['2020'][i]==df['2019'][i]:
            print(df['name'][i])
            index.append(i)
            #count+=1
    df = df.drop(index)
    df.to_csv('./recList.csv')