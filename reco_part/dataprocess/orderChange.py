# -*- coding: utf-8 -*-
# @Time    : 2021/10/15 9:51
# @Author  : naptmn
# @File    : orderChange.py
# @Software: PyCharm
import pandas as pd
import numpy as np
if __name__ =='__main__':
    df = pd.read_csv('./data/recList.csv',encoding='ansi')
    for i in range(len(df)):
        if '（' in df['2021rank'][i]:
            print(df['2021rank'][i])
            string = df['2021rank'][i].split('（')
            string = string[1]
            string = string.split('）')[0]
            print(string)
            df['2021rank'][i] = string
        if '+' in df['2021rank'][i]:
            print(df['2021rank'][i])
            string = df['2021rank'][i].split('+')
            string = string[0]
            print(string)
            df['2021rank'][i] = string
    df.to_csv('recList.csv')