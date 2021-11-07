# -*- coding: UTF-8 -*-
from time import sleep

import pandas as pd
import numpy as np
if __name__ =='__main__':
    df = pd.read_csv('data/henanFinal.csv', encoding='ansi')
    print(df)
    #sleep(1000)
    year = ['2021','2020', '2019', '2018', '2017', '2016', '2015']
    #print(df['2020'])
    for i in year:
        temp = df[i][0]
        for j in range(len(df)):
            # 先判断是否为空 这个条件是建立在第一行全齐的条件下
            if pd.isnull(df[i][j]):
                df[i][j] = temp
            else:
                temp = df[i][j]
        #print(df[i])
    df[year] = df[year].astype('int')
    df.to_csv('fill_final.csv',index = True)