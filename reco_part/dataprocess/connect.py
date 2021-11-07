# -*- coding: utf-8 -*-
# @Time    : 2021/10/19 0:30
# @Author  : naptmn
# @File    : connect.py
# @Software: PyCharm
import pandas as pd
import numpy as np
if __name__ =='__main__':
    df = pd.read_csv('./data/recList.csv',encoding='ansi')
    dfnew = pd.read_csv('./data/BPnew.csv',encoding='ansi')
    for i in range(len(df)):
        for k in range(len(dfnew)):
            if df['name'][i]==dfnew['name'][k]:
                df['bp'][i] = dfnew['rank'][k]
    df.to_csv('newbp.csv')