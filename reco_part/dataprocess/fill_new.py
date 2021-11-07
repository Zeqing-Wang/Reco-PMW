# -*- coding: utf-8 -*-
# @Time    : 2021/9/20 21:47
# @Author  : naptmn
# @File    : fill_new.py
# @Software: PyCharm
from time import sleep
import pandas as pd
import numpy as np
if __name__ =='__main__':
    df = pd.read_csv('data/henanFinal.csv', encoding='ansi')
    year = ['name','2021','2020', '2019', '2018', '2017', '2016', '2015']
    yearOnly = ['2021','2020', '2019', '2018', '2017', '2016', '2015']
    print(df)
    #sleep(1000)
    for i in range(len(df)):
        for j in range(7):
            if pd.isnull(df[yearOnly[j]][i]) or df[yearOnly[j]][i]==0:
                df.iloc[[i],[j+2]] = df.iloc[[i],[j+1]]
    print(df[yearOnly])
    df[yearOnly] = df[yearOnly].astype('int')
    df.to_csv('rankMarkHenan21-15.csv', index=True)