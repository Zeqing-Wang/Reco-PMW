# -*- coding: utf-8 -*-
# @Time    : 2021/9/14 22:36
# @Author  : naptmn
# @File    : getrank.py
# @Software: PyCharm
from time import sleep

import pandas as pd
import numpy as np
def getrank(mark,year):
    df = pd.read_csv('data/henan_yifenyiduan.csv')
    yearDict = [2015,2016,2017,2018,2019,2020,2021]
    #year = year-2015
    for i in range(len(df[str(year)])):
        if int(df['mark'][i]) == mark:
            #print(int(df[str(year)][i]))
            #sleep(10)
            return int(df[str(year)][i])
if __name__ =='__main__':
    print(getrank(600,2021))