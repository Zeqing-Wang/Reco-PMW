# -*- coding: utf-8 -*-
# @Time    : 2021/9/14 22:00
# @Author  : naptmn
# @File    : washHenan.py
# @Software: PyCharm
import numpy as np
import pandas as pd
if __name__ == '__main__':
    df = pd.read_csv('./data/henan15_21.csv',encoding='ansi')
    for i in range(len(df['2015年'])):
        length = len(str(df['2015年'][i]))
        temp = str(df['2015年'][i])
        if temp[length-1] == '?':
            temp = temp.strip( '?' )
            df['2015年'][i] = temp
    print(df)
    df.to_csv('./data/washHenan.csv')