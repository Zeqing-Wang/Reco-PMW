# -*- coding: utf-8 -*-
# @Time    : 2021/9/29 23:09
# @Author  : naptmn
# @File    : getmark.py
# @Software: PyCharm
# 用于通过排名+年份获取分数
# 请注意 对于一分一档表的数据来说   所有数据都是存在的 但是对于预测数据来说
# 并非所有数据都存在  所以需要模糊一下  这里采用向下取分
import pandas as pd
def getmark(rank,year):
    df = pd.read_csv('data/henan_yifenyiduan.csv')
    yearDict = [2015,2016,2017,2018,2019,2020,2021]
    #year = year-2015
    for i in range(len(df[str(year)])):
        if int(df[str(year)][i]) < rank and int(df[str(year)][i+1]) >= rank:
            return int(df['mark'][i+1])
if __name__ =='__main__':
    print(getmark(28,2021))