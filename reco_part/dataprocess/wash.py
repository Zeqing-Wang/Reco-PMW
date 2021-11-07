# -*- coding: UTF-8 -*-
from time import sleep

import pandas as pd
import numpy as np
if __name__=='__main__':
    df = pd.read_csv('./data/henanMarkFinal.csv', encoding='ansi')
    #print(df)
    df = np.array(df)
    output = open('data/henanFinalMARK.csv', 'w')
    # 分析数据特点--以学校为
    tempName = df[0][0]
    tempBatch = df[0][4]
    output.write(tempName)
    output.write(',')
    output.write(tempBatch)
    output.write(',')
    flag = True # 判断是否是第一次  因为 每一行需要有个学校名
    for i in range(len(df)):
        if tempName == df[i][0] and tempBatch == df[i][4] : #仍然是一所学校  并且判断2021是否招生
            output.write(str(df[i][7]))
            output.write(',')
        if tempName!=df[i][0] or tempBatch !=df[i][4]:
            output.write('\n')
            tempName = df[i][0]
            tempBatch = df[i][4]
            # 第一次写 则写上学校名 以及
            output.write(tempName)
            output.write(',')
            output.write(tempBatch)
            output.write(',')
            if df[i][2]!=2021:
                output.write('none2021')
                output.write(',')
            output.write(str(df[i][7]))
            output.write(',')




