import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlwt

file = "河南排名信息.csv"
df = pd.read_csv(file)
height, weight = df.shape
top=0

for i in range(0, height):# 这个用来记录是否异常
    print("到%d次"%(i+1))
    sum1 = (df.values[i][4] + df.values[i][5] + df.values[i][6] + df.values[i][7] + df.values[i][8] + df.values[i][9]) / 6
    if (df.values[i][3] >= (sum1 * 1.15) or df.values[i][3] <= (sum1 * 0.85)):
        # print("第%d次数据不符合要求"%i)
        top=top+1
print("异常的数据有%d所"%top)
print("不异常的数据有%d所"%(height-top))


