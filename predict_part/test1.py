import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlwt

def logsig(x):

    return 1/(1+np.exp(-x))


def main():

    # 读入数据
    file="河南排名信息.csv"
    df=pd.read_csv(file)
    height,weight=df.shape
    #print(height,weight)
    # print(df.values[0])



    #这里是将一些排名波动过大的学校删去
    #top=0
    year=np.array([2020,2019,2018,2017,2016,2015])
    x1 = (year - year.min()) / (year.max() - year.min())
    #ability=np.array([1,1,1,1,1,1])
    height=height-1
    for i in range(0,1):
        top=0 #这个用来记录是否异常
        sum1=(df.values[i][4]+df.values[i][5]+df.values[i][6]+df.values[i][7]+df.values[i][8]+df.values[i][9])/6
        if(df.values[i][3]>=(sum1*1.15) or df.values[i][3]<=(sum1*0.85)):
            #print("第%d次数据不符合要求"%i)
            top=1
        rank = [] #这个用来记录排名
        for j in range(4, 10):
            rank.append(df.values[i][j])
        #print(rank)

        sampleinnorm = np.array([x1])
        sampleoutnorm = np.array([rank])
        print(sampleinnorm)







if __name__ == '__main__':
    main()
