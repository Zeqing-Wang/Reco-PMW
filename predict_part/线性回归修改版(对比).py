import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import xlwt


def main():
    file="recList.csv"
    df=pd.read_csv(file,encoding="gbk")
    height,weight=df.shape
    #print(df.values[0])

    top = 0  # 这个用来记录是否异常
    sum=0
    sum2=0



    #输入
    i=0
    ji=0
    while(i<height):

        try:
            #输入
            train_x0 = np.array([df.values[i][14], df.values[i][13], df.values[i][12], df.values[i][11]])
            train_x1=np.array([df.values[i][13],df.values[i][12],df.values[i][11],df.values[i][10]])
            train_x2=np.ones(len(train_x0))  #这个地方是b
            train_x=np.stack((train_x2,train_x0,train_x1),axis=1)
            train_y=np.array([df.values[i][12],df.values[i][11],df.values[i][10],df.values[i][9]])
            train_y=np.array(train_y).reshape(-1,1)


            #训练模型
            xt=np.transpose(train_x)
            xt_1=np.linalg.inv(np.matmul(xt,train_x))
            xt_1_xt=np.matmul(xt_1,xt)
            w=np.matmul(xt_1_xt,train_y)
            w=w.reshape(-1)
            # print(w)
            #
            #进行预测
            y_pred=w[1]*df.values[i][10]+w[2]*df.values[i][9]+w[0]
            y_pred=int(y_pred)
            distance = abs(y_pred - df.values[i][8])
            # print(y_pred)

            #统计数据有无异常

            sum1 = (df.values[i][9] + df.values[i][10] + df.values[i][11] + df.values[i][12] + df.values[i][13] + df.values[i][14]) / 6
            top = 0  # 这个用来记录是否异常
            if (df.values[i][8] >= (sum1 * 1.20) or df.values[i][8] <= (sum1 * 0.80)):
                # print("第%d次数据不符合要求"%i)
                top = 1

            if(y_pred>0 and distance<=df.values[i][8]*0.2 and top==0):
                sum=sum+1

            if(top==0):
                sum2=sum2+1

            print("这是第%d次"%(i+1))
            ji=ji+1
            i=i+1
            # print(i,w)

        except:
            ji=ji-1
            i=i+1
            print("这是第%d次" % (i + 1))
            # print("矩阵不可逆")

    print(sum)
    print(sum2)







if __name__ == '__main__':
    main()