import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


#读取数据
file = "recList.csv"
df = pd.read_csv(file,encoding="gbk")
height, weight = df.shape

# 输入
i = 0
ji = 0
sum11=0
sum12=0
while (i < height):

    try:
        top = 0  # 这个用来记录是否异常
        sum1 = (df.values[i][9] + df.values[i][10] + df.values[i][11] + df.values[i][12] + df.values[i][13] + df.values[i][14]) / 6
        if (df.values[i][8] >= (sum1 * 1.2) or df.values[i][8] <= (sum1 * 0.8)):
            # print("第%d次数据不符合要求"%i)
            top = 1

        #数据预处理
        toudang = np.array([df.values[i][13],df.values[i][12], df.values[i][11], df.values[i][10]])
        year = np.array([df.values[i][14],df.values[i][13], df.values[i][11], df.values[i][11]])
        rank=np.array([df.values[i][12],df.values[i][11], df.values[i][10], df.values[i][9]])
        num = len(year)





        #数据处理
        x0 = np.ones(num)
        x1=(year-year.min())/(year.max()-year.min())
        x2=(toudang-toudang.min())/(toudang.max()-toudang.min())

        x=np.stack((x0,x1,x2),axis=1)
        rank=np.array(rank)
        y=rank.reshape(-1,1)
        #设置超参数，学习率和迭代次数
        learn_rate=0.001
        iter=20000
        display_step=500

        #设置模型参数初值w0和b0
        np.random.seed(612)
        w=np.random.randn(3,1)
        #b=np.random.randn()

        #训练模型
        mse=[]

        for j in range(0,iter+1):
            dl_dw=np.matmul(np.transpose(x),np.matmul(x,w)-y)  #计算偏导数
            w=w-learn_rate*dl_dw #迭代

            pred=np.matmul(x,w) #计算分数的估计值
            loss=np.mean(np.square(y-pred))/2   #计算损失函数
            mse.append(loss)


            # if j % display_step == 0:
            #     print("j: %d,loss: %f"%(j,mse[j]))




        # 进行预测
        futureyear=np.array([df.values[i][10]])
        futuretoudang=np.array([df.values[i][9]])
        futurex0=np.ones(1)
        futurex1=(futureyear-year.min())/(year.max()-year.min())
        futurex2=(futuretoudang-toudang.min())/(toudang.max()-toudang.min())

        futurex=np.stack((futurex0,futurex1,futurex2),axis=1)
        futurepre=np.matmul(futurex,w)
        # sum=(futurepre[0][0]+futurepre[1][0]+futurepre[2][0]+futurepre[3][0]+futurepre[4][0]+futurepre[5][0])/6
        futurepre[0][0]=int(futurepre[0][0])
        distance=abs(futurepre[0][0]-df.values[i][8])
        # print(futurepre)

        if (futurepre[0][0] > 0 and distance <= df.values[i][8] * 0.2 and top == 0):
            sum11 = sum11 + 1

        if (top == 0):
            sum12 = sum12 + 1


        print("这是进行到了第%d次" % (i + 1))

        ji = ji + 1
        i = i + 1
    except:
        print("这是进行到了第%d次" % (i + 1))
        ji = ji - 1
        i = i + 1
        # print("矩阵不可逆")


print(sum11)
print(sum12)





