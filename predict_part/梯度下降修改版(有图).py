import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import xlwt
#通过分数和投档线，建立二元回归模型，数据从2005到2020



file = "recList.csv"
df = pd.read_csv(file)
height, weight = df.shape

for i in range(0, 1):
    top = 0  # 这个用来记录是否异常
    sum1 = (df.values[i][9] + df.values[i][10] + df.values[i][11] + df.values[i][12] + df.values[i][13] + df.values[i][14]) / 6
    if (df.values[i][8] >= (sum1 * 1.15) or df.values[i][8] <= (sum1 * 0.85)):
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


    X1, X2 = np.meshgrid(year, toudang)
    X3=np.array(pred)
    # print(X3)
    # print(X1)
    # print(X2)


    #进行预测
    # futureyear=np.array([df.values[]])
    # futuretoudang=np.array([518])
    # futurex0=np.ones(1)
    # futurex1=(futureyear-year.min())/(year.max()-year.min())
    # futurex2=(futuretoudang-toudang.min())/(toudang.max()-toudang.min())
    #
    # futurex=np.stack((futurex0,futurex1,futurex2),axis=1)
    # futurepre=np.matmul(futurex,w)
    # # sum=(futurepre[0][0]+futurepre[1][0]+futurepre[2][0]+futurepre[3][0]+futurepre[4][0]+futurepre[5][0])/6
    # futurepre[0][0]=int(futurepre[0][0])
    # distance=abs(futurepre[0][0]-df.values[i][3])
    # print(futurepre)


    # #可视化
    # plt.rcParams['font.sans-serif']=['SimHei']
    # # ##损失函数
    # # ##plt.figure(figsize=(12,4))
    # # ##plt.subplot(1,2,1)
    # # #plt.figure()
    # # #plt.plot(range(0,5000),mse[0:5000])
    # # #plt.xlabel("迭代次数",fontsize=15)
    # # #plt.ylabel("损失值",fontsize=15)
    # # #plt.show()
    # #
    # #
    # # #plt.subplot(1,2,2)
    # # #plt.plot(range(5000,200000),mse[5000:200000])
    # # #plt.xlabel("迭代次数",fontsize=15)
    # # #plt.ylabel("损失值",fontsize=15)
    # # #plt.show()
    # #
    # #
    # #
    # # # #plt.subplot(1,2,2)
    # plt.figure()
    # pred=pred.reshape(-1)
    # plt.plot(year,y,color="red",marker="o",label="实际分数")
    # plt.plot(year,pred,color="blue",marker="o",label="预测分数")
    # plt.xlabel("年份",fontsize=15)
    # plt.ylabel("分数",fontsize=15)
    #
    # plt.legend()
    # plt.show()

    plt.rcParams['font.sans-serif'] = ['SimHei']

    fig = plt.figure(figsize=(8, 8))
    ax3d = Axes3D(fig)
    # ax3d.view_init(elev=0,azim=-90)

    ax3d.scatter(rank, toudang, y, color="b", marker="o", label="已知排名")
    ax3d.scatter(rank, toudang, pred, color="r", marker="*", label="预测排名")
    ax3d.plot_wireframe(X1, X2, X3, color="c", linewidth=0.5, label="拟合平面")

    ax3d.set_xlabel('score1', color='r', fontsize=16)
    ax3d.set_ylabel('score2', color='r', fontsize=16)
    ax3d.set_zlabel('score3', color='r', fontsize=16)

    # ax3d.set_yticks([1,2,3])
    # ax3d.set_zlim3d(60,100)

    plt.suptitle("梯度下降预测模型", fontsize=20)
    plt.legend(loc="upper left")

    plt.show()
