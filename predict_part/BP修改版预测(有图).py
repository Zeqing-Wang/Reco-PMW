import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xlwt

def logsig(x):

    return 1/(1+np.exp(-x))


def main():

    # 读入数据
    file="recList.csv"
    df=pd.read_csv(file,encoding="gbk")
    height,weight=df.shape
    #print(height,weight)
    # print(df.values[0])



    #开始处理数据

    #这里是将一些排名波动过大的学校删去
    #top=0


    for i in range(0,1):

        sum1 = (df.values[i][9] + df.values[i][10] + df.values[i][11] + df.values[i][12] + df.values[i][13] +
                df.values[i][14]) / 6
        top = 0  # 这个用来记录是否异常
        if (df.values[i][8] >= (sum1 * 1.15) or df.values[i][8] <= (sum1 * 0.85)):
            # print("第%d次数据不符合要求"%i)
            top = 1

        train_x0 = np.array([df.values[i][14], df.values[i][13], df.values[i][12], df.values[i][11]])
        train_x1 = np.array([df.values[i][13], df.values[i][12], df.values[i][11], df.values[i][10]])
        train_y = np.array([df.values[i][12], df.values[i][11], df.values[i][10], df.values[i][9]])

        sampleinnorm = np.array([train_x0,train_x1])
        sampleoutnorm = np.array([train_y])

        # 训练模型

        maxepochs = 1000  # 迭代次数
        learnrate = 0.001
        errorfinal = 0.65 * 10 ** (-3)
        samnum = 4
        indim = 2
        outdim = 1
        hiddenunitnum = 8
        w1 = 0.5 * np.random.rand(hiddenunitnum, indim) - 0.1
        b1 = 0.5 * np.random.rand(hiddenunitnum, 1) - 0.1
        w2 = 0.5 * np.random.rand(outdim, hiddenunitnum) - 0.1
        b2 = 0.5 * np.random.rand(outdim, 1) - 0.1

        errhistory = []

        for j in range(maxepochs):
            hiddenout = logsig((np.dot(w1, sampleinnorm).transpose() + b1.transpose())).transpose()
            networkout = (np.dot(w2, hiddenout).transpose() + b2.transpose()).transpose()
            err = sampleoutnorm - networkout
            sse = sum(sum(err ** 2))  # 这个地方是求矩阵所有元素的和
            errhistory.append(sse)
            if sse < errorfinal:
                break
            delta2 = err
            delta1 = np.dot(w2.transpose(), delta2) * hiddenout * (1 - hiddenout)
            dw2 = np.dot(delta2, hiddenout.transpose())
            db2 = np.dot(delta2, np.ones((samnum, 1)))
            dw1 = np.dot(delta1, sampleinnorm.transpose())
            db1 = np.dot(delta1, np.ones((samnum, 1)))
            w2 += learnrate * dw2
            b2 += learnrate * db2
            w1 += learnrate * dw1
            b1 += learnrate * db1

        # print(errhistory)

        # # 进行结果预测
        # futureyear = np.array([2021])
        # futureyear=(futureyear-year.min())/(year.max()-year.min())
        # # futurerank=np.array([round(i+1/300,4)])
        # sampleinnorm1=np.array([futureyear])
        # hiddenout1 = logsig((np.dot(w1, sampleinnorm1).transpose() + b1.transpose())).transpose()
        # networkout1 = (np.dot(w2, hiddenout1).transpose() + b2.transpose()).transpose()
        # networkout1[0][0]=int(networkout1[0][0])
        # # futurerank = getrank(2019,df.values[i][7])
        # loss = abs(networkout1 -df.values[i][3])
        # # print(loss)

        #绘图

        # X1, X2 = np.meshgrid(train_x0, train_x1)
        # X3 = np.array(networkout)
        #
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        #
        # fig = plt.figure(figsize=(8, 8))
        # ax3d = Axes3D(fig)
        # # ax3d.view_init(elev=0,azim=-90)
        #
        # ax3d.scatter(train_x0, train_x1, train_y, color="b", marker="o", label="已知排名")
        # ax3d.scatter(train_x0, train_x1, networkout, color="r", marker="*", label="预测排名")
        # ax3d.plot_wireframe(X1, X2, X3, color="c", linewidth=0.5, label="拟合平面")
        #
        # ax3d.set_xlabel('score1', color='r', fontsize=16)
        # ax3d.set_ylabel('score2', color='r', fontsize=16)
        # ax3d.set_zlabel('score3', color='r', fontsize=16)
        #
        # # ax3d.set_yticks([1,2,3])
        # # ax3d.set_zlim3d(60,100)
        #
        # plt.suptitle("BP神经网络预测模型", fontsize=20)
        # plt.legend(loc="upper left")
        #
        # plt.show()


        #绘图,可视化
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.figure()
        plt.plot(errhistory[0:500])
        plt.xlabel("迭代次数", fontsize=15)
        plt.ylabel("损失值", fontsize=15)
        plt.show()






if __name__ == '__main__':
    main()
