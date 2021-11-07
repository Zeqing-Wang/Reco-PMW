import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlwt

#将三个模型的结果放到一起进行横向对比
def logsig(x):

    return 1/(1+np.exp(-x))


def main():
    file = "河南排名信息.csv"
    df = pd.read_csv(file)
    height, weight = df.shape


    # 第一步加载样本数据
    year = np.array([2015, 2016, 2017, 2018, 2019, 2020])
    toudang = np.array([529, 523, 484, 499, 502, 544])
    num2 = len(year)
    total1=0
    total2=0
    total3=0

    for i in range(0, height):
        print("现在已经运行到第%d次"%(i+1))
        top = 0  # 这个用来记录是否异常
        sum1 = (df.values[i][4] + df.values[i][5] + df.values[i][6] + df.values[i][7] + df.values[i][8] + df.values[i][9]) / 6
        if (df.values[i][3] >= (sum1 * 1.15) or df.values[i][3] <= (sum1 * 0.85)):
            # print("第%d次数据不符合要求"%i)
            top = 1
        rank = []  # 这个用来记录排名
        for j in range(9, 3, -1):  # 这个地方得从后往前遍历列表
            rank.append(df.values[i][j])

        #线性回归
        x1=year
        y1 = rank
        # 求均值
        meanX = np.mean(x1)
        meanY = np.mean(y1)

        # 利用numpy广播原则
        sumXY = np.sum((x1 - meanX) * (y1 - meanY))  # 分子
        sumX = np.sum((x1 - meanX) * (x1 - meanX))  # 分母
        w1 = sumXY / sumX
        b1 = meanY - w1 * meanX

        # print("w=",w)
        # print("b=",b)
        # pred=w*x+b
        # 第三步预测
        x_test = np.array([2021])
        y_pred = w1 * x_test + b1
        y_pred = int(y_pred)
        distance1 = abs(y_pred - df.values[i][3])
        if(distance1<=df.values[i][3]*0.15 and top==0):
            total1=total1+1
        # print(sum,distance)



        #梯度下降模型
        # 数据处理
        x20 = np.ones(num2)
        x21 = (year - year.min()) / (year.max() - year.min())
        x22 = (toudang - toudang.min()) / (toudang.max() - toudang.min())

        x2 = np.stack((x20, x21, x22), axis=1)
        rank = np.array(rank)
        y2 = rank.reshape(-1, 1)
        # 设置超参数，学习率和迭代次数
        learn_rate2 = 0.001
        iter2 = 2000
        display_step2 = 500

        # 设置模型参数初值w0和b0
        np.random.seed(612)
        w2 = np.random.randn(3, 1)
        # b2=np.random.randn()

        # 训练模型
        mse2 = []

        for j in range(0, iter2 + 1):
            dl_dw2 = np.matmul(np.transpose(x2), np.matmul(x2, w2) - y2)  # 计算偏导数
            w2 = w2 - learn_rate2 * dl_dw2  # 迭代

            pred2 = np.matmul(x2, w2)  # 计算分数的估计值
            loss2 = np.mean(np.square(y2 - pred2)) / 2  # 计算损失函数
            mse2.append(loss2)

        # 进行预测
        futureyear2 = np.array([2021])
        futuretoudang2 = np.array([518])
        futurex20 = np.ones(1)
        futurex21 = (futureyear2 - year.min()) / (year.max() - year.min())
        futurex22 = (futuretoudang2 - toudang.min()) / (toudang.max() - toudang.min())

        futurex2 = np.stack((futurex20, futurex21, futurex22), axis=1)
        futurepre2 = np.matmul(futurex2, w2)
        futurepre2[0][0] = int(futurepre2[0][0])
        distance2 = abs(futurepre2[0][0] - df.values[i][3])
        if(distance2<=df.values[i][3]*0.15 and top==0):
            total2=total2+1


        #bp神经网络模型
        year3 = np.array([2020, 2019, 2018, 2017, 2016, 2015])
        x3 = (year3 - year3.min()) / (year3.max() - year3.min())
        sampleinnorm = np.array([x3])
        sampleoutnorm = np.array([rank])

        # 训练模型

        maxepochs = 10000  # 迭代次数
        learnrate = 0.001
        errorfinal = 0.65 * 10 ** (-3)
        samnum = 6
        indim = 1
        outdim = 1
        hiddenunitnum = 8
        w31 = 0.5 * np.random.rand(hiddenunitnum, indim) - 0.1
        b31 = 0.5 * np.random.rand(hiddenunitnum, 1) - 0.1
        w32 = 0.5 * np.random.rand(outdim, hiddenunitnum) - 0.1
        b32 = 0.5 * np.random.rand(outdim, 1) - 0.1

        errhistory = []

        for j in range(maxepochs):
            hiddenout3 = logsig((np.dot(w31, sampleinnorm).transpose() + b31.transpose())).transpose()
            networkout3 = (np.dot(w32, hiddenout3).transpose() + b32.transpose()).transpose()
            err3 = sampleoutnorm - networkout3
            sse3 = sum(sum(err3 ** 2))  # 这个地方是求矩阵所有元素的和
            errhistory.append(sse3)
            if sse3 < errorfinal:
                break
            delta2 = err3
            delta1 = np.dot(w32.transpose(), delta2) * hiddenout3 * (1 - hiddenout3)
            dw2 = np.dot(delta2, hiddenout3.transpose())
            db2 = np.dot(delta2, np.ones((samnum, 1)))
            dw1 = np.dot(delta1, sampleinnorm.transpose())
            db1 = np.dot(delta1, np.ones((samnum, 1)))
            w32 += learnrate * dw2
            b32 += learnrate * db2
            w31 += learnrate * dw1
            b31 += learnrate * db1

        # 进行结果预测
        futureyear3 = np.array([2021])
        futureyear3=(futureyear3-year3.min())/(year3.max()-year3.min())
        # futurerank=np.array([round(i+1/300,4)])
        sampleinnorm31=np.array([futureyear3])
        hiddenout31 = logsig((np.dot(w31, sampleinnorm31).transpose() + b31.transpose())).transpose()
        networkout31 = (np.dot(w32, hiddenout31).transpose() + b32.transpose()).transpose()
        networkout31[0][0]=int(networkout31[0][0])
        # futurerank = getrank(2019,df.values[i][7])
        loss3 = abs(networkout31 -df.values[i][3])
        if(loss3<=df.values[i][3] and top==0):
            total3=total3+1

    print(total1,total2,total3)





if __name__ == '__main__':
    main()

