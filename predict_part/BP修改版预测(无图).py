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

    #写入数据
    storepath = "BP修改版预测河南2021年的录取排名和误差值.xls"
    #存储数据
    book = xlwt.Workbook(encoding="utf-8", style_compression=0)
    sheet = book.add_sheet("BP修改版预测河南2021年的录取排名和损失值", cell_overwrite_ok=True)
    col = ("学校", "预测录取排名","与实际误差值","数据标记")  # 元组
    for i in range(0, 4):
        sheet.write(0, i, col[i])



    #开始处理数据

    #这里是将一些排名波动过大的学校删去
    #top=0

    # 输入
    i = 0
    ji = 0
    while (i < height):

        try:

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

            maxepochs = 5000  # 迭代次数
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

            # 进行结果预测
            futureyear1 = np.array([df.values[i][10]])
            futureyear2 = np.array([df.values[i][9]])

            # futurerank=np.array([round(i+1/300,4)])
            sampleinnorm1=np.array([futureyear1,futureyear2])
            hiddenout1 = logsig((np.dot(w1, sampleinnorm1).transpose() + b1.transpose())).transpose()
            networkout1 = (np.dot(w2, hiddenout1).transpose() + b2.transpose()).transpose()
            networkout1[0][0]=int(networkout1[0][0])
            # print(networkout1)
            # futurerank = getrank(2019,df.values[i][7])
            loss = abs(networkout1[0][0] -df.values[i][8])
            # print(loss)

            # 写入数据
            sheet.write(ji + 1, 0, df.values[i][2])
            sheet.write(ji + 1, 1, networkout1[0][0])
            sheet.write(ji + 1, 2, loss)
            if (top == 1):
                sheet.write(ji + 1, 3, "数据本身异常")
            else:
                sheet.write(ji + 1, 3, "数据本身无异常")
            print("这是进行到了第%d次" % (i + 1))
            ji=ji+1
            i=i+1
            # print(i,w)

        except:
            ji=ji-1
            i=i+1
            print("矩阵不可逆")

    book.save(storepath)






if __name__ == '__main__':
    main()
