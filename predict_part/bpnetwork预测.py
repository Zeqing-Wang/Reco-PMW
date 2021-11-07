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


    #写入数据
    storepath = "BP预测河南2021年的录取排名和误差值.xls"
    #存储数据
    book = xlwt.Workbook(encoding="utf-8", style_compression=0)
    sheet = book.add_sheet("BP预测河南2021年的录取排名和损失值", cell_overwrite_ok=True)
    col = ("学校", "预测录取排名","与实际误差值","数据标记")  # 元组
    for i in range(0, 4):
        sheet.write(0, i, col[i])

    #开始处理数据

    #这里是将一些排名波动过大的学校删去
    #top=0
    year=np.array([2020,2019,2018,2017,2016,2015])
    x1 = (year - year.min()) / (year.max() - year.min())
    #ability=np.array([1,1,1,1,1,1])
    height=height-1
    for i in range(0,height):
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

        # 训练模型

        maxepochs = 10000  # 迭代次数
        learnrate = 0.001
        errorfinal = 0.65 * 10 ** (-3)
        samnum = 6
        indim = 1
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

        # 进行结果预测
        futureyear = np.array([2021])
        futureyear=(futureyear-year.min())/(year.max()-year.min())
        # futurerank=np.array([round(i+1/300,4)])
        sampleinnorm1=np.array([futureyear])
        hiddenout1 = logsig((np.dot(w1, sampleinnorm1).transpose() + b1.transpose())).transpose()
        networkout1 = (np.dot(w2, hiddenout1).transpose() + b2.transpose()).transpose()
        networkout1[0][0]=int(networkout1[0][0])
        # futurerank = getrank(2019,df.values[i][7])
        loss = abs(networkout1 -df.values[i][3])
        # print(loss)


        #将数据写入excel表中
        sheet.write(i+1,0,df.values[i][1])
        sheet.write(i+1,1,networkout1[0][0])
        sheet.write(i+1,2,loss[0][0])
        if (top==1):
            sheet.write(i+1,3,"数据本身异常")
        else:
            sheet.write(i+1,3,"数据本身无异常")
        print("这是进行到了第%d次"%(i+1))


        # # #绘图,可视化
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        #
        # plt.figure()
        # sampleoutnorm = sampleoutnorm.reshape(-1)
        # networkout=networkout.reshape(-1)
        # #print(networkout)
        # # networkout1=np.array(networkout1)
        # # networkout1=networkout1(-1)
        # #plt.plot(futureyear[0], networkout1[0], color="green", marker="p", label="预测年份排名")
        # plt.plot(year, sampleoutnorm, color="red", marker="o", label="实际排名")
        # plt.plot(year, networkout, color="blue", marker="o", label="预测排名")
        # plt.xlabel("年份", fontsize=15)
        # plt.ylabel("排名", fontsize=15)
        #
        # plt.legend()
        # plt.show()

    book.save(storepath)
    #print("%d"%top)


if __name__ == '__main__':
    main()
