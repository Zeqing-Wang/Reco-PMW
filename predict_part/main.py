import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlwt

def logsig(x):

    return 1/(1+np.exp(-x))


def main():

    # 读入数据
    file="rankMark.csv"
    df=pd.read_csv(file)
    height,weight=df.shape
    #print(height)




    #写入数据
    storepath = "预测2020年的录取排名和误差值.xls"
    #存储数据
    book = xlwt.Workbook(encoding="utf-8", style_compression=0)
    sheet = book.add_sheet("预测2020年的录取排名和损失值", cell_overwrite_ok=True)
    col = ("学校", "预测录取排名","与实际误差值")  # 元组
    for i in range(0, 3):
        sheet.write(0, i, col[i])




    #数据预处理
    year = np.array([2018,2017,2016,2015])
    for i in range(0,300):
        rank1=[]#临时读入的高校排名
        for j in range(0,4):
            rank1.append(i+1)

        #print(rank1)

        # #rank1=np.array(rank1)
        rank=[]
        for j in range(0,4):
            rank.append(round(rank1[j]/300,4)) #这个地方是保留四位小数

        #print(rank)

        score=[]
        for j in range(8,12):
            score.append(df.values[i][j])
        #print(score)
        #这个地方将分数转化成排名
        for j in range(0,4):
            score[j] = getrank(year[j], score[j])

        #print(score)

        #归一化
        x1 = (year - year.min()) / (year.max() - year.min())
        x2=rank
        sampleinnorm=np.array([x1,x2])
        sampleoutnorm=np.array([score])







        #训练模型

        maxepochs =6000  #迭代次数
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
            sse = sum(sum(err ** 2))#这个地方是求矩阵所有元素的和
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

        # for j in range(len(errhistory)):
        #     print(errhistory[j])





        # 进行结果预测
        futureyear = np.array([2019])
        futureyear=(futureyear-year.min())/(year.max()-year.min())
        futurerank=np.array([round(i+1/300,4)])
        sampleinnorm1=np.array([futureyear,futurerank])
        hiddenout1 = logsig((np.dot(w1, sampleinnorm1).transpose() + b1.transpose())).transpose()
        networkout1 = (np.dot(w2, hiddenout1).transpose() + b2.transpose()).transpose()

        futurerank = getrank(2019,df.values[i][7])
        loss = abs(networkout1 -futurerank)



        #将数据写入excel表中
        sheet.write(i+1,0,df.values[i][1])
        sheet.write(i+1,1,networkout1[0][0])
        sheet.write(i+1,2,loss[0][0])
        print("这是进行到了第%d次"%(i+1))



        #print(networkout1,loss)
        #
        # #绘图,可视化
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # # # # plt.figure()
        # # # # plt.plot(errhistory[0:500])
        # # # # plt.xlabel("迭代次数", fontsize=15)
        # # # # plt.ylabel("损失值", fontsize=15)
        # # # # plt.show()
        # #
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

if __name__ == '__main__':
    main()
