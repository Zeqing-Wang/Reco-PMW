import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlwt


#numpy实现


# 写入数据
storepath = "线性回归预测河南2021年的录取排名和误差值.xls"
# 存储数据
book = xlwt.Workbook(encoding="utf-8", style_compression=0)
sheet = book.add_sheet("线性回归预测河南2021年的录取排名和损失值", cell_overwrite_ok=True)
col = ("学校", "预测录取排名", "与实际误差值", "数据标记")  # 元组
for i in range(0, 4):
    sheet.write(0, i, col[i])

#第一步加载样本数据
x=np.array([2015,2016,2017,2018,2019,2020])

file = "河南排名信息.csv"
df = pd.read_csv(file)
height, weight = df.shape

for i in range(0, height):
    top = 0  # 这个用来记录是否异常
    sum1 = (df.values[i][4] + df.values[i][5] + df.values[i][6] + df.values[i][7] + df.values[i][8] + df.values[i][9]) / 6
    if (df.values[i][3] >= (sum1 * 1.15) or df.values[i][3] <= (sum1 * 0.85)):
        # print("第%d次数据不符合要求"%i)
        top = 1
    rank = []  # 这个用来记录排名
    for j in range(9,3,-1):   #这个地方得从后往前遍历列表
        rank.append(df.values[i][j])


    #第二步学习模型
    y=rank
    #求均值
    meanX=np.mean(x)
    meanY=np.mean(y)

    #利用numpy广播原则
    sumXY=np.sum((x-meanX)*(y-meanY))#分子
    sumX=np.sum((x-meanX)*(x-meanX))#分母
    w=sumXY/sumX
    b=meanY-w*meanX

    #print("w=",w)
    #print("b=",b)

    # pred=w*x+b


    #第三步预测

    x_test=np.array([2021])
    y_pred=w*x_test+b
    y_pred=int(y_pred)
    distance=abs(y_pred-df.values[i][3])
    # print(sum,distance)

    #写入数据
    sheet.write(i + 1, 0, df.values[i][1])
    sheet.write(i + 1, 1, y_pred)
    sheet.write(i + 1, 2, distance)
    if (top == 1):
        sheet.write(i + 1, 3, "数据本身异常")
    else:
        sheet.write(i + 1, 3, "数据本身无异常")
    print("这是进行到了第%d次" % (i + 1))
    #print(y_pred)


    # #画图
    # plt.figure()
    # pred = pred.reshape(-1)
    # plt.plot(x, y, color="red", marker="o", label="实际分数")
    # plt.plot(x, pred, color="blue", marker="o", label="预测分数")
    # plt.xlabel("年份", fontsize=15)
    # plt.ylabel("分数", fontsize=15)
    #
    # plt.legend()
    # plt.show()

book.save(storepath)