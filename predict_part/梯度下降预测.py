import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlwt
#通过分数和投档线，建立二元回归模型，数据从2005到2020


# 写入数据
storepath = "梯度下降预测河南2021年的录取排名和误差值.xls"
# 存储数据
book = xlwt.Workbook(encoding="utf-8", style_compression=0)
sheet = book.add_sheet("梯度下降预测河南2021年的录取排名和损失值", cell_overwrite_ok=True)
col = ("学校", "预测录取排名", "与实际误差值", "数据标记")  # 元组
for i in range(0, 4):
    sheet.write(0, i, col[i])

#加载样本
year=np.array([2015,2016,2017,2018,2019,2020])
toudang=np.array([529,523,484,499,502,544])
num=len(year)

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
    # print(rank)



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


    #进行预测
    futureyear=np.array([2021])
    futuretoudang=np.array([518])
    futurex0=np.ones(1)
    futurex1=(futureyear-year.min())/(year.max()-year.min())
    futurex2=(futuretoudang-toudang.min())/(toudang.max()-toudang.min())

    futurex=np.stack((futurex0,futurex1,futurex2),axis=1)
    futurepre=np.matmul(futurex,w)
    # sum=(futurepre[0][0]+futurepre[1][0]+futurepre[2][0]+futurepre[3][0]+futurepre[4][0]+futurepre[5][0])/6
    futurepre[0][0]=int(futurepre[0][0])
    distance=abs(futurepre[0][0]-df.values[i][3])
    #print(futurepre)

    #写入数据
    sheet.write(i + 1, 0, df.values[i][1])
    sheet.write(i + 1, 1, futurepre[0][0])
    sheet.write(i + 1, 2, distance)
    if (top == 1):
        sheet.write(i + 1, 3, "数据本身异常")
    else:
        sheet.write(i + 1, 3, "数据本身无异常")
    print("这是进行到了第%d次" % (i + 1))


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
    # # #plt.subplot(1,2,2)
    # plt.figure()
    # pred=pred.reshape(-1)
    # plt.plot(year,y,color="red",marker="o",label="实际分数")
    # plt.plot(year,pred,color="blue",marker="o",label="预测分数")
    # plt.xlabel("年份",fontsize=15)
    # plt.ylabel("分数",fontsize=15)
    #
    # plt.legend()
    # plt.show()

book.save(storepath)
