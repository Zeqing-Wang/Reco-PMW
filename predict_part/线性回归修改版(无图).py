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

    # 写入数据
    storepath = "线性回归修改版预测河南2021年的录取排名和误差值.xls"
    # 存储数据
    book = xlwt.Workbook(encoding="utf-8", style_compression=0)
    sheet = book.add_sheet("线性回归预测河南2021年的录取排名和损失值", cell_overwrite_ok=True)
    col = ("学校", "预测录取排名", "与实际误差值", "数据标记")  # 元组
    for i in range(0, 4):
        sheet.write(0, i, col[i])


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
            if (df.values[i][8] >= (sum1 * 1.15) or df.values[i][8] <= (sum1 * 0.85)):
                # print("第%d次数据不符合要求"%i)
                top = 1

            # 写入数据
            sheet.write(ji + 1, 0, df.values[i][2])
            sheet.write(ji + 1, 1, y_pred)
            sheet.write(ji + 1, 2, distance)
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