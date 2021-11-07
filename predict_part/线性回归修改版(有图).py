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


    #输入
    for i in range(0,1):
        train_x0 = np.array([df.values[i][14], df.values[i][13], df.values[i][12], df.values[i][11]])
        train_x1=np.array([df.values[i][13],df.values[i][12],df.values[i][11],df.values[i][10]])
        train_x2=np.ones(len(train_x0))  #这个地方是b
        train_x=np.stack((train_x2,train_x0,train_x1),axis=1)
        train_y=np.array([df.values[i][12],df.values[i][11],df.values[i][10],df.values[i][9]])
        train_y=np.array(train_y).reshape(-1,1)
        # test_x=np.array([[df.values[i][10],df.values[i][9]]])
        # print(train_x)
        # print(train_y)
        # print(test_x)
        #
        xt=np.transpose(train_x)
        xt_1=np.linalg.inv(np.matmul(xt,train_x))
        xt_1_xt=np.matmul(xt_1,xt)
        w=np.matmul(xt_1_xt,train_y)
        w=w.reshape(-1)
        # print(w)
        #
        # #拟合的直线
        y_pred=w[1]*train_x0+w[2]*train_x1+w[0]
        print(y_pred)
        X1,X2=np.meshgrid(train_x0,train_x1)
        Y_PRED=w[0]+w[1]*X1+w[2]*X2
        # print(y_pred)

        #最终的预测
        # y_pred=w[0]*test_x[0][0]+w[1]*test_x[0][1]+w[2]
        #print(y_pred)

        #绘制3D散点图
        plt.rcParams['font.sans-serif']=['SimHei']

        fig=plt.figure(figsize=(8,8))
        ax3d=Axes3D(fig)
        # ax3d.view_init(elev=0,azim=-90)

        ax3d.scatter(train_x0,train_x1,train_y,color="b",marker="o",label="已知排名")
        ax3d.scatter(train_x0, train_x1, y_pred, color="r", marker="*",label="预测排名")
        ax3d.plot_wireframe(X1,X2,Y_PRED,color="c",linewidth=0.5,label="拟合平面")

        ax3d.set_xlabel('score1',color='r',fontsize=16)
        ax3d.set_ylabel('score2', color='r', fontsize=16)
        ax3d.set_zlabel('score3', color='r', fontsize=16)

        #ax3d.set_yticks([1,2,3])
        #ax3d.set_zlim3d(60,100)

        plt.suptitle("线性回归预测模型",fontsize=20)
        plt.legend(loc="upper left")

        plt.show()



if __name__ == '__main__':
    main()