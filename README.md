# Reco-PMW

A Novel College Entrance Filling Recommendation Algorithm Based on Score Line Prediction and Multi-weight Fusion(Reco-PMW)Code and Data

## 项目说明

这是高考志愿推荐算法（Reco-PMW）的代码以及原始数据。包含了论文中用到的所有数据以及流程代码。result_new.xlsx为与其他系统对比的详细结果

### 代码说明

本项目包含三个部分：

1. orgindata
2. predict_part
3. reco_part

如果需要直接最终推荐可以直接运行reco_part的rec.py文件

#### orgindata

这一部分为该项目通过爬虫、下载等方式获取到的河南省高考相关数据（包括了：2021年软科中国大学排名（主榜）、河南2015-2021年理科最低投档分数线、河南2015-2021理科一分一段表），爬虫部分相关代码并不在项目中，如果需要请联系我。

#### predict_part

这一部分为算法最低投档位次预测部分代码以及数据，同时包括了运行结果以及对比效果。预测算法部分包括了文中使用的三个对比算法，BP神经网络、梯度下降法。线性回归。

#### reco_part

这一部分为算法的推荐部分，包括了推荐部分使用到的最终处理后数据，推荐算法主体、遗传算法调优。data文件夹是算法使用到的最终处理后数据，dataprocess文件夹包含了处理数据过程中使用到的代码，不过这一部分并未标注，可以直接使用处理完后的数据（data文件夹）。部分重要文件介绍如下：

1. GAoptim.py

算法参数调优代码。使用了遗传算法进行参数调优，运行main函数即可。代码实现部分参考了[Luqiang_Shi](https://blog.csdn.net/Luqiang_Shi/article/details/84619456)的文章，在此表示感谢！

2. rec.py

推荐的最终实现部分，大体有三个部分：

##### 遗传算法调优时

在进行遗传算法调优时，直接运行GAoptima.py即可  这里调用的recGA中的代码

##### 结果画图时

调用rec.py代码中plot()函数即可

##### 推荐时

调用rec.py中recAll函数，其中a,b,c,d为已经经过调优后的参数，在参数mark输入2021年河南理科考生分数即可。代码默认在三个类别中各给出两所院校，可以根据程序输出的三个类别入选数量，在函数final中对chonglen、wenlen、baolen进行调整。

### 联系我们

如果您对项目有任何疑问，请联系我的邮箱：920626166@qq.com。

## 最后

希望这个项目对您有帮助！
