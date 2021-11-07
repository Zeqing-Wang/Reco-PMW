# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
if __name__ =='__main__':
    df_rank = pd.read_csv('./data/ruankeAll.csv',encoding='ansi')
    df = pd.read_csv('./BPprodict.csv',encoding='ansi')
    print(df,df_rank)
    df_f = pd.merge(left=df_rank,right=df,sort=False,how='inner')
    print(df_f)
    # for i in range(len(df_rank)):
    #     flag = False
    #     for j in range(len(df)):
    #         if df_rank[i][1] in df[j][1]: # 排名的名称是不全的 分数是全的
    #             flag = True
    #             break
    #     if flag == False:
    #         df = df.drop([j])
    df_f.to_csv('recListProdict.csv', index=False)
    #df.to_csv('fill.csv', index=True)