import pandas as pd
import numpy as np
import xlwt

#将河南的数据进行去重的操作

def main():
    #加载文档
    file="rank.xlsx"
    df=pd.read_excel(file)
    #np.array(df)
    height,weight=df.shape

    # print(df.values)


    #删除重复
    df4=deletechongfu(df,height)
    #print(df4)

    #存储
    storepath = "河南.xls"
    height1, weight1 = df4.shape
    storedata(df4,storepath,height1)


# 3.保存数据
def storedata(df1, storepath,height1):
    book = xlwt.Workbook(encoding="utf-8", style_compression=0)
    sheet = book.add_sheet("河南版", cell_overwrite_ok=True)
    col = ("college","province","year","type","batch","min","min_rank","batch_min","gap")  # 元组,如果需要院校简介另加
    for i in range(0, 9):
        sheet.write(0, i, col[i])
    for i in range(0, height1):
        print("第%d条" % (i + 1))
        data = df1[i]
        for j in range(0, 9):
            sheet.write(i + 1, j, data[j])
    book.save(storepath)



#4.删除重复的
def deletechongfu(df,height):
    df4=df.values
    i=0
    while(i<height-2):
        if(df4[i][2]==df4[i+1][2]):
            df4=np.delete(df4,i+1,axis=0)
            i=i-1
            height=height-1
        i=i+1
    return df4


if __name__ == "__main__":
    main()




