#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 08:53:01 2018
按年份、关键词分层抽样，每个大类划分训练集测试集，生成训练集表格和测试集表格

@author: situ
"""

# 分层抽样   
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import time
import os
import re
import pandas as pd
    
def sampling(subpath):
    file_names = os.listdir(subpath)
    file_name = [f for f in file_names if len(re.findall(r"all_df.csv",f))>0][0] #只读取后缀名为csv的文件
    all_df = pd.read_csv(os.path.join(subpath,file_name),encoding="gb18030",engine="python")
    all_df["sample"] = all_df.apply(lambda line: line["keyword"]+"-"+str(line["year"]),axis=1)   
    x = all_df.drop("sample",1)
    label = all_df["sample"]
    x_train,x_test,_,_ = train_test_split(x,label,test_size=0.6,random_state=1994)
    x_train.to_csv(os.path.join(subpath,"train1.csv"),encoding="gb18030",index=False) #需要人工打标签的
    x_test.to_csv(os.path.join(subpath,"test1.csv"),encoding="gb18030",index=False)  


def main():

    os.chdir("E:/graduate/Paper/")
    path = "E:/graduate/Paper/raw_data"

    #只做一个类别
    sp = "物价"
    sampling(os.path.join(path,sp))

    #并行方法，同时生成所有类别的训练集测试集
#    subpaths = [os.path.join(path,sp) for sp in os.listdir(path)]
#    p=Pool(len(subpaths))
#    p.map(sampling,subpaths)      
#    p.close()
#    p.join()    
    
if __name__ == '__main__':
    time_start=time.time()
    main()    
    time_end=time.time()
    print('totally cost',time_end-time_start)    #3s
    
    
    
    