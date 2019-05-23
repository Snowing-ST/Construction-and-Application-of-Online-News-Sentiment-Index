# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 01:18:04 2019
所有分指数整合一起，合成总指数
包括训练集、测试集、全集得到的指数

@author: situ
"""

import pandas as pd
import numpy as np
import os
import re


def batch_read_index(path):
    classes = os.listdir(path)
    all_index = pd.DataFrame()
    for asp in classes:
        sp = os.path.join(path,asp,asp+ALL_INDEX_NAME)
        temp_df = pd.read_csv(sp,encoding="gb18030",engine =  "python")
        print(asp)
        temp_df.set_index(["year","quarter"], inplace=True) #列变索引
        temp_df.columns = temp_df.columns.map(lambda x:asp+x) #更改列名
        all_index = pd.concat([all_index,temp_df],axis=1)
        
    
    all_index.reset_index(inplace=True) #索引变列
    train_list = [tr for tr in all_index.columns if len(re.findall(r"train",tr))>0]
    all_index["index_train"] = np.sum(all_index.ix[:,train_list],axis=1)-500
    
    test_list = [tr for tr in all_index.columns if len(re.findall(r"test",tr))>0]
    all_index["index_test"] = np.sum(all_index.ix[:,test_list],axis=1)-500
    
    all_list = [tr for tr in all_index.columns if len(re.findall(r"all",tr))>0]
    all_index["index_all"] =  np.sum(all_index.ix[:,all_list],axis=1)-500
    all_index.to_csv(ALL_INDEX_NAME,encoding="gb18030",index=False)

def main():  
    path = "E:/graduate/Paper/raw_data"
    os.chdir("E:/graduate/Paper")
    ALL_INDEX_NAME = "CNN总指数.csv"
    batch_read_index(path)
    
if __name__ == '__main__':
    main()    
        
