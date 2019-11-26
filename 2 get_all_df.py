#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 12:45:50 2018
1 合并表格+最初的文本预处理：得到六大类新闻合并表all_df.csv
去重（只去掉除搜索关键词外其他变量都一样的）
去除非中国的，外国新闻标记
去除某些网站
去除非需要时间段的


2 读取六个all_df.csv：得到统计年份、类别的表summary.xlsx


@author: situ
"""

import pandas as pd
import numpy as np
import os
import re
import time

def batch_read_csv(full_class_name,encoding="gb18030"):
    file_names = os.listdir(full_class_name)
    file_names = [f for f in file_names if len(re.findall(r"title.csv",f))>0] #只读取后缀名为csv的文件
    all_df = pd.DataFrame()
    for i in range(len(file_names)):
        try:
            temp_df = pd.read_csv(os.path.join(full_class_name,file_names[i]),encoding=encoding)
        except:
            temp_df = pd.read_csv(os.path.join(full_class_name,file_names[i]),encoding=encoding,engine =  "python")

        all_df = pd.concat([all_df,temp_df],axis=0,ignore_index=True)
#        print(all_df.head())
    l1 = len(all_df)
    print("处理前新闻有%d条"%(l1))
    #去重
    var_list = list(all_df.columns)
    var_list.remove("keyword")
    all_df.drop_duplicates(subset = var_list,inplace=True)#只有完全一样的才删除
    l2 = len(all_df)
    print("删除重复新闻%d条"%(l1-l2))
#    all_df.to_csv("F:/毕业论文/raw_data/经济/economics.csv",encoding="gb18030",index=False)

    
    #all_df删除重复行后，index有空缺，重新索引
    all_df = all_df.reset_index(drop=True)
    return all_df

def rm_foreign_news(full_class_name):
    all_df = batch_read_csv(full_class_name)
    is_foreign = all_df["title"].apply(is_foreign_name)
    is_foreign = np.array(is_foreign)
    dele_num = np.where(is_foreign>0)[0]
#    all_df["title"][dele_num]
    all_df.drop(dele_num,inplace = True)
    all_df = all_df.reset_index(drop=True)
    print("删除外国新闻%d条"%len(dele_num))
    ad_web = np.array(all_df["source"].apply(is_ad_web))
    dele_num = np.where(ad_web==1)[0]
    all_df.drop(dele_num,inplace = True)
    print("删除广告新闻%d条"%len(dele_num))
    
    all_df = all_df.reset_index(drop=True)    
    return all_df
    
def is_ad_web(text):
    ad_web = open("./code/ad_web.txt",'rb').read().decode('utf-8').split("\r\n")
    isadweb = sum([text==token for token in ad_web])
    return isadweb

    
def is_foreign_name(text):
    countryname =  open("./code/countryname.txt", 'rb').read().decode('utf-8').split('\r\n')
    isforeign = sum([len(re.findall(token,text))>=1 for token in countryname])>0
    notchina = sum([len(re.findall(token,text))>=1 for token in ["北京","中国","内蒙古"]])==0
    return  isforeign and notchina
    
def dele_wrongtime(full_class_name,bt_ymd,et_ymd):

    all_df = rm_foreign_news(full_class_name)
    
    #统计年份季度的新闻数:新增年份列与季度列
    time = pd.to_datetime(all_df["date"],format="%Y年%m月%d日")
    dele_num1 = np.where(bt_ymd>time)[0]
    dele_num2 = np.where(time>et_ymd)[0]
    dele_num = np.append(dele_num1 , dele_num2)
    print("删除非需要时间段新闻%d条"%len(dele_num))
    all_df.drop(dele_num,inplace = True)
    all_df = all_df.reset_index(drop=True)
    print("处理后新闻有%d条"%(len(all_df)))
    return all_df

def get_all_df(path,bt_ymd,et_ymd):
    class_names  = os.listdir(path)
    for class_name in class_names:
        print("正在合并【%s】类新闻......"%class_name)
        full_class_name = os.path.join(path,class_name)
#        all_df = batch_read_csv(full_class_name)  #原版
        all_df = dele_wrongtime(full_class_name,bt_ymd,et_ymd)
        all_df.to_csv(os.path.join(full_class_name,class_name+"all_df.csv"),encoding="gb18030",index=False)    
    
    #新闻情况统计表
def onestat(param):
    full_class_name,excel_writer = param
    class_name = os.path.split(full_class_name)[1]
    all_df = pd.read_csv(os.path.join(full_class_name,class_name+"all_df.csv"),encoding="gb18030")
    
    source_websites = list(all_df["source"].value_counts().index)[:15]
    keywords = list(all_df["keyword"].value_counts().index)
    ram = np.random.randint(0, high=len(all_df)-1, size=4, dtype='l')
    examples = all_df.ix[ram,"title"]
    examples = "\n".join(examples)
    
    time = pd.to_datetime(all_df["date"],format="%Y年%m月%d日")
    all_df["year"] = list(pd.DatetimeIndex(time).year)
    all_df["quarter"] = list(pd.DatetimeIndex(time).quarter)
    
    date_stat = all_df["title"].groupby([all_df["year"],all_df["quarter"]]).count()
    date_stat.to_excel(excel_writer,class_name)
    return source_websites,keywords,len(all_df),examples

def allstat(path,writer):
    classes = os.listdir(path)
    subpaths = [os.path.join(path,sp) for sp in os.listdir(path)]
    param = [(sp,writer) for sp in subpaths]      
    result = list(map(onestat,param))   
    source_websites = [[tup[0] for tup in result]][0]
    source_websites_str = ["，".join(name) for name in source_websites]
    keywords = [[tup[1] for tup in result]][0]
    keywords_str = ["，".join(k) for k in keywords]
    num_class = [tup[2] for tup in result]
    examples = [tup[3] for tup in result]
    summary = pd.DataFrame(data = np.array([classes,keywords_str,source_websites_str,num_class,examples]).T,
                                           columns = ["class","keywords","source","num_class","examples"])

    
    summary.to_excel(writer,'summary')
    writer.save()
    print("excel已生成，请打开【"+os.getcwd()+"】查看详情")



def main():
    bt_ymd = "2018-07-01"
    et_ymd = "2019-06-30"
    os.chdir("E:/graduate/Paper")
    path = "./renew_data"
    # 生成合并处理版
    get_all_df(path,bt_ymd,et_ymd)
    
    # 统计新闻信息
    writer = pd.ExcelWriter('summary.xlsx')
    allstat(path,writer)


if __name__ == '__main__':
    time_start=time.time()
    main()    
    time_end=time.time()
    m, s = divmod(time_end-time_start, 60)
    h, m = divmod(m, 60)
    print ("totally cost:   %02d:%02d:%02d" % (h, m, s)) #00:02:31
    
