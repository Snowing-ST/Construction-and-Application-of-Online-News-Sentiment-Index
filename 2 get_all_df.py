#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 12:45:50 2018
爬虫文本预处理：
去重（只去掉除搜索关键词外其他变量都一样的）
去除非中国的，外国新闻标记
去除2018年第三季度的
去除某些网站

从多个关键词爬取的原始数据中
1. 得到统计年份、类别的表
2. 得到新闻合并表all_df


@author: situ
"""

import pandas as pd
import numpy as np
import os
import re
import time

def batch_read_csv(path,encoding="gb18030"):
    file_names = os.listdir(path)
    file_names = [f for f in file_names if len(re.findall(r"title.csv",f))>0] #只读取后缀名为csv的文件
    all_df = pd.DataFrame()
    for i in range(len(file_names)):
        try:
            temp_df = pd.read_csv(os.path.join(path,file_names[i]),encoding=encoding)
        except:
            temp_df = pd.read_csv(os.path.join(path,file_names[i]),encoding=encoding,engine =  "python")

        all_df = pd.concat([all_df,temp_df],axis=0,ignore_index=True)
#        print(all_df.head())
    l1 = len(all_df)
    print("%s,处理前新闻有%d条"%(path,l1))
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

def rm_foreign_news(path):
    all_df = batch_read_csv(path)
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
    
def stat(param):
    path = param[0]
    excel_writer = param[1]
    class_name = path.split("\\")[1]
    os.chdir("E:/graduate/Paper")
    path = "E:/graduate/Paper/raw_data"
    class_name = "生活状况"
    all_df = rm_foreign_news(os.path.join(path,class_name))
    
    #统计年份季度的新闻数:新增年份列与季度列
    time = pd.to_datetime(all_df["date"],format="%Y年%m月%d日")
    all_df["year"] = list(pd.DatetimeIndex(time).year)
    all_df["quarter"] = list(pd.DatetimeIndex(time).quarter)
    #只保留2018年上半年及之前的数据  
    is2018Q3 = all_df.apply(lambda line: line["year"]==2018 and line["quarter"]>2,axis=1)
    
    dele_num =np.where(np.array(is2018Q3))[0]
#    all_df.ix[dele_num,:]
    all_df.drop(dele_num,inplace = True)
    all_df = all_df.reset_index(drop=True)
    print("删除2018Q3新闻%d条"%len(dele_num))
    is2019 = all_df.apply(lambda line: line["year"]==2019,axis=1)
    
    dele_num =np.where(np.array(is2019))[0]
    all_df.drop(dele_num,inplace = True)
    all_df = all_df.reset_index(drop=True)
    print("删除2019新闻%d条"%len(dele_num))
    print("处理后新闻有%d条"%(len(all_df)))
    
    before2009 = all_df.apply(lambda line: line["year"]<2009,axis=1)
    dele_num =np.where(np.array(before2009))[0]
    all_df.drop(dele_num,inplace = True)
    all_df = all_df.reset_index(drop=True)
    print("删除2009年之前的新闻%d条"%len(dele_num))
    print("处理后新闻有%d条"%(len(all_df)))
    all_df.to_csv(os.path.join(path,class_name+"all_df.csv"),encoding="gb18030",index=False)    
    
    source_websites = list(all_df["source"].value_counts().index)[:15]
    keywords = list(all_df["keyword"].value_counts().index)
    ram = np.random.randint(0, high=len(all_df)-1, size=4, dtype='l')
    examples = all_df.ix[ram,"title"]
    examples = "\n".join(examples)
    
    date_stat = all_df["title"].groupby([all_df["year"],all_df["quarter"]]).count()
    
    date_stat.to_excel(excel_writer,class_name)
    return source_websites,keywords,len(all_df),examples



def main():
    os.chdir("E:/graduate/Paper")
#    os.chdir("F:/毕业论文/")
    path = "./raw_data"
    writer = pd.ExcelWriter('summary.xlsx')
    
    classes = os.listdir(path)
    subpaths = [os.path.join(path,sp) for sp in os.listdir(path)]
    param = [(sp,writer) for sp in subpaths]      
    result = list(map(stat,param))

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



if __name__ == '__main__':
    time_start=time.time()
    main()    
    time_end=time.time()
    print('totally cost',time_end-time_start)    #25s
    
