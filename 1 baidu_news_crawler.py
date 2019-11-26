#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 13:32:03 2018
crawl the new related with key word from Baidu News 

@author: situ
"""
import numpy as np
import pandas as pd
import os
import re
import time
import requests
from urllib.parse import urlencode
from lxml import etree
from multiprocessing import Pool

class baidu_news:
    
    def __init__(self,word,bt_ymd,et_ymd,headers):
        self.word = word
        self.bt_ymd = bt_ymd
        self.et_ymd = et_ymd
        self.headers = headers
        self.mode = "title"
    
    def get_url(self,page):
        bt = self.bt_ymd+" 00:00:00"
        et = self.et_ymd+" 00:00:00"
        bts = int(time.mktime(time.strptime(bt, "%Y-%m-%d %H:%M:%S")))#时间戳
        ets = int(time.mktime(time.strptime(et, "%Y-%m-%d %H:%M:%S")))
        
        pn = 20*(page-1)# 页码对应：0 20 40 60
        if self.mode=="news":
            qword = urlencode({'word': self.word.encode('utf-8')})
            url = "http://news.baidu.com/ns?%s&pn=%d&cl=2&ct=1&tn=newsdy&rn=20&ie=utf-8&bt=%d&et=%d"%(qword,pn,bts,ets)
        if self.mode=="title": 
            qword = self.word
            #url可能需要常常根据网站的更新程度来修改
            url = "https://www.baidu.com/s?tn=news&rtt=1&bsst=1&cl=2&wd="+qword+"&medium=1&gpc=stf%3D"+str(bts)+"%2C"+str(ets)+"%7Cstftype%3D2&pn="+str(pn)
        return url



    def crawl(self,word):
        self.word = word
        i = 1
        is_nextpage=True
        newsData = pd.DataFrame()
        while is_nextpage:
            print("--------------正在爬取【%s】第%d页新闻----------------"%(self.word,i))
            url = self.get_url(i)
            print(url)

            result = requests.get(url,timeout=60,headers=self.headers)
            if result.status_code==200:
                print("\n请求成功")
            result.encoding = 'utf-8'
            selector = etree.HTML(result.text)  
            if self.mode=="news":
    
                for item in selector.xpath('//*[@class="result"]'):
        #            item = selector.xpath('//*[@class="result"]')[0]
                    newsdict = {"title":[0],"date":[0],"time":[0],"source":[0],
                                "abstract":[0],"href":[0]}
                    onenews = pd.DataFrame(newsdict)
                    
                    onenews["title"] = item.xpath('h3/a')[0].xpath("string(.)").strip()
                    print(onenews["title"])
                    onenews["href"] = item.xpath('h3/a/@href')[0]
                    info = item.xpath('div')[0].xpath("string(.)")
                    onenews["source"] , onenews["date"] , onenews["time"]= info.split()[:3]
                    onenews["abstract"] = " ".join(info.split()[3:len(info.split())-1])
                    newsData = newsData.append(onenews)
            if self.mode=="title":
                for item in selector.xpath('//*[@class="result"]'):
    #                item = selector.xpath('//*[@class="result"]')[0]
                    newsdict = {"title":[0],"date":[0],"time":[0],"source":[0],"href":[0]}
                    onenews = pd.DataFrame(newsdict)
                    
                    onenews["title"] = item.xpath('h3/a')[0].xpath("string(.)").strip()
                    onenews["href"] = item.xpath('h3/a/@href')[0]
                    info = item.xpath('div')[0].xpath("string(.)")
    #                print(info)
                    #如果新闻是今天发的，则会显示“X小时前”，则日期改成今天
                    if len(re.findall(r"小时前",info.split()[1]))>0:
                        onenews["source"] = info.split()[0]
                        nowtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        onenews["date"] = "%s年%s月%s日"%tuple((nowtime.split()[0].split("-")))
                        onenews["time"] = nowtime.split()[1][:5]#只取分秒                        
                    
                    else:
                        onenews["source"] , onenews["date"] , onenews["time"]= info.split()[:3]
                    newsData = newsData.append(onenews)
            page_info = selector.xpath('//*[@id="page"]/a[@class="n"]/text()')
            print(page_info)
            if len(page_info)>=1 and "下一页>" in page_info:
                is_nextpage=True
                i=i+1
            else:
                is_nextpage=False
        newsData["keyword"] = self.word
        newsData.to_csv(self.word+"_"+self.bt_ymd+"_"+self.et_ymd+"_"+self.mode+".csv",index = False,encoding = "gb18030")

def main():
#    os.chdir("E:/graduate/Paper")
    # 输入爬取新闻的起止时间,浏览器表头（可能常常需要更改）
    bt_ymd = "2018-07-01"
    et_ymd = "2019-06-30"
    headers = {
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',
        'Host':'www.baidu.com',
#       'Referer':'http://tieba.baidu.com/i/i/fans?u=4f5fe69d8ee4b9904d16',
        'Cookie':'BIDUPSID=7C2C739A7BA8C15B187303565C792CA0; PSTM=1509410172; BD_UPN=12314753; BAIDUID=70698648FD1C0D4909420893B868092B:FG=1; MCITY=-%3A; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; BDUSS=N5eGZLbWZ5eWNuSTc5TUpobUIxWXU3ZmpoQklSUGJNZ1R5cnIwLTd6LWdBRVJkRVFBQUFBJCQAAAAAAAAAAAEAAAA1izQO0sDIu9DS0MQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKBzHF2gcxxdZ1; pgv_pvi=166330368; ___wk_scode_token=Ct4MH%2FuNEgumb9NGCk8o1Aj%2BjCUcLU2ClmExi0Qz51M%3D; BD_CK_SAM=1; PSINO=7; BDRCVFR[PaHiFN6tims]=9xWipS8B-FspA7EnHc1QhPEUf; BDRCVFR[C0p6oIjvx-c]=mk3SLVN4HKm; BD_HOME=1; BDRCVFR[feWj1Vr5u3D]=I67x6TjHwwYf0; delPer=0; H_PS_PSSID=; sug=3; sugstore=1; ORIGIN=2; bdime=0; H_PS_645EC=f263%2FGdJfRrManRLCydAHWcUoMS0z2QF37c4uymvBok2x75KBHmMBsxhzWSqrwKXegg9lBNs; BDSVRTM=104'}
     #初始化
    baidu_news_crawl = baidu_news(word = "",bt_ymd = bt_ymd,et_ymd=et_ymd,headers=headers)
    # 从搜索关键词表读取关键词
    keywords = pd.read_csv("E:/graduate/Paper/code/keywords.csv",encoding = "gbk") 

    for j in range(6):
        cl = keywords["class"][j]
        para = keywords["keywords"][j].split(",")
        full_class_name = os.path.join("E:/graduate/Paper/renew_data",cl)
        if not os.path.exists(full_class_name):
            os.makedirs(full_class_name) 
        os.chdir(full_class_name)
        
    
    #也可以手动输入关键词
    #para = ["中国贫富差距"]
            
        p=Pool(8)
        p.map(baidu_news_crawl.crawl,para)      
        p.close()
        p.join()
        print(cl +"新闻爬取完成，请打开【"+os.getcwd()+"】查看详情")


if __name__ == '__main__':
    time_start=time.time()
    main()    
    time_end=time.time()
    m, s = divmod(time_end-time_start, 60)
    h, m = divmod(m, 60)
    print ("totally cost:   %02d:%02d:%02d" % (h, m, s)) #00:02:31

