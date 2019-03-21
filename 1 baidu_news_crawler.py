#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 13:32:03 2018
类

@author: situ
"""
import time
from urllib.parse import urlencode
import pandas as pd
import os
import requests
import numpy as np
from lxml import etree
from multiprocessing import Pool

class baidu_news:
    
    def __init__(self,word,year):
        self.word = word
        self.year = year
        self.mode = "title"
    
    def get_url(self,page):
        bt = str(self.year)+"-01-01 00:00:00"
        et = str(self.year)+"-12-31 00:00:00"
        bts = int(time.mktime(time.strptime(bt, "%Y-%m-%d %H:%M:%S")))#时间戳
        ets = int(time.mktime(time.strptime(et, "%Y-%m-%d %H:%M:%S")))
        
        pn = 20*(page-1)# 页码对应：0 20 40 60
        if self.mode=="news":
            qword = urlencode({'word': self.word.encode('utf-8')})
            url = "http://news.baidu.com/ns?%s&pn=%d&cl=2&ct=1&tn=newsdy&rn=20&ie=utf-8&bt=%d&et=%d"%(qword,pn,bts,ets)
        if self.mode=="title":
            qword = "word=intitle%3A%28"+self.word+"%29"
            url = "http://news.baidu.com/ns?%s&pn=%d&cl=2&ct=0&tn=newstitledy&rn=20&ie=utf-8&bt=%d&et=%d"%(qword,pn,bts,ets)
        return url



    def crawl(self,word):
        self.word = word
        i = 1
        is_nextpage=True
        newsData = pd.DataFrame()
        while is_nextpage:
            print("--------------正在爬取【%s】%d年第%d页新闻----------------"%(self.word,self.year,i))
            url = self.get_url(i)
            print(url)
            result = requests.get(url,timeout=60)
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
                for item in selector.xpath('//*[@class="result title"]'):
    #                item = selector.xpath('//*[@class="result title"]')[0]
                    newsdict = {"title":[0],"date":[0],"time":[0],"source":[0],"href":[0]}
                    onenews = pd.DataFrame(newsdict)
                    
                    onenews["title"] = item.xpath('h3/a')[0].xpath("string(.)").strip()
                    onenews["href"] = item.xpath('h3/a/@href')[0]
                    info = item.xpath('div')[0].xpath("string(.)")
    #                print(info)
                    #如果新闻是今天发的，则会显示“X个小时前”，则日期改成今天
                    if (info.split()[-1]=="查看更多相关新闻>>" and len(info.split())==3) or len(info.split())==2:
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
        newsData.to_csv(self.word+"_"+str(self.year)+"_"+self.mode+".csv",index = False,encoding = "gb18030")

def main():
    os.chdir("E:/graduate/Paper")
    # 从搜索关键词表读取关键词
#    keywords = pd.read_csv("keywords.csv",encoding = "gbk") 
#    for cl in np.unique(keywords["class"]):
#        para = keywords["keyword"][keywords["class"]==cl].tolist()
#    para = keywords["keyword"][keywords["class"]=="消费"].tolist()
    
    #手动输入关键词
    para = [
"中国贫富差距"


            ]
    for y in range(2009,2019,1):
        baidu_news_crawl = baidu_news(word = "",year=y)
        p=Pool(len(para))
        p.map(baidu_news_crawl.crawl,para)      
        p.close()
        p.join()
    print("爬取成功，请打开【"+os.getcwd()+"】查看详情")


if __name__ == '__main__':
    main()

