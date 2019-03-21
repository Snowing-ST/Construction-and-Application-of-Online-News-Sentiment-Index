# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 17:36:53 2019
文本预处理函数

@author: situ
"""

import numpy as np
import re
import jieba
import thulac 
import pkuseg
from jieba.analyse import extract_tags
from collections import Counter
import operator



def get_text(text):
    text = text.dropna() 
    len(text)
    text=[t.encode('utf-8').decode("utf-8") for t in text] 
    return text


def get_stop_words(file='./code/stopWord.txt'):
    file = open(file, 'rb').read().decode('utf8').split(',')
    file = [line.strip() for line in file]
    return set(file)                                         #查分停用词函数


def rm_tokens(words):                                        # 去掉一些停用词和完全包含数字的字符串
    words_list = list(words)
    stop_words = get_stop_words()
    for i in range(words_list.__len__())[::-1]:
        if words_list[i] in stop_words:                      # 去除停用词
            words_list.pop(i)
        elif words_list[i].isdigit():
            words_list.pop(i)
    return words_list



def rm_char(text):

    text = re.sub('\x01', '', text)                        #全角的空白符  感觉问好 感叹号不应该删除
    text = re.sub('\u3000', '', text) 
    text = re.sub(']'," ", text) 
    text = re.sub('\['," ", text) 
    text = re.sub('"'," ", text) 
    text = re.sub(r"[\)(↓%·▲】&【]","", text) 
    text = re.sub(r"[\d（）《》〖〗><‘’“”""''.,_:|-…]"," ",text,flags=re.I)
    text = re.sub('\n+', " ", text)
    text = re.sub('[，、：。；——]', " ", text)
    text = re.sub(' +', " ", text)
    text = re.sub(';', " ", text)
    return text

def convert_doc_to_wordlist(text, tool = "jieba",cut_all=False,mode = "accuracy"):
    text = get_text(text)
    
    sent_list = map(rm_char, text)                       # 去掉一些字符，例如\u3000
    if tool=="jieba":
        jieba.load_userdict("./code/dict.txt")
        if mode == "accuracy":
            word_2dlist = [rm_tokens(jieba.cut(part, cut_all=cut_all))
                       for part in sent_list]                     # 分词
        if mode == "search":
            word_2dlist = [rm_tokens(jieba.cut_for_search(part))
                       for part in sent_list]
    if tool=="thulac":
        thu1 = thulac.thulac(user_dict="./code/dict_thu1.txt",seg_only=True)  #只进行分词，不进行词性标注
        word_2dlist = [rm_tokens(thu1.cut(part, text=True).split()) for part in sent_list]
    if tool=="pku":
        seg = pkuseg.pkuseg(user_dict="./code/dict_thu1.txt")
        word_2dlist = [rm_tokens(seg.cut(part)) for part in sent_list]
    def rm_space_null(alist):
        alist = [s for s in alist if s not in [""," "]]
        return alist
    rm_space = [rm_space_null(ws) for ws in word_2dlist if len(ws)>0]
    return rm_space

def rm_low_high_freq(texts,low_freq=1,high_topK=10):#texts为包含多个句子的列表
    whole_text = []
    for doc in texts:
        whole_text.extend(doc.split())
    frequency_dict = Counter(whole_text)
    frequency_dict = sorted(Counter(whole_text).items(), key = operator.itemgetter(1), reverse=True) #=True 降序排列
#    print("the top %d wordcount is:\n" %(high_topK),frequency_dict[:high_topK],"/n")
    word_count = np.array(frequency_dict)
    print("原词典长度为%d"%len(word_count))
#    high_freq_w = [wc[0] for wc in word_count[:high_topK]]
    low_freq_w = word_count[word_count[:,1]==str(low_freq),0].tolist()
    dele_list = low_freq_w
    print("现词典长度为%d"%(len(word_count)-len(dele_list)))
#    dele_list = high_freq_w+low_freq_w
    rm_freq_texts = [[token for token in doc.split() if token not in dele_list] for doc in texts]
#    sum(np.array(list(map(len,rm_freq_texts)))==1)
    dele_num = np.where(np.array(list(map(len,rm_freq_texts)))<1)[0]
    #哪些新闻被删得只剩0 个或1个词
#    data.ix[dele_index,"title"]
#    data = data.drop(dele_num,inplace = False)
#    data = data.reset_index(drop=True)
    print("删除词数少于1的新闻%d条"%len(dele_num))
    new_texts = [" ".join(line) for line in rm_freq_texts if len(line)>0]

    return new_texts

def view_keywords(data,word_name = "word_seg",tag_name="label",topK=20):
    "用jieba看每个标签的关键词"
    def get_kw(text):
        return extract_tags(text, topK=topK, withWeight=True, allowPOS=())
    text_groupbyLabel = [" ".join(data[word_name][data[tag_name]==i]) for i in  range(-1,2)]
    news_kw = list(map(get_kw,text_groupbyLabel))
    
    for j in range(len(news_kw)):
        print("\n第"+str(j+1)+"类新闻的关键词：\n")
        for i in range(len(news_kw[j])):
            print(news_kw[j][i])


