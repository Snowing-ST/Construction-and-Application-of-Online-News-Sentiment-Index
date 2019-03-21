# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:17:51 2019
本脚本存放被别的包调用的函数
1. 生成由别人训练好的词向量替换后的词向量
2. 与word2vec+svm有关函数

@author: situ
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import re
import fileinput
from collections import Counter
import operator
from sklearn.preprocessing import scale


os.chdir("E:/graduate/Paper/code")
import text_preprocess as tp


def view_bar(text, num, total):
    """优化进度条显示"""
    rate = num / total
    rate_num = int(rate * 100)
    r = '\r' + text + '[%s%s]%d%%' % ("=" * rate_num, " " * (100 - rate_num), rate_num,)
    sys.stdout.write(r)
    sys.stdout.flush()


def get_all_news(TRAIN_DATA):
    """将所有方面的all_df.csv合并"""
    path = "./raw_data/"
    aspect = os.listdir(path)
    all_news_title = pd.DataFrame()
    for asp in aspect:
        subpath = os.path.join(path,asp)
        temp_df_name = [f for f in os.listdir(subpath) if len(re.findall(r"all_df.csv",f))>0]
        
        temp_df = pd.read_csv(os.path.join(subpath,temp_df_name[0]),encoding="gb18030",engine =  "python")
        all_news_title = pd.concat([all_news_title,temp_df],axis=0,ignore_index=True)
    all_news_title.to_csv(TRAIN_DATA,encoding="gb18030",index=False)    
    return all_news_title


def get_news_dic(TRAIN_DATA,DICTIONARY_DIC,WORD_FREQUENCY_LOW):
    """
    从所有新闻训练样本得到字典
    """
    dic = [] #存放词
    

    print('正在加载字典……')
    # 统计数据包总条数
    if os.path.isfile(DICTIONARY_DIC):#如果已有所有新闻生成的词典，则直接读取
        with open(DICTIONARY_DIC, "rb") as f:
            print("已存在由所有新闻生成的字典，直接读取")
            dic = pickle.load(f) 
    else:
        if os.path.isfile(TRAIN_DATA):
            print("已存在所有新闻合集，直接读取")
            all_news_title = pd.read_csv(TRAIN_DATA,encoding="gb18030",engine="python")
        else:
            all_news_title = get_all_news(TRAIN_DATA)
        
        tolal_line_num = len(all_news_title)
        print("共有新闻有%d条"%(tolal_line_num))
        #调用文本预处理函数
        x_train = tp.convert_doc_to_wordlist(all_news_title["title"],tool = "jieba",cut_all=False,mode ="accuracy")
        whole_text = []
        for line in x_train:
            whole_text.extend(line)

        frequency_dict = Counter(whole_text)    
#        frequency_dict = sorted(Counter(whole_text).items(), key = operator.itemgetter(1), reverse=True) #=True 降序排列
            
        for word in frequency_dict:
            if WORD_FREQUENCY_LOW < frequency_dict[word]:#去掉低频词
                dic.append(word)
        
        with open(DICTIONARY_DIC, 'wb') as f:
            pickle.dump(dic, f) # 把所有新闻的字典保存入文件

    print('字典加载完成,去除低频词后字典长度为%d'%(len(dic)))
    return dic
    

def get_word2vec(WORD2VEC_DIC, # 已预训练好的词向量文件地址
                 WORD2VEC_SUB, # 替换好的字典
                 TRAIN_DATA,DICTIONARY_DIC,WORD_FREQUENCY_LOW):# get_news_dic的变量
    """
    采用预训练好的部分词向量。仅使用部分，是为了节省内存。
    
    1. 遍历已训练好的词向量文件
    2. 替换掉本例词典中存在词的词向量
    
    在神经网络训练时，再用这个WORD2VEC_SUB将文本转化为词向量
    """
    print("正在加载预训练词向量……")
    
    if os.path.isfile(WORD2VEC_SUB): #如果之前已经生成过替换后的词向量
        print("已存在替换好的词向量，直接读取")
        with open(WORD2VEC_SUB, "rb") as f:
            pha = pickle.load(f)  
    else:
        if os.path.isfile(DICTIONARY_DIC):#如果之前已经生成过所有新闻的字典
            print("已存在由所有新闻生成的字典，直接读取")
            with open(DICTIONARY_DIC, "rb") as f:
                DICTIONARY = pickle.load(f)   
        else:
             # 生成利用所有新闻归纳出的字典
            DICTIONARY = get_news_dic(TRAIN_DATA,DICTIONARY_DIC,WORD_FREQUENCY_LOW)
        
        # 1. 生成[词向量个数, 300维]的随机均匀分布
        pha = np.random.uniform(-1.0, 1.0, [len(DICTIONARY), 300]) 
        # 2. 使用预训练好的词向量替换掉随机生成的分布
        if os.path.isfile(WORD2VEC_DIC):
            with fileinput.input(files=(WORD2VEC_DIC), openhook=fileinput.hook_encoded('UTF-8')) as f:
                count = 0
                for line in f:
                    word_and_vec = line.split(' ')
                    word = word_and_vec[0]
                    vec = word_and_vec[1:301]
                    
                    if word in DICTIONARY:#替换
                        pha[DICTIONARY.index(word)] = vec
                        #进度条
                        count += 1
                        if count % 36000 == 0:
                            # print('处理进度：', count / total_line_num * 100, '%')
                            view_bar('处理进度：', count,364991) #别人训练好的词向量有36万词 
            with open(WORD2VEC_SUB, 'wb') as f:
                pickle.dump(pha, f) # 把所有新闻的词向量保存入文件   
    print("预训练词向量加载完毕。")
    return pha

def get_asp_w2v(ASPECT_WORD2VEC,data_train,WORD2VEC_DIC,WORD2VEC_SUB, TRAIN_DATA,DICTIONARY_DIC,WORD_FREQUENCY_LOW,set_sequence_length):
    """
    将每个类别的新闻文本，进行文本预处理后，用替换好的所有新闻词典，生成词向量3d矩阵
    """
    if os.path.isfile(ASPECT_WORD2VEC):
        print("已存在该类别词向量3d矩阵，直接读取")
        with open(ASPECT_WORD2VEC, "rb") as f:
            x_train_3d_vecs = pickle.load(f)
            max_len = x_train_3d_vecs.shape[2]
    else:
        #文本预处理
        x_train = tp.convert_doc_to_wordlist(data_train["title"],tool = "jieba",cut_all=False,mode ="accuracy")
        #计算最大句子长度
        if set_sequence_length!=None:#直接指定最大句子长度
            max_len = set_sequence_length
        else:
            max_len = max(list(map(len,x_train)))
        #补零
        def padding_sent(sent,max_len=max_len ,padding_token="空"):
            """给不满足句子长度的句子补零"""
            if len(sent) > max_len:
                sent = sent[:max_len]
            else:
                sent.extend([padding_token] * (max_len - len(sent)))
            return sent
        x_train_pad = list(map(padding_sent,x_train))
        #读取词向量
        if os.path.isfile(WORD2VEC_SUB): #如果之前已经生成过替换后的词向量
            print("已存在替换好的词向量，直接读取")
            with open(WORD2VEC_SUB, "rb") as f:
                pha = pickle.load(f)
        else:
            pha = get_word2vec(WORD2VEC_DIC,WORD2VEC_SUB, TRAIN_DATA,DICTIONARY_DIC,WORD_FREQUENCY_LOW)
        #读取词典
        if os.path.isfile(DICTIONARY_DIC):
            print("已存在由所有新闻生成的字典，直接读取")
            with open(DICTIONARY_DIC, "rb") as f:
                DICTIONARY = pickle.load(f)
        else:
            DICTIONARY = get_news_dic(TRAIN_DATA,DICTIONARY_DIC,WORD_FREQUENCY_LOW) 
        #生成该类别新闻词向量3d矩阵
        x_train_3d_vecs = np.array([convert_sent_to_mat(sent,pha,DICTIONARY) for sent in x_train_pad])
        x_train_3d_vecs.shape
        #保存该类别的词向量3d矩阵
        with open(ASPECT_WORD2VEC, 'wb') as f:
            pickle.dump(x_train_3d_vecs, f) # 保存入文件       
    return x_train_3d_vecs,max_len
 
def convert_sent_to_mat(sent,pha,DICTIONARY):
    """给定词典和所有新闻词向量，将句子转化为词向量矩阵"""
    size =  pha.shape[1]
    embeddingUnknown = [0 for i in range(size)]
    vec = []
    for word in sent:
        if word in DICTIONARY:
            vec.append(pha[DICTIONARY.index(word)])
        else:
            vec.append(embeddingUnknown)
    return np.array(vec) #返回句子长度*size的矩阵

def get_asp_2d_w2v(data_train,WORD2VEC_DIC,WORD2VEC_SUB, TRAIN_DATA,DICTIONARY_DIC,WORD_FREQUENCY_LOW):
    """
    生成用svm分类的句子向量，句子向量由句子的每个词向量的值平均而成
    """
    x_train = tp.convert_doc_to_wordlist(data_train["title"],tool = "jieba",cut_all=False,mode ="accuracy")
    #读取词向量
    if os.path.isfile(WORD2VEC_SUB): #如果之前已经生成过替换后的词向量
        print("已存在替换好的词向量，直接读取")
        with open(WORD2VEC_SUB, "rb") as f:
            pha = pickle.load(f)
    else:
        pha = get_word2vec(WORD2VEC_DIC,WORD2VEC_SUB, TRAIN_DATA,DICTIONARY_DIC,WORD_FREQUENCY_LOW)
    #读取词典
    if os.path.isfile(DICTIONARY_DIC):
        print("已存在由所有新闻生成的字典，直接读取")
        with open(DICTIONARY_DIC, "rb") as f:
            DICTIONARY = pickle.load(f)
    else:
        DICTIONARY = get_news_dic(TRAIN_DATA,DICTIONARY_DIC,WORD_FREQUENCY_LOW) 
    #生成2d词向量矩阵（一个句子是一个向量）
    x_train_2d_vecs = np.concatenate([convert_sent_to_vec(sent,pha,DICTIONARY) for sent in x_train])
    x_train_2d_vecs = scale(x_train_2d_vecs)

    return x_train_2d_vecs

def convert_sent_to_vec(sent,pha,DICTIONARY):
    """
    给定所有新闻的字典、词向量，生成句子向量
    每个句子向量是词向量的均值
    """
    size =  pha.shape[1]
    embeddingUnknown = [0 for i in range(size)]
    vec = np.zeros(size).reshape((1, size))
    count = 0
    #print text
    for word in sent:
        #print word
        if word in DICTIONARY:
            vec += pha[DICTIONARY.index(word)].reshape((1, size))
            count += 1.
        else:
            vec += embeddingUnknown
    if count != 0:
        vec /= count
    return vec



def main():
    os.chdir("E:/graduate/Paper/")
    WORD2VEC_DIC = './code/pretrained word2vec/sgns.sogou.word' # Chinese Word Vectors提供的预训练词向量
    TRAIN_DATA = "./code/all_news_title.csv" #所有标题合集
    DICTIONARY_DIC ="./code/all_news_original_dic.pickle" # 存放总结出的字典，以节省时间
    WORD2VEC_SUB = "./code/word2vec_sub.pickle" # 替换后的词向量地址
    WORD_FREQUENCY_LOW = 0
    vecs = get_word2vec(WORD2VEC_DIC,WORD2VEC_SUB, TRAIN_DATA,DICTIONARY_DIC,WORD_FREQUENCY_LOW)
    

if __name__ == '__main__':
    main()
