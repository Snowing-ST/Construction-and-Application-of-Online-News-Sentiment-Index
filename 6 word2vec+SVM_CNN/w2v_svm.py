# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 21:51:03 2019
预训练word2vec+SVM
跑很久，，，等不了等不了，，，弃了

@author: situ
"""

import numpy as np
import os
import pandas as pd
import time
import datetime
from sklearn.model_selection import KFold
os.chdir("E:/graduate/Paper/code/")
#from word2vec_doc2vec import get_word2vec
from read_w2v_model import get_asp_2d_w2v
os.chdir("E:/graduate/Paper/code/")
from VSM import train_lg


def load_data_and_labels(aspect,file_name):

    data_train = pd.read_csv(os.path.join("./raw_data/"+aspect,file_name),sep = ",",encoding="gb18030",engine="python")

    WORD2VEC_DIC = './code/pretrained word2vec/sgns.sogou.word'      # Chinese Word Vectors提供的预训练词向量
    TRAIN_DATA = "./code/all_news_title.csv" #所有标题合集
    DICTIONARY_DIC ="./code/all_news_original_dic.pickle"      # 存放总结出的字典，以节省时间
    WORD2VEC_SUB = "./code/word2vec_sub.pickle" # 替换后的词向量地址
    

    x = get_asp_2d_w2v(data_train,WORD2VEC_DIC,WORD2VEC_SUB, TRAIN_DATA,DICTIONARY_DIC,WORD_FREQUENCY_LOW=1)
    y =  data_train["label"]
    
    return x,y

def main():
    os.chdir("E:/graduate/Paper/")
    aspect="经济"  
    file_name = "train.csv"
    x,y = load_data_and_labels(aspect,file_name)
    cv = KFold(n_splits=10, shuffle=True, random_state=1994)
    print('word2vec+lg begins----------')
    #跑逻辑回归已经要跑很久了，感觉还是别用word2vec+svm/lg了，太慢了。。。还不如用fasttext
    time1=time.time()
    lg = train_lg(x,y,cv)
#    cross_val_score(lg,x_train_w2v,y_train,scoring = "accuracy", cv=cv).mean()
    time2=time.time()
    print('word2vec+lg totally cost:',str(datetime.timedelta(seconds=(int(time2 - time1)))))

    

if __name__=="__main__":
    main()
