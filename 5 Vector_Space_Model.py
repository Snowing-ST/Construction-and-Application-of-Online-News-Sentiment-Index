# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 21:16:23 2018
(如果分类效果不好，考虑将“经济”的一些0标签标题按关键词删除)
分词、文本预处理
构建向量空间模型
文本分类总模型：
参数：去不去低频词高频词，一元文法还是二元文法还是两个混合，
     文本向量化方式（tf-idf）（如果是one-hot则要降维），特征词的个数，是否要矩阵分解，
     选择分类方法（朴素贝叶斯、逻辑回归、SVM、随机森林）
返回值：分类效果各种评价指标值
main函数：得出多种情况下的分类效果



投资：需删除含“投资意向”“意向”样本，删除“投资意愿”下方不含“意愿”或不含“投资”的样本
删除包含“不动产”、“房地产”，“楼市”的样本

生活：1的新闻偏多，测试集多偏向1，指数高估
感觉关键词并不全面


@author: situ
"""

import pandas as pd
import numpy as np
import os
import re
import jieba
from jieba.analyse import extract_tags
from collections import Counter
import operator
import time
import datetime

from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_val_predict,KFold
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn import svm
from sklearn.model_selection import GridSearchCV


#import xgboost as xgb

os.chdir("E:/graduate/Paper/code/")
import text_preprocess as tp




                                       
def vectorize(data_train,data_test,word_name = "word_seg",tag_name="label",vectype = "tf-idf",ngram_range=(0,1),max_features=None,min_df=2):
    """
    文本表示：tf-idf稀疏矩阵,好像sklearn fit_transform可以选择几gram
    """
#    data.isnull().any()#哪些列存在缺失值
    words = data_train[word_name].tolist()+data_test[word_name].tolist()
    if vectype == "tf-idf":
        transformer=TfidfVectorizer(ngram_range=ngram_range,max_features=max_features,min_df=min_df)
        data_tfidf=transformer.fit_transform(words)
        transformer2 = TfidfVectorizer(vocabulary = transformer.vocabulary_)
        data_train_tfidf=transformer2.fit_transform(data_train[word_name])
        data_test_tfidf=transformer2.fit_transform(data_test[word_name])          
        return data_train_tfidf,data_train[tag_name].tolist(),data_test_tfidf
    if vectype == "one-hot":
        transformer=CountVectorizer(ngram_range=ngram_range,max_features=max_features,min_df=min_df)
         
        data_onehot=transformer.fit_transform(words)
        transformer2 = CountVectorizer(vocabulary = transformer.vocabulary_)
        data_train_onehot=transformer2.fit_transform(data_train[word_name])
        data_test_onehot=transformer2.fit_transform(data_test[word_name])           
        return data_train_onehot,data_train[tag_name].tolist(),data_test_onehot
        
    
#NMF降维
#        model = NMF(n_components=max_features, random_state=0,tol=0.01)
#        U = model.fit_transform(data_count.T)
#        VT = model.components_
#        V = VT.T

def n_feature_selection(n_feature_list,data,word_name = "word_seg",tag_name="label",vectype = "tf-idf",ngram_range=(0,2)):
    """
    特征选择（未使用）
    n_feature_list：选择保留几个特征的列表，如[100,500,1000]
    """
    for n_feature in n_feature_list:
        data_tfidf, tags = vectorize(data,word_name = word_name,tag_name=tag_name,vectype = vectype,ngram_range=ngram_range,max_features=n_feature)
        cv = KFold(n_splits=10, shuffle=True, random_state=1994)
        print("when n_feature=%d,svm cv accuracy is "%n_feature,train_SVM(data_tfidf, tags,cv,C=15,gamma=0.5,lr=1))
        print("when n_feature=%d,NB cv accuracy is "%n_feature,train_NB(data_tfidf, tags,cv,0.18))
        print("when n_feature=%d,lg cv accuracy is "%n_feature,train_lg(data_tfidf, tags,cv,tol=0.1,C=28))
#n_feature_list =  range(500,4001,500)       
#n_feature_selection(n_feature_list,data)    
#n_feature_list = range(2000,4001,200)
#n_feature_selection(n_feature_list,data)  #3500

def train_NB(data_tfidf,tags,cv):
    grid_values = {'alpha':np.arange(0.1,1.1,0.1)} # Decide which settings you want for the grid search. 

    grid = GridSearchCV(MultinomialNB(), 
                        grid_values, scoring = "accuracy", cv = cv) 
    grid.fit(data_tfidf,tags) 
    grid.grid_scores_
    print("【NB】The best parameters are %s with a score of %0.4f"
          % (grid.best_params_, grid.best_score_))

#    clf = MultinomialNB(alpha=list(grid.best_params_.values())[0])
#    scores = cross_val_score(clf,data_tfidf,tags,scoring = "accuracy", cv=cv)
#    scores.mean()
    return grid.best_estimator_

def train_lg(data_tfidf,tags,cv):
    grid_values = {'tol':[0.001,0.1,1],'C':range(1,10,2)} # Decide which settings you want for the grid search. 

    grid = GridSearchCV(LogisticRegression(penalty="l2", dual=True), 
                        grid_values, scoring = "accuracy", cv = cv,n_jobs=7) 
    grid.fit(data_tfidf,tags) 
    grid.grid_scores_
    print("【lg】The best parameters are %s with a score of %0.4f"
          % (grid.best_params_, grid.best_score_))
    
#    clf = LogisticRegression(penalty="l2", dual=True, tol=tol, C=C)  
#    scores = cross_val_score(clf,data_tfidf,tags,scoring = "accuracy", cv=cv)
#    return scores.mean()
    return grid.best_estimator_

def train_ada(data_tfidf,tags,cv):
    """运行速度慢，不使用"""
    grid_values = {'n_estimators':[50,100,200],'learning_rate':[0.1,0.4,0.7,1]} # Decide which settings you want for the grid search. 

    grid = GridSearchCV(AdaBoostClassifier( algorithm="SAMME.R", random_state=10), 
                        grid_values, scoring = "accuracy", cv = cv) 
    grid.fit(data_tfidf,tags) 
    grid.grid_scores_
    print("【ada】The best parameters are %s with a score of %0.4f"
          % (grid.best_params_, grid.best_score_))
    
#    clf = AdaBoostClassifier(n_estimators=n_trees, learning_rate=lr, algorithm="SAMME.R", random_state=10)
#    scores = cross_val_score(clf,data_tfidf,tags,scoring = "accuracy", cv=cv)
#    return scores.mean()
    return grid.best_estimator_


def train_rf(data_tfidf,tags,cv):
    """运行速度慢，不使用"""
    grid_values = {"min_samples_leaf":[1,3],"min_samples_split":[2,4,6,8]} # Decide which settings you want for the grid search. 

    grid = GridSearchCV(RandomForestClassifier(n_estimators=100,oob_score=True, max_features = "sqrt",random_state=10,max_depth=None) , 
                        grid_values, scoring = "accuracy", cv = cv,n_jobs=7) 
    grid.fit(data_tfidf,tags) 
    grid.grid_scores_
    print("【rf】The best parameters are %s with a score of %0.4f"
          % (grid.best_params_, grid.best_score_))
#    clf = RandomForestClassifier(n_estimators= n_trees, oob_score=True, max_features = max_features,max_depth=max_depth,min_samples_split=min_samples_split,random_state=10)  
#    scores = cross_val_score(clf,data_tfidf,tags,scoring = "accuracy", cv=cv)
#    scores.mean()
    return grid.best_estimator_

def train_SVM(data_tfidf,tags,cv):#5,1,1 #4,0.9,1 0.8238
    "调参影响大。学习率越小，所需迭代次数越多"
    grid_values = {'C':[1,4,7],'gamma':[0.1,0.5,0.9]} # Decide which settings you want for the grid search. 

    grid = GridSearchCV(svm.SVC(kernel='rbf',tol=1, degree=3, coef0=0.0, shrinking=True, probability=False),
                        grid_values, scoring = "accuracy", cv = cv) 
    grid.fit(data_tfidf,tags) 
    
    grid.grid_scores_
    print("【SVM】The best parameters are %s with a score of %0.4f"
          % (grid.best_params_, grid.best_score_))

#    clf = svm.SVC(C=C, kernel='rbf', degree=3, gamma=gamma, coef0=0.0, shrinking=True, probability=False,tol=lr)
#    scores = cross_val_score(clf,data_tfidf,tags,scoring = "accuracy", cv=cv)
#    return scores.mean()
    return grid.best_estimator_


    
#模型集成预测（投票制）
def ensemble(clf_list,data_test_tfidf):
    """生成预测值"""
    i=0
    pre_mat = np.zeros((data_test_tfidf.shape[0],len(clf_list)+1),dtype="int64")
    for clf in clf_list:
        pre_mat[:,i] = clf.predict(data_test_tfidf)
        i=i+1
#       np.bincount()：统计次数,不允许序列中出现负数,因此加1
    pre_mat[:,len(clf_list)] =  list(map(lambda a:np.argmax(np.bincount(a)),pre_mat+1))
    pre_mat[:,len(clf_list)] = pre_mat[:,len(clf_list)]-1
    return pre_mat
         
def evaluate(actual, pred):
    m_accuracy = metrics.accuracy_score(actual, pred)
    m_precision = metrics.precision_score(actual, pred, average='macro')
    m_recall = metrics.recall_score(actual, pred, average='macro')
    print('accuracy:{0:.3f}'.format(m_accuracy))
    # print('precision:{0:.3f}'.format(m_precision)) #二分类使用
    # print('recall:{0:0.3f}'.format(m_recall)) #二分类使用
    print("confusion matrix:\n",metrics.confusion_matrix(actual, pred))

    
def main():  
    os.chdir("E:/graduate/Paper/")
#    aspect = input('请输入六大分指数之一： \n')
    aspect = "物价"
    path = "./raw_data/"+aspect  
    file_name = "train.csv"
    data_train = pd.read_csv(os.path.join(path,file_name),sep = ",",encoding="gb18030",engine="python")
    data_test = pd.read_csv(os.path.join(path,"test.csv"),sep = ",",encoding="gb18030",engine="python")
    data_train.dropna(inplace=True)
    data_train.head()
#    data_train.tail()
    data_train.shape
    data_train["label"].value_counts()
    
    full_data = [data_train,data_test]
    time1 = time.time()
    for dataset in full_data:
        clean_text=tp.convert_doc_to_wordlist(dataset["title"],tool="pku",cut_all=False,mode ="accuracy")
        dataset["word_seg"] = [" ".join(line) for line in clean_text]
#    data_train.to_csv("data_with_wordseg.csv",index = False,encoding = "gb18030")
    time2 = time.time()
    print("分词用时"+str(datetime.timedelta(seconds=(int(time2 - time1)))))
    
    data_train.shape
    data_train["label"].astype(int)
    tp.view_keywords(data_train,word_name = "word_seg",tag_name="label")
#    data1 = rm_low_high_freq(data_train["word_seg"],data_train,low_freq=1,high_topK=15) #除去高频词
    data_train.shape
#    data1.shape
#    view_keywords(data1,word_name = "rm_low_high_freq",tag_name="label")

#是否去除高低频词,是否做特征选择       
#    data_tfidf, tags = vectorize(data1,word_name = "rm_low_high_freq",tag_name="label",vectype = "tf-idf",ngram_range=(0,1))
#    data_tfidf, tags = vectorize(data_train,word_name = "word_seg",tag_name="label",vectype = "tf-idf",ngram_range=(0,1),max_features=None)
    data_train_tfidf, tags ,data_test_tfidf = vectorize(data_train,data_test,word_name = "word_seg",tag_name="label",
                                                        vectype = "tf-idf",ngram_range=(0,1),max_features=None,min_df=1)
    data_train_tfidf.shape
    #单个模型比较
    cv = KFold(n_splits=10, shuffle=True, random_state=1994)
    NB = train_NB(data_train_tfidf, tags,cv)
    lg = train_lg(data_train_tfidf, tags,cv)
    SVM = train_SVM(data_train_tfidf, tags,cv)
    
#    rf = train_rf(data_train_tfidf, tags,cv)
#    ada = train_ada(data_train_tfidf, tags,cv)

# 查看哪些关键词判断准确率高
    data_train["predicted"] = cross_val_predict(lg,data_train_tfidf,tags, cv=cv)
    data_train["TorF"] = data_train["predicted"]==data_train["label"]
    print(data_train.groupby(["keyword"]).agg({"predicted":len,"TorF":np.mean,
                             "label":lambda x:np.mean(x==-1)}))
    metrics.confusion_matrix(data_train["label"], data_train["predicted"] )
##prediction 
    clf_list = [rf,lg,SVM]
    data_test["label"] = ensemble(clf_list,data_test_tfidf)[:,-1]
    data_test.to_csv(os.path.join(path,"test_VSM.csv"),index = False,encoding = "gb18030")
##合并表格，计算季度消费者情感指数
    data = pd.concat([data_train,data_test],axis=0)
    # data.to_csv(os.path.join(path,"all_df_pridicted.csv"),index = False,encoding = "gb18030") 
#    data.shape
    var_list = list(data.columns)
    var_list.remove("keyword");var_list.remove("label")
    data.drop_duplicates(subset = var_list,inplace=True)#只有完全一样的才删除

    consumer_index_all = get_index(data,prefix="all")
    consumer_index_train = get_index(data_train,prefix="train")
    consumer_index_test = get_index(data_test,prefix="test")
    consumer_index = pd.concat([consumer_index_all["index_all"],consumer_index_train["index_train"],consumer_index_test["index_test"]],axis=1)
    consumer_index.reset_index() #索引变列
    consumer_index.dropna(inplace=True)
    consumer_index.to_csv(os.path.join(path,aspect+"VSM指数.csv"))
    print("训练集指数与测试集指数的相关系数为：%f"%consumer_index[["index_train","index_test"]].corr().ix[0,1])

if __name__ == '__main__':
    main()    
