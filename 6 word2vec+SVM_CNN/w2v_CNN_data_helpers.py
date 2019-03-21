import numpy as np
import os
import pandas as pd
os.chdir("E:/graduate/Paper/code/")
#from word2vec_doc2vec import get_word2vec
from read_w2v_model import get_asp_w2v
os.chdir("E:/graduate/Paper/")

def load_data_and_labels(aspect,file_name,set_sequence_length=None,istest=False):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
#    aspect="经济"  
#    file_name = "train.csv"
    data_train = pd.read_csv(os.path.join("./raw_data/"+aspect,file_name),sep = ",",encoding="gb18030",engine="python")
#    x_text = [w.split() for w in data_train["word_seg"]]
#    x_text,sequence_length = get_word2vec(x_text,size=embedded_size,min_count =min_count ,window = 5,method="padding",seq_dim=2)
    if istest:
        ASPECT_WORD2VEC = "./code/%s%sword2vec.pickle"%(aspect,"test")
        y = np.zeros((len(data_train)),int)
    else:
        ASPECT_WORD2VEC = "./code/%sword2vec.pickle"%aspect
        y =  np.array(pd.get_dummies(data_train["label"]))

    WORD2VEC_DIC = './code/pretrained word2vec/sgns.sogou.word'      # Chinese Word Vectors提供的预训练词向量
    TRAIN_DATA = "./code/all_news_title.csv" #所有标题合集
    DICTIONARY_DIC ="./code/all_news_original_dic.pickle"      # 存放总结出的字典，以节省时间
    WORD2VEC_SUB = "./code/word2vec_sub.pickle" # 替换后的词向量地址
    WORD_FREQUENCY_LOW = 0
    

    x_text,sequence_length = get_asp_w2v(ASPECT_WORD2VEC,data_train,WORD2VEC_DIC,WORD2VEC_SUB, TRAIN_DATA,DICTIONARY_DIC,WORD_FREQUENCY_LOW,set_sequence_length)
    
    return x_text, y,sequence_length


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


