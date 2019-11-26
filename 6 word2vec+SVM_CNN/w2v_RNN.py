# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 22:51:27 2019

#网络结构：（有padding）
词向量+一层、两层隐藏层+全连接1+全连接2（变成3列）+softmax
https://github.com/gaussic/text-classification-cnn-rnn

其他技术博客中的结构（有padding）
词向量+一层隐藏层+meanpooling+softmax
http://deeplearning.net/tutorial/lstm.html

@author: situ
"""

import tensorflow as tf
import numpy as np


class TextRNN(object):
    
    def __init__(self,embedding_dim,sequence_length,num_classes,vocab_size,
                 num_layers,hidden_dim,rnn_type,l2_reg_lambda):

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")


        def dropout(rnn_type,hidden_dim): # 为每一个rnn核后面加一个dropout层
            if (rnn_type == 'lstm'):
                cell =  tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
            if (rnn_type == 'gru'):
                cell = tf.contrib.rnn.GRUCell(hidden_dim)
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
    
        # 词向量映射
        with tf.device('/cpu:0'),tf.name_scope("embedding_layer"):
            embedding = tf.get_variable('embedding', [vocab_size, embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
    
        with tf.name_scope("rnn"):
            # 多层rnn网络
            cells = [dropout(rnn_type,hidden_dim) for _ in range(num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
    
            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果
    
        with tf.name_scope("output"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.dropout_keep_prob)
            fc = tf.nn.relu(fc)
    
            # 分类器
            self.logits = tf.layers.dense(fc, num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1,name="predictions")  # 预测类别
    
        with tf.name_scope("loss"):
            # 损失函数，交叉熵
            tv = tf.trainable_variables()#得到所有可以训练的参数，即所有trainable=True 的tf.Variable/tf.get_variable
            l2_loss = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ]) #0.001是lambda超参数
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)+l2_loss*l2_reg_lambda
    
        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))