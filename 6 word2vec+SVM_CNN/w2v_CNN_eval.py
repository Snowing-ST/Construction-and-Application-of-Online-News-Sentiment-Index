#! /usr/bin/env python
"""将剩下的文本使用CNN模型进行分类"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os



os.chdir("E:/graduate/Paper/code/")
import w2v_CNN_data_helpers as CNN_data_helpers
os.chdir("E:/graduate/Paper/code/")
from VSM import get_index
os.chdir("E:/graduate/Paper")

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("aspect", "物价", "Data source for the positive data.")
tf.flags.DEFINE_string("file_name", "test.csv", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "code/runs/prices/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# y_test只是零
#如果训练集和测试集的sequence_length不相符，则需要指定测试集的sequence_length

x_test, y_test,sequence_length = CNN_data_helpers.load_data_and_labels(FLAGS.aspect, 
                                                                       FLAGS.file_name,
                                                                       istest=True,
                                                                       set_sequence_length=None)

print("\nEvaluating...\n")

# Evaluation
# ==================================================
#tf.train.latest_checkpoint model路径一旦改了就不行了
#tf.train.get_checkpoint_state只认原来的最后一个model，删除原来的最后一个也不行
#checkpoint_file = tf.train.latest_checkpoint(os.path.join(os.path.curdir,FLAGS.checkpoint_dir))
#ckpt = tf.train.get_checkpoint_state(os.path.join(os.path.curdir,FLAGS.checkpoint_dir))
#checkpoint_file = os.path.join(os.path.curdir,FLAGS.checkpoint_dir,ckpt.model_checkpoint_path.split("\\")[-1])
model_file = os.listdir(os.path.join(os.path.curdir,FLAGS.checkpoint_dir))[-1]
checkpoint_file = os.path.join(os.path.curdir,FLAGS.checkpoint_dir,model_file.split(".")[0])
print("checkpoint_file is",checkpoint_file)


graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = CNN_data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

#预测类别为012，要改成-1 0 1 
all_predictions = all_predictions-1


# Save the evaluation to a csv
out_path = os.path.join("./raw_data/"+FLAGS.aspect,"test_CNN.csv")
if os.path.isfile(out_path):
    data_test = pd.read_csv(out_path,encoding="gb18030",engine="python")
    
else:
    data_test = pd.read_csv(os.path.join("./raw_data/"+FLAGS.aspect,FLAGS.file_name),sep = ",",encoding="gb18030",engine="python")
    data_test["label"] = all_predictions
    print("Saving evaluation to {0}".format(out_path))
    data_test.to_csv(out_path,encoding="gb18030",index=False)

#计算index_train index_test index_all及其相关系数
consumer_index_file = os.path.join("./raw_data/"+FLAGS.aspect,FLAGS.aspect+"CNN指数.csv")
if os.path.isfile(consumer_index_file):
    print("已存在该指数，直接读取")
    consumer_index = pd.read_csv(consumer_index_file,encoding="gb18030",engine="python")
else:
    data_train = pd.read_csv(os.path.join("./raw_data/"+FLAGS.aspect,"train.csv"),sep = ",",encoding="gb18030",engine="python")
    data = pd.concat([data_train,data_test],axis=0)
    consumer_index_all = get_index(data,prefix="all")
    consumer_index_train = get_index(data_train,prefix="train")
    consumer_index_test = get_index(data_test,prefix="test")
    consumer_index = pd.concat([consumer_index_all["index_all"],consumer_index_train["index_train"],consumer_index_test["index_test"]],axis=1)
    consumer_index.reset_index() #索引变列
    consumer_index.dropna(inplace=True)
    consumer_index.to_csv(consumer_index_file,encoding="gb18030")
print("训练集指数与测试集指数的相关系数为：%0.4f"%(consumer_index[["index_train","index_test"]].corr().ix[0,1]))
print("训练集指数与全集指数的相关系数为：%0.4f"%(consumer_index[["index_train","index_all"]].corr().ix[0,1]))

consumer_index[["index_train","index_test"]].plot()








