# -*- coding: utf-8 -*-
import gensim
from pymongo import *
import os
import numpy as np
import gc
import ast
import random

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.optimizers import *
from keras.models import Sequential
from keras.layers import Merge
from keras.layers.recurrent import LSTM, GRU


DOC2VEC_FILE = '../model/array.txt'
DOC2VEC_DIR = '../model/'
MAX_VOCAB_NUM = 200000
EMBEDDING_DIM = 100
MAX_LENGTH = 1000       # 每个文本的最长选取长度，较短的文本可以设短些

word_embeddings_dict = {}
word_index_dict = {} # 应该是实际出现过的vocab的index



# 获取单词到词向量的映射，单词到编号的映射
word2vecModel = gensim.models.Word2Vec.load('./Word2Vec/word2vec.model')
for idx, word in enumerate(word2vecModel.wv.vocab):
    word_embeddings_dict[word] = np.asarray(word2vecModel[word], dtype='float32')
    index = len(word_index_dict)
    word_index_dict[word] = index



# doc2vec_file = open(DOC2VEC_FILE)
# lines = doc2vec_file.readlines()
# for line in lines:
#     split_res = line.split(' ')
#     word = unicode(split_res[0], "utf-8")
#     word_embeddings_dict[word] = np.asarray([float(x) for x in split_res[1:101]], dtype='float32')
#     index = len(word_index_dict)
#     word_index_dict[word] = index
#     print index
# doc2vec_file.close()
# 释放doc2vec model占用的内存
del word2vecModel
gc.collect()

client = MongoClient('localhost', 27017)
db = client.AnalystReport
collection = db.RatedSegReport

# 把每个文本序列映射成以数字序列表示的词向量形式，读入训练数据和标记数据
train_texts = []              # 存储训练样本的list
train_labels = []             # 存储训练样本，类别编号的文本，比如文章a属于第1类文本
test_texts = []               # 存储测试样本的list
test_labels = []              # 存储测试样本，类别编号的文本，比如文章a属于第1类文本
# for label_seg_report in collection.find({}).limit(1000):
train_texts_091011_1 = []
train_texts_091011_2 = []
train_texts_091011_3 = []
train_labels_091011_1 = []
train_labels_091011_2 = []
train_labels_091011_3 = []

cur_count = 0
for label_seg_report in collection.find({}).limit(10000):
    # if label_seg_report['label_six_month'] > 1: # 无效样本点
    #     continue
    if label_seg_report['price_rate_six_month'] == -1 \
            or label_seg_report['zz500_rate_six_month'] == -1:    # 无效样本点
        continue

    relative_rate = label_seg_report['price_rate_six_month'] - label_seg_report['zz500_rate_six_month']
    relative_label = 0 if relative_rate >= 0.05 else 1 if relative_rate > -0.05 else 2
    # relative_label = 1 if relative_rate >= 0.05 else 0 if relative_rate > -0.05 else -1
    # relative_label = 1 if relative_rate >= 0.1 else 0 if relative_rate > -0.1 else -1
    # relative_label = 1 if relative_rate > 0 else 0

    # print label_seg_report['price_rate_six_month'], label_seg_report['hs300_rate_six_month']
    # print  relative_rate, relative_label

    new_list = []
    for word in label_seg_report['seg_conent']:
        # print type(word)
        # print type(word), word
        idx = word_index_dict.get(word)
        # print  idx
        if idx == None:
            # print word, None
            continue
        new_list.append(idx)
    # print len(new_list), label_seg_report['_id']
    print cur_count, label_seg_report['_id']
    cur_count += 1
    # if label_seg_report['Date'].year < 2015:
    #     train_texts.append(new_list)
    #     train_labels.append(label_seg_report['label_six_month'])
    # else:
    #     test_texts.append(new_list)
    #     test_labels.append(label_seg_report['label_six_month'])

    # if label_seg_report['Date'].year < 2014:
    #     train_texts.append(new_list)
    #     train_labels.append(relative_label)
    # elif label_seg_report['Date'].year == 2014:
    #     test_texts.append(new_list)
    #     test_labels.append(relative_label)

    if label_seg_report['Date'].year < 2012:
        # label = relative_label
        if relative_label==0:
            train_texts_091011_1.append(new_list)
            train_labels_091011_1.append(relative_label)
        elif relative_label == 1:
            train_texts_091011_2.append(new_list)
            train_labels_091011_2.append(relative_label)
        elif relative_label == 2:
            train_texts_091011_3.append(new_list)
            train_labels_091011_3.append(relative_label)
        # train_texts.append(new_list)
        # train_labels.append(label_seg_report['label_six_month'])
    elif label_seg_report['Date'].year == 2012:
        test_texts.append(new_list)
        test_labels.append(relative_label)

# print train_labels

print '091011 1:', len(train_texts_091011_1)
print '091011 0:', len(train_texts_091011_2)
print '091011 -1:', len(train_texts_091011_3)

L1 = random.sample([i for i in range(len(train_texts_091011_1))], 500)
L2 = random.sample([i for i in range(len(train_texts_091011_2))], 500)
L3 = random.sample([i for i in range(len(train_texts_091011_3))], 500)
# L1 = random.sample([i for i in range(len(train_texts_091011_1))], 20000)
# L2 = random.sample([i for i in range(len(train_texts_091011_2))], 20000)
# L3 = random.sample([i for i in range(len(train_texts_091011_3))], 0)

for i in L1:
    train_texts.append(train_texts_091011_1[i])
    train_labels.append(train_labels_091011_1[i])
for i in L2:
    train_texts.append(train_texts_091011_2[i])
    train_labels.append(train_labels_091011_2[i])
for i in L3:
    train_texts.append(train_texts_091011_3[i])
    train_labels.append(train_labels_091011_3[i])

L = random.sample(range(len(train_texts)), len(train_texts))
temp_train_texts = []
temp_train_labels = []
for i in L:
    temp_train_texts.append(train_texts[i])
    temp_train_labels.append(train_labels[i])
# random.shuffle(train_texts)
train_texts = temp_train_texts
train_labels = temp_train_labels

print len(train_texts)
print len(test_texts)

# print train_labels

x_train = pad_sequences(train_texts, maxlen=MAX_LENGTH)
y_train = to_categorical(np.asarray(train_labels), 3)
# y_train = to_categorical(np.asarray(train_labels), 2)
print train_labels
print y_train

x_test = pad_sequences(test_texts, maxlen=MAX_LENGTH)
y_test = to_categorical(np.asarray(test_labels), 3)
# y_test = to_categorical(np.asarray(test_labels), 2)
print  y_test

# prepare embedding matrix 这部分主要是创建一个词向量矩阵，使每个词都有其对应的词向量相对应
# print len(word_index_dict)
nb_words = min(MAX_VOCAB_NUM, len(word_index_dict))
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index_dict.items():
    if i > MAX_VOCAB_NUM:
        continue
    embedding_vector = word_embeddings_dict.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector  # word_index to word_embedding_vector ,<20000(nb_words)

# 神经网路的第一层，词向量层，本文使用了预训练glove词向量，可以把trainable那里设为False
embedding_layer = Embedding(nb_words + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_LENGTH,
                            weights=[embedding_matrix],
                            trainable=False)

model = Sequential()

model.add(embedding_layer)

model.add(GRU(output_dim=128, activation='tanh', inner_activation='hard_sigmoid'))  # try using a GRU instead, for fun

model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax'))

# try using different optimizers and different optimizer configs

model.compile(loss='categorical_crossentropy', optimizer='adam', class_mode="categorical")

json_string = model.to_json()

print(json_string)

print("Train...")

model.fit(x_train, y_train, nb_epoch=4, show_accuracy=True)

score = model.evaluate(x_train, y_train, verbose=0)  # 评估模型在训练集中的效果，准确率约99%
# print('train score:', score[0])
# print('train accuracy:', score[1])
print score
score = model.evaluate(x_test, y_test, verbose=0)  # 评估模型在测试集中的效果，准确率约为97%，迭代次数多了，会进一步提升
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
print score
