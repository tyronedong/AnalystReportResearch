# -*- coding: utf-8 -*-
import gensim
c
import os
import numpy as np
import gc
import ast
import random

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model
from keras.optimizers import *
from keras.models import Sequential
from keras.layers import Merge


DOC2VEC_FILE = '../model/array.txt'
DOC2VEC_DIR = '../model/'
MAX_VOCAB_NUM = 200000
EMBEDDING_DIM = 100
MAX_LENGTH = 1500       # 每个文本的最长选取长度，较短的文本可以设短些

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

train_texts_09_13_0 = []
train_texts_09_13_1 = []
train_texts_09_13_2 = []
train_labels_09_13_0 = []
train_labels_09_13_1 = []
train_labels_09_13_2 = []

test_texts_14_0 = []
test_texts_14_1 = []
test_texts_14_2 = []
test_labels_14_0 = []
test_labels_14_1 = []
test_labels_14_2 = []

cur_count = 0
test_0_count = 0
test_1_count = 0

out_count = 0
for label_seg_report in collection.find({}):
    # if label_seg_report['label_six_month'] > 1: # 无效样本点
    #     continue
    if label_seg_report['price_rate_six_month'] == -1 \
            or label_seg_report['zz500_rate_six_month'] == -1:    # 无效样本点
        continue

    relative_rate = label_seg_report['price_rate_six_month'] - label_seg_report['zz500_rate_six_month']
    # relative_rate = label_seg_report['price_rate_six_month']
    relative_label = 0 if relative_rate > 0.15 else 1 if (relative_rate > -0.05 and relative_rate < 0.05) else 2 if relative_rate < -0.15 else 100
    # test_relative_label = 0 if relative_rate >= 0.1 else 1 if relative_rate > -0.1 else 2
    # relative_label = 0 if relative_rate > 0.15 else 1 if relative_rate < -0.15 else 100
    # test_relative_label = 0 if relative_rate > 0.15 else 1 if relative_rate < -0.15 else 100
    # relative_label = 0 if relative_rate >= 0.05 else 1 if relative_rate > -0.05 else 2
    test_relative_label = relative_label
    # relative_label = 1 if relative_rate >= 0.05 else 0 if relative_rate > -0.05 else -1
    # relative_label = 1 if relative_rate >= 0.1 else 0 if relative_rate > -0.1 else -1
    # relative_label = 1 if relative_rate > 0 else 0

    # print label_seg_report['price_rate_six_month'], label_seg_report['hs300_rate_six_month']
    # print  relative_rate, relative_label

    new_list = []
    w_count = 0
    for word in label_seg_report['seg_conent']:
        # print type(word)
        # print type(word), word
        w_count += 1
        idx = word_index_dict.get(word)
        # print  idx
        if idx == None:
            # print word, None
            continue
        new_list.append(idx)
    # print len(new_list), label_seg_report['_id']
    print cur_count, w_count, label_seg_report['_id']

    cur_count += 1
    # if label_seg_report['Date'].year < 2015:
    #     train_texts.append(new_list)
    #     train_labels.append(relative_label)
    # else:
    #     test_texts.append(new_list)
    #     test_labels.append(relative_label)

    if label_seg_report['Date'].year < 2012:
        if relative_label==0:
            train_texts_09_13_0.append(new_list)
            train_labels_09_13_0.append(relative_label)
	    if w_count > 3000:
	        out_count += 1
        elif relative_label == 1:
            train_texts_09_13_1.append(new_list)
            train_labels_09_13_1.append(relative_label)
	    if w_count > 3000:
	        out_count += 1
        elif relative_label == 2:
            train_texts_09_13_2.append(new_list)
            train_labels_09_13_2.append(relative_label)
        # train_texts.append(new_list)
        # train_labels.append(relative_label)
    elif label_seg_report['Date'].year == 2012:
        if test_relative_label == 0:
            test_texts_14_0.append(new_list)
            test_labels_14_0.append(relative_label)
	    if w_count > 3000:
	        out_count += 1
        elif test_relative_label == 1:
            test_texts_14_1.append(new_list)
            test_labels_14_1.append(relative_label)
	    if w_count > 3000:
        	out_count += 1
        elif test_relative_label == 2:
            test_texts_14_2.append(new_list)
            test_labels_14_2.append(relative_label)
        # test_texts.append(new_list)
        # test_labels.append(test_relative_label)

    # if label_seg_report['Date'].year < 2012:
    #     # label = relative_label
    #     if relative_label==0:
    #         train_texts_091011_1.append(new_list)
    #         train_labels_091011_1.append(relative_label)
    #     elif relative_label == 1:
    #         train_texts_091011_2.append(new_list)
    #         train_labels_091011_2.append(relative_label)
    #     elif relative_label == 2:
    #         train_texts_091011_3.append(new_list)
    #         train_labels_091011_3.append(relative_label)
    #     # train_texts.append(new_list)
    #     # train_labels.append(label_seg_report['label_six_month'])
    # elif label_seg_report['Date'].year == 2012:
    #     test_texts.append(new_list)
    #     test_labels.append(relative_label)

# # print train_labels
#
# print '091011 1:', len(train_texts_091011_1)
# print '091011 0:', len(train_texts_091011_2)
# print '091011 -1:', len(train_texts_091011_3)

print '09101111213 1:', len(train_texts_09_13_0)
print '0910111213 0:', len(train_texts_09_13_1)
print '0910111213 -1:', len(train_texts_09_13_2)

print '12 1:', len(test_texts_14_0)
print '12 0', len(test_texts_14_1)
print '12 -1', len(test_texts_14_2)

L1 = random.sample([i for i in range(len(train_texts_09_13_0))], 10000)
L2 = random.sample([i for i in range(len(train_texts_09_13_1))], 10000)
L3 = random.sample([i for i in range(len(train_texts_09_13_2))], 10000)

for i in L1:
    train_texts.append(train_texts_09_13_0[i])
    train_labels.append(train_labels_09_13_0[i])
for i in L2:
    train_texts.append(train_texts_09_13_1[i])
    train_labels.append(train_labels_09_13_1[i])
for i in L3:
    train_texts.append(train_texts_09_13_2[i])
    train_labels.append(train_labels_09_13_2[i])

L = random.sample(range(len(train_texts)), len(train_texts))
temp_train_texts = []
temp_train_labels = []
for i in L:
    temp_train_texts.append(train_texts[i])
    temp_train_labels.append(train_labels[i])
# random.shuffle(train_texts)
train_texts = temp_train_texts
train_labels = temp_train_labels

L0 = random.sample([i for i in range(len(test_texts_14_0))], 5000)
L1 = random.sample([i for i in range(len(test_texts_14_1))], 5000)
L2 = random.sample([i for i in range(len(test_texts_14_2))], 5000)
for i in L0:
    test_texts.append(test_texts_14_0[i])
    test_labels.append(test_labels_14_0[i])
for i in L1:
    test_texts.append(test_texts_14_1[i])
    test_labels.append(test_labels_14_1[i])
for i in L2:
    test_texts.append(test_texts_14_2[i])
    test_labels.append(test_labels_14_2[i])
# L = random.sample(range(len(test_texts)), len(test_texts))
# temp_test_texts = []
# temp_test_labels = []
# for i in L:
#     temp_test_texts.append(test_texts[i])
#     temp_test_labels.append(test_labels[i])
# test_texts = temp_test_texts
# test_labels = temp_test_labels

# # L1 = random.sample([i for i in range(len(train_texts_091011_1))], 20000)
# # L2 = random.sample([i for i in range(len(train_texts_091011_2))], 20000)
# # L3 = random.sample([i for i in range(len(train_texts_091011_3))], 0)
#
# for i in L1:
#     train_texts.append(train_texts_091011_1[i])
#     train_labels.append(train_labels_091011_1[i])
# for i in L2:
#     train_texts.append(train_texts_091011_2[i])
#     train_labels.append(train_labels_091011_2[i])
# for i in L3:
#     train_texts.append(train_texts_091011_3[i])
#     train_labels.append(train_labels_091011_3[i])
#
# L = random.sample(range(len(train_texts)), len(train_texts))
# temp_train_texts = []
# temp_train_labels = []
# for i in L:
#     temp_train_texts.append(train_texts[i])
#     temp_train_labels.append(train_labels[i])
# # random.shuffle(train_texts)
# train_texts = temp_train_texts
# train_labels = temp_train_labels
#
# print len(train_texts)
# print len(test_texts)
#
# # print train_labels

print 'Train:', len(train_texts)
print 'Test:', len(test_texts)
print 'out_3000', out_count

# print train_labels

x_train = pad_sequences(train_texts, maxlen=MAX_LENGTH)
y_train = to_categorical(np.asarray(train_labels), 3)
# y_train = to_categorical(np.asarray(train_labels), 2)
# print train_labels
# print y_train

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

# model = Sequential()

model_left = Sequential()
model_left.add(embedding_layer)
model_left.add(Conv1D(128, 5, activation='tanh'))
model_left.add(MaxPooling1D(5))
model_left.add(Conv1D(128, 5, activation='tanh'))
model_left.add(MaxPooling1D(5))
model_left.add(Conv1D(128, 5, activation='tanh'))
model_left.add(MaxPooling1D(35))
model_left.add(Flatten())

# right model <span style="font-family: Arial, Helvetica, sans-serif;">第二块神经网络，卷积窗口是4*50</span>

model_right = Sequential()
model_right.add(embedding_layer)
model_right.add(Conv1D(128, 4, activation='tanh'))
model_right.add(MaxPooling1D(4))
model_right.add(Conv1D(128, 4, activation='tanh'))
model_right.add(MaxPooling1D(4))
model_right.add(Conv1D(128, 4, activation='tanh'))
model_right.add(MaxPooling1D(28))
model_right.add(Flatten())

# third model <span style="font-family: Arial, Helvetica, sans-serif;">第三块神经网络，卷积窗口是6*50</span>
model_3 = Sequential()
model_3.add(embedding_layer)
model_3.add(Conv1D(128, 6, activation='tanh'))
model_3.add(MaxPooling1D(3))
model_3.add(Conv1D(128, 6, activation='tanh'))
model_3.add(MaxPooling1D(3))
model_3.add(Conv1D(128, 6, activation='tanh'))
model_3.add(MaxPooling1D(30))
model_3.add(Flatten())

merged = Merge([model_left, model_right, model_3], mode='concat')  # 将三种不同卷积窗口的卷积层组合 连接在一起，当然也可以只是用三个model

model = Sequential()
model.add(merged) # add merge
# model.add(embedding_layer)
# model.add(Conv1D(128, 5, activation='tanh'))
# model.add(MaxPooling1D(5))
# model.add(Conv1D(128, 5, activation='tanh'))
# model.add(MaxPooling1D(5))
# model.add(Conv1D(128, 5, activation='tanh'))
# model.add(MaxPooling1D(35)) #这一层只有35个，pool后只有1个
# model.add(Flatten())

model.add(Dense(128, activation='tanh'))  # 全连接层
# model.add(Dense(len(word_index_dict), activation='softmax'))  # softmax，输出文本属于20种类别中每个类别的概率
# model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))  # softmax，输出文本属于20种类别中每个类别的概率
# model.add(Dense(2, activation='softmax'))

# 优化器我这里用了adadelta，也可以使用其他方法
model.compile(loss='categorical_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])

# =下面开始训练，nb_epoch是迭代次数，可以高一些，训练效果会更好，但是训练会变慢
model.fit(x_train, y_train, nb_epoch=6, shuffle=True)

score = model.evaluate(x_train, y_train, verbose=0)  # 评估模型在训练集中的效果，准确率约99%
print('train score:', score[0])
print('train accuracy:', score[1])
score = model.evaluate(x_test, y_test, verbose=0)  # 评估模型在测试集中的效果，准确率约为97%，迭代次数多了，会进一步提升
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save('./CNN/cnn.model')
