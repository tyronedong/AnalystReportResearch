# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import numpy as np

np.random.seed(1337)

# from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.optimizers import *
from keras.models import Sequential
from keras.layers import Merge
import sys
from gensim.models import Doc2Vec
from pymongo import *

BASE_DIR = '..'  # 这里是指当前目录
DOC2VEC_DIR = BASE_DIR + '/model/'
GLOVE_DIR = BASE_DIR + '/glove.6B/'  # 根据实际目录名更改
TEXT_DATA_DIR = BASE_DIR + '/20_newsgroup/'  # 根据实际目录名更改
MAX_SEQUENCE_LENGTH = 1000  # 每个文本的最长选取长度，较短的文本可以设短些
MAX_NB_WORDS = 200000  # 整体词库字典中，词的多少，可以略微调大或调小

EMBEDDING_DIM = 100  # 词向量的维度，可以根据实际情况使用，如果不了解暂时不要改

VALIDATION_SPLIT = 0.4  # 这里用作是测试集的比例，单词本身的意思是验证集

# first, build index mapping words in the embeddings set
# to their embedding vector  这段话是指建立一个词到词向量之间的索引，比如 peking 对应的词向量可能是（0.1,0,32,...0.35,0.5)等等。

print('Indexing word vectors.')

word_embeddings_dict = {}
word_index_dict = {}

label_id = len(labels_index)
labels_index[name] = label_id
# f = open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'))  # 读入50维的词向量文件，可以改成100维或者其他
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()
#
# print('Found %s word vectors.' % len(embeddings_index))

# doc2vecModel = Doc2Vec.load(os.path.join(DOC2VEC_DIR, 'doc2vec.model'))
# for idx, line in enumerate(doc2vecModel.vocab):
#     embeddings_index[line] = np.asarray(doc2vecModel[line], dtype='float32')

    # for idx,lv in enumerate(doc2vecModel[line]):
    #     aa = (str(lv)+" ")

# second, prepare text samples and their labels
print('Processing text dataset')  # 下面这段代码，主要作用是读入训练样本，并读入相应的标签，并给每个出现过的单词赋一个编号，比如单词peking对应编号100

client = MongoClient('localhost', 27017)
db = client.AnalystReport
collection = db.LabeledSegReport

texts = []          # 存储训练样本的list
labels_index = {}   # 词到词编号的字典，比如peking对应100
labels = []         # 存储训练样本，类别编号的文本，比如文章a属于第1类文本
for label_seg_report in collection.find({}):
    if label_seg_report['label_six_month'] > 1: # 无效样本点
        continue
    texts.append(label_seg_report['seg_conent'])
    labels.append(label_seg_report['label_six_month'])


# for name in sorted(os.listdir(TEXT_DATA_DIR)):
#     path = os.path.join(TEXT_DATA_DIR, name)
#     if os.path.isdir(path):
#         label_id = len(labels_index)
#         labels_index[name] = label_id
#         for fname in sorted(os.listdir(path)):
#             if fname.isdigit():
#                 fpath = os.path.join(path, fname)
#                 if sys.version_info < (3,):
#                     f = open(fpath)
#                 else:
#                     f = open(fpath, encoding='latin-1')
#                 texts.append(f.read())
#                 f.close()
#                 labels.append(label_id)

print('Found %s texts.' % len(texts))  # 输出训练样本的数量

# finally, vectorize the text samples into a 2D integer tensor,下面这段代码主要是将文本转换成文本序列，比如 文本'我爱中华' 转化为[‘我爱’，'中华']，然后再将其转化为[101,231],最后将这些编号展开成词向量，这样每个文本就是一个2维矩阵，这块可以参加本文‘<span style="font-size:18px;">二.卷积神经网络与词向量的结合’这一章节的讲述</span>
# Tokenizer.fit_on_texts(texts)
# sequences = Tokenizer.texts_to_sequences(texts)

# word_index = Tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))
# sequences = texts
# data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels), 3)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set,下面这段代码，主要是将数据集分为，训练集和测试集（英文原意是验证集，但是我略有改动代码）
# indices = np.arange(data.shape[0])
# np.random.shuffle(indices)
# data = data[indices]
# labels = labels[indices]
# nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]  # 训练集
y_train = labels[:-nb_validation_samples]  # 训练集的标签
x_val = data[-nb_validation_samples:]  # 测试集，英文原意是验证集
y_val = labels[-nb_validation_samples:]  # 测试集的标签

print('Preparing embedding matrix.')

# prepare embedding matrix 这部分主要是创建一个词向量矩阵，使每个词都有其对应的词向量相对应
# nb_words = min(MAX_NB_WORDS, len(word_index))
# embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
# for word, i in word_index.items():
#     if i > MAX_NB_WORDS:
#         continue
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector  # word_index to word_embedding_vector ,<20000(nb_words)

# load pre-trained word embeddings into an Embedding layer
# 神经网路的第一层，词向量层，本文使用了预训练glove词向量，可以把trainable那里设为False
# embedding_layer = Embedding(nb_words + 1,
#                             EMBEDDING_DIM,
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             weights=[embedding_matrix],
#                             trainable=True)

print('Training model.')

# train a 1D convnet with global maxpoolinnb_wordsg

# left model 第一块神经网络，卷积窗口是5*50（50是词向量维度）
model_left = Sequential()
# model.add(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))
# model_left.add(embedding_layer)
model_left.add(Conv1D(128, 5, activation='tanh', input_dim=100))
model_left.add(MaxPooling1D(5))
model_left.add(Conv1D(128, 5, activation='tanh'))
model_left.add(MaxPooling1D(5))
model_left.add(Conv1D(128, 5, activation='tanh'))
model_left.add(MaxPooling1D(35))
model_left.add(Flatten())

# right model <span style="font-family: Arial, Helvetica, sans-serif;">第二块神经网络，卷积窗口是4*50</span>

model_right = Sequential()
# model_right.add(embedding_layer)
model_right.add(Conv1D(128, 4, activation='tanh', input_dim=100))
model_right.add(MaxPooling1D(4))
model_right.add(Conv1D(128, 4, activation='tanh'))
model_right.add(MaxPooling1D(4))
model_right.add(Conv1D(128, 4, activation='tanh'))
model_right.add(MaxPooling1D(28))
model_right.add(Flatten())

# third model <span style="font-family: Arial, Helvetica, sans-serif;">第三块神经网络，卷积窗口是6*50</span>
model_3 = Sequential()
# model_3.add(embedding_layer)
model_3.add(Conv1D(128, 6, activation='tanh', input_dim=100))
model_3.add(MaxPooling1D(3))
model_3.add(Conv1D(128, 6, activation='tanh'))
model_3.add(MaxPooling1D(3))
model_3.add(Conv1D(128, 6, activation='tanh'))
model_3.add(MaxPooling1D(30))
model_3.add(Flatten())

merged = Merge([model_left, model_right, model_3],
               mode='concat')  # 将三种不同卷积窗口的卷积层组合 连接在一起，当然也可以只是用三个model中的一个，一样可以得到不错的效果，只是本文采用论文中的结构设计
model = Sequential()
model.add(merged)  # add merge
model.add(Dense(128, activation='tanh'))  # 全连接层
model.add(Dense(len(labels_index), activation='softmax'))  # softmax，输出文本属于20种类别中每个类别的概率

# 优化器我这里用了adadelta，也可以使用其他方法
model.compile(loss='categorical_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])

# =下面开始训练，nb_epoch是迭代次数，可以高一些，训练效果会更好，但是训练会变慢
model.fit(x_train, y_train, nb_epoch=3)

score = model.evaluate(x_train, y_train, verbose=0)  # 评估模型在训练集中的效果，准确率约99%
print('train score:', score[0])
print('train accuracy:', score[1])
score = model.evaluate(x_val, y_val, verbose=0)  # 评估模型在测试集中的效果，准确率约为97%，迭代次数多了，会进一步提升
print('Test score:', score[0])
print('Test accuracy:', score[1])
