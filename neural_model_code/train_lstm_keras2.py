'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classication of newsgroup messages into 20 different categories).

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

from __future__ import print_function
import os
import numpy as np
np.random.seed(1337)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense,Dropout, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.optimizers import *
from keras.models import Sequential
from keras.layers import Merge
import sys
import codecs
from gensim.models import word2vec
import theano
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.svm import SVC,LinearSVC
import pandas as pd
import multiprocessing
from keras.layers.recurrent import LSTM


#BASE_DIR = '.'
#GLOVE_DIR = BASE_DIR + '/glove.6B/'
#TEXT_DATA_DIR = BASE_DIR + '/20_newsgroup/'
MAX_NB_WORDS = 80000
MAX_SEQUENCE_LENGTH = 100
batch_size = 32
VALIDATION_SPLIT = 0.3

# second, prepare text samples and their labels
print('Processing text dataset')

texts=[]
texts_train=[]
texts_test=[]
x_train=[]
y_train=[]
#padded_sentences_train = []
t=0
lines=codecs.open('/home/jinxiu_li/usr/data/lstm/data_train_seg_nlpir_index_sample.txt', 'rU', 'utf-8').readlines()
for line in lines:
    values = line.split('\t')
    t=len(y_train)
#    sentence = values[1].split(" ")
#    num_padding = MAX_SEQUENCE_LENGTH - len(sentence)
#    new_sentence = sentence + [padding_word] * num_padding
#    x_train.append(new_sentence)
#    y_train1.append(int(values[0]))
    if values[0]=='1':
        y_train.append([1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='2':
        y_train.append([0, 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='3':
        y_train.append([0, 0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='4':
        y_train.append([0, 0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='5':
        y_train.append([0, 0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='11':
        y_train.append([0, 0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='12':
        y_train.append([0, 0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='13':
        y_train.append([0, 0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='14':
        y_train.append([0, 0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='15':
        y_train.append([0, 0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='16':
        y_train.append([0, 0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='17':
        y_train.append([0, 0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='18':
        y_train.append([0, 0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='19':
        y_train.append([0, 0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
    if values[0]=='21':
        y_train.append([0, 0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
    if values[0]=='22':
        y_train.append([0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
    if values[0]=='31':
        y_train.append([0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])
    if values[0]=='33':
        y_train.append([0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
    if values[0]=='35':
        y_train.append([0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
    if values[0]=='36':
        y_train.append([0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])
    if values[0]=='37':
        y_train.append([0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])
    if values[0]=='38':
        y_train.append([0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
    if values[0]=='39':
        y_train.append([0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
    if(t!=len(y_train)):
        texts.append(values[1])
        texts_train.append(values[1])
print("train text size: ")
print(len(texts_train))
print("train value size: ")
print(len(y_train))

x_val=[]
y_val=[]
lines=codecs.open('/home/jinxiu_li/usr/data/lstm/data_test_seg_nlpir_index_sample.txt', 'rU', 'utf-8').readlines()
for line in lines:
    values = line.split('\t')
    t=len(y_val)

#    sentence = values[1].split(" ")
#    num_padding = MAX_SEQUENCE_LENGTH - len(sentence)
#    new_sentence = sentence + [padding_word] * num_padding
#    x_val.append(new_sentence)
#    y_val1.append(int(values[0]))
    if values[0]=='1':
        y_val.append([1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='2':
        y_val.append([0, 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='3':
        y_val.append([0, 0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='4':
        y_val.append([0, 0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='5':
        y_val.append([0, 0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='11':
        y_val.append([0, 0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='12':
        y_val.append([0, 0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='13':
        y_val.append([0, 0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='14':
        y_val.append([0, 0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='15':
        y_val.append([0, 0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='16':
        y_val.append([0, 0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='17':
        y_val.append([0, 0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='18':
        y_val.append([0, 0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
    if values[0]=='19':
        y_val.append([0, 0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
    if values[0]=='21':
        y_val.append([0, 0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
    if values[0]=='22':
        y_val.append([0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
    if values[0]=='31':
        y_val.append([0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])
    if values[0]=='33':
        y_val.append([0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
    if values[0]=='35':
        y_val.append([0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
    if values[0]=='36':
        y_val.append([0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])
    if values[0]=='37':
        y_val.append([0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])
    if values[0]=='38':
        y_val.append([0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
    if values[0]=='39':
        y_val.append([0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])

    if(t!=len(y_val)):
        texts.append(values[1])
        texts_test.append(values[1])
print('Found %s texts.' % len(texts))

print("test text size: ")
print(len(texts_test))
print("test value size: ")
print(len(y_val))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

print(type(sequences))
print(sequences[0])
print(sequences[1])
print(len(sequences))
word_index = tokenizer.word_index

tokenizer_train = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer_train.fit_on_texts(texts_train)
sequences_train = tokenizer_train.texts_to_sequences(texts_train)

word_index_train = tokenizer_train.word_index

tokenizer_test = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer_test.fit_on_texts(texts_test)
sequences_test = tokenizer_test.texts_to_sequences(texts_test)

word_index_test = tokenizer_test.word_index
print('Found %s unique tokens.' % len(word_index))

data_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

x_train = data_train
#y_train = labels[:-nb_validation_samples]
x_val = data_test
#y_val = labels[-nb_validation_samples:]

print("train size: ")
print(x_train.shape)
print("train value size: ")
print(len(y_train))

print('Training model.')


model = Sequential()
model.add(Embedding(MAX_NB_WORDS, 128, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
model.add(Dense(128, activation='softmax'))
model.add(Dense(23, activation='softmax'))

#model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])

# happy learning!
model.fit(x_train, y_train, nb_epoch=2)
#model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=4)
    #          nb_epoch=2, batch_size=128)
#model.fit([x_train,x_train], y_train, nb_epoch=5)


score = model.evaluate(x_train, y_train, verbose=0) 
print (type(score))
print('train score:', score[0])
print('train accuracy:', score[1])
score = model.evaluate(x_val, y_val, verbose=0) 
print('Test score:', score[0])
print('Test accuracy:', score[1])

results = model.predict(x_val[0])
print(type(results))
print(results)
print(result.shape())

#results = model.predict(x_val, batch_size=32, verbose=0)


#f = open("/home/jinxiu_li/usr/model/word2vec/result.txt",'w',encoding='utf-8')
#for res in results:
#    f.write(res)
#    f.write("\n")
#f.close();
