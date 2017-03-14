# -*- coding: utf-8 -*-
from pymongo import *
import gensim
import os
import numpy as np
import gc
import ast

word_embeddings_dict = {}
word_index_dict = {} # 应该是实际出现过的vocab的index



# 获取单词到词向量的映射，单词到编号的映射
word2vecModel = gensim.models.Word2Vec.load('./Word2Vec/word2vec.model')
for idx, word in enumerate(word2vecModel.wv.vocab):
    word_embeddings_dict[word] = np.asarray(word2vecModel[word], dtype='float32')
    index = len(word_index_dict)
    word_index_dict[word] = index

client = MongoClient('localhost', 27017)
db = client.AnalystReport
collection = db.LabeledSegReport

# 把每个文本序列映射成以数字序列表示的词向量形式，读入训练数据和标记数据
train_texts = []              # 存储训练样本的list
train_labels = []             # 存储训练样本，类别编号的文本，比如文章a属于第1类文本
test_texts = []               # 存储测试样本的list
test_labels = []              # 存储测试样本，类别编号的文本，比如文章a属于第1类文本
# for label_seg_report in collection.find({}).limit(1000):
cur_count = 0
train_cate_1_count_091011 = 0
train_cate_2_count_091011 = 0
train_cate_3_count_091011 = 0
train_cate_1_count = 0
train_cate_2_count = 0
train_cate_3_count = 0
test_cate_1_count = 0
test_cate_2_count = 0
test_cate_3_count = 0
test_year_2015 = 0
test_year_2016 = 0
test_year_2012_1 = 0
test_year_2012_2 = 0
test_year_2012_3 = 0
for label_seg_report in collection.find({}):
    if label_seg_report['label_six_month'] > 1: # 无效样本点
        continue
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
    if label_seg_report['Date'].year < 2015:
        if label_seg_report['label_six_month']==1:
            train_cate_1_count += 1;
            if label_seg_report['Date'].year<2012:
                train_cate_1_count_091011 += 1
            elif label_seg_report['Date'].year==2012:
                test_year_2012_1 += 1
        elif label_seg_report['label_six_month']==0:
            train_cate_2_count += 1
            if label_seg_report['Date'].year<2012:
                train_cate_2_count_091011 += 1
            elif label_seg_report['Date'].year==2012:
                test_year_2012_2 += 1
        elif label_seg_report['label_six_month']==-1:
            train_cate_3_count += 1
            if label_seg_report['Date'].year<2012:
                train_cate_3_count_091011 += 1
            elif label_seg_report['Date'].year==2012:
                test_year_2012_3 += 1
    else:
        if label_seg_report['label_six_month']==1:
            test_cate_1_count += 1;
        elif label_seg_report['label_six_month']==0:
            test_cate_2_count += 1
        elif label_seg_report['label_six_month']==-1:
            test_cate_3_count += 1
        if label_seg_report['Date'].year == 2015:
            test_year_2015 += 1
        else:
            test_year_2016 += 1

print 'train 1:', train_cate_1_count
print 'train 0:', train_cate_2_count
print 'train -1:', train_cate_3_count
print 'test 1:', test_cate_1_count
print 'test 0:', test_cate_2_count
print 'test -1:', test_cate_3_count
print 'test 2015:', test_year_2015
print 'test 2016:', test_year_2016
print 'train 091011 1:',train_cate_1_count_091011
print 'train 091011 2:',train_cate_2_count_091011
print 'train 091011 3:',train_cate_3_count_091011
print 'test 12 1:',test_year_2012_1
print 'test 12 0:',test_year_2012_2
print 'test 12 -1:',test_year_2012_3
