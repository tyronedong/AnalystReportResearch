# -*- coding: utf-8 -*-

from pymongo import *
import datetime
import jieba
import jieba.posseg as pseg

seg_file = file('./data/phrases/seg_phrases_4.txt')
for line in seg_file.readlines():
    print line
