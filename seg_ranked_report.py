# -*- coding: utf-8 -*-
from pymongo import *
import jieba
import jieba.posseg as pseg
import re

jieba.load_userdict('./data/phrases/user_dict_phrases.txt')
jieba.enable_parallel(8)

client = MongoClient('localhost', 27017)
db = client.AnalystReport
collection1 = db.RatedSegReport
collection2 = db.RankedReport
ins_collection = db.SegRankedReport

for report in collection1.find({}):
    pair_report = collection2.find_one({'_id':report['_id']})
    ranked_content_list = pair_report['ranked_content']
    content =  u'。'.join(ranked_content_list)
    seg_list = pseg.cut(content)
    new_content_list = []
    for w in seg_list:
        if w.flag == 'm':
            continue
        if w.flag == 'x':
            continue
        new_content_list.append(w.word)
    report['seg_ranked_content'] = new_content_list
    report.pop('ranked_content')
    ins_collection.insert_one(report)
