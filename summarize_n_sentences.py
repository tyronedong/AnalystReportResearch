# -*- coding: utf-8 -*-
from pymongo import *
from snownlp import SnowNLP
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool

client = MongoClient('localhost', 27017)
db = client.AnalystReport
collection = db.Report
collection_ins = db.RankedReport

#
#  count = 0
# list = []
# for report in collection.find({}):
#     if report["Content"] == "":
#         continue
#
#     new_report = {}
#     s = SnowNLP(report['Content'])
#     ranked_sentences = s.summary(1500)
#     new_report['_id'] = report['_id']
#     new_report['ranked_content'] = ranked_sentences
#
#     collection_ins.insert(new_report)
#     print report['_id']
#     # list.append(new_report)
#     # count += 1
#     #
#     # if count > 100:
#     #     collection_ins.insert_many(list)
#     #     print report['_id']
#     #     list = []
#     #     count = 0
# list = []
# for report in collection.find({}):
#     list.append(report)

def execute(report):
    if report['Content'] == "":
        return
    new_report = {}
    s = SnowNLP(report['Content'])
    ranked_sentences = s.summary(1500)
    new_report['_id'] = report['_id']
    new_report['ranked_content'] = ranked_sentences
    print report['_id']
    collection_ins.insert(new_report)

#pool = ThreadPool(8)
pool = Pool(processes=6)
pool.map(execute, collection.find({}))
pool.close()
pool.join()

