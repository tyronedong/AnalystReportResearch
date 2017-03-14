# -*- coding: utf-8 -*-

from pymongo import *

client = MongoClient('localhost', 27017)
db = client.AnalystReport
collection1 = db.Reports
insert_coll = db.Report

table_file = open('./data/id_code_table.txt')
lines = table_file.readlines()
dict = {}
for line in lines:
    if(line == '\n'):
        continue
    tokens = line.split('\t')
    dict[tokens[0]] = tokens[1].replace('\n', '')

for report in collection1.find({}):
    if dict.get(report["_id"]) == None:
        continue
    report["StockCode"] = dict[report["_id"]]
    insert_coll.insert(report)
    print report["_id"], report["StockCode"]


