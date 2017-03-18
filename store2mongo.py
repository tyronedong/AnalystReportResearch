# -*- coding: utf-8 -*-

from pymongo import *
import hashlib

client = MongoClient('183.174.228.13', 38018)
db = client.StockReport
collection = db.AdvReport
ins_collection = db.ConvertedReport

ins_list = []
for report in collection.find({}):
    new_report = {}
    new_report['_id'] = report['_id']
    new_report['docId'] = report['_id']
    new_report['url'] = report['_id']
    hash_md5 = hashlib.md5(report['_id'])
    new_report['urlHash'] =hash_md5.hexdigest()
    new_report['createTime'] = report['Date']
    new_report['fetchTime'] = report['Date']
    if len(report['Analysts']) > 0:
        new_report['userId'] = report['Analysts'][0]['_id']
        new_report['userName'] = report['Analysts'][0]['Name']
    new_report['title'] = report['ReportTitle']
    new_report['content'] = report['Content']
    topics = []
    topics.append(report['StockName'])
    new_report['topics'] = topics
    new_report['contentBytes'] = bytes(report['Content'].encode('utf-8'))

    dict_1 = {}
    list_1 = []
    list_1.append(report['StockRating'])
    dict_1['scockRating'] = list_1

    dict_2 = {}
    list_2 = []
    list_2.append(report['RatingChanges'])
    dict_2['ratingChange'] = list_2

    dict_3 = {}
    list_3 = []
    list_3.append(report['StockName'])
    dict_3['stockName'] = list_3

    dict_4 = {}
    list_4 = []
    list_4.append(report['ReportType'])
    dict_4['reportType'] = list_4

    dict_5 = {}
    list_5 = []
    list_5.append(report['StockCode'])
    dict_5['stockCode'] = list_5

    dict_6 = {}
    list_6 = []
    list_6.append(report['Brokerage'])
    dict_6['brokerage'] = list_6

    dict_7 = {}
    dict_7['FLI'] = report['FLISencentes']

    dict_8 = {}
    dict_8['INNOV'] = report['INNOVSentences']

    docDatas = []
    docDatas.append(dict_1)
    docDatas.append(dict_2)
    docDatas.append(dict_3)
    docDatas.append(dict_4)
    docDatas.append(dict_5)
    docDatas.append(dict_6)
    docDatas.append(dict_7)
    docDatas.append(dict_8)
    new_report['docDatas'] = docDatas

    new_report['size'] = len(report['Content'])
    new_report['groupId'] = report['StockCode']

    ins_list.append(new_report)

    if len(ins_list) > 100:
        ins_collection.insert_many(ins_list)
        print report['_id']
        ins_list = []
