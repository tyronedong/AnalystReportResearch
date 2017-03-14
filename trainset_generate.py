# -*- coding: utf-8 -*-

from pymongo import *
import datetime
import jieba
import jieba.posseg as pseg

jieba.load_userdict('./data/userdict_mix_label.txt')
jieba.enable_parallel(4)

client = MongoClient('localhost', 27017)
db = client.AnalystReport
collection1 = db.Report
collection2 = db.StockPrice

collection_ins = db.LabeledSegReport


timedelta_half_year = datetime.timedelta(180)
timedelta_three_month = datetime.timedelta(90)
timedelta_one_month = datetime.timedelta(30)
timedelta_minus_one_day = datetime.timedelta(-1)

list = []
ins_count = 0

for report in collection1.find({}):
    date = report['Date']
    newdate_half_year = date + timedelta_half_year
    newdate_three_month = date + timedelta_three_month
    newdate_one_month = date + timedelta_one_month

    price_half_year = 0
    price_three_month = 0
    price_one_month = 0
    price_now = 0
    # res = collection2.find({"date":newdate_half_year, "StockCode":report["StockCode"]})
    # 获取半年后的股价信息
    newid_half_year = '{0}_{1}{2:0>2}{3:0>2}'.format(report["StockCode"], newdate_half_year.year,
                                                     newdate_half_year.month, newdate_half_year.day)
    res = collection2.find_one({"_id": newid_half_year})
    count = 0
    while res == None:
        newdate_half_year += timedelta_minus_one_day
        newid_half_year = '{0}_{1}{2:0>2}{3:0>2}'.format(report["StockCode"], newdate_half_year.year,
                                                         newdate_half_year.month, newdate_half_year.day)
        res = collection2.find_one({"_id": newid_half_year})
        count += 1
        if count > 7:
            break
    if res != None:
        price_half_year = res["price"]

    #获取三个月后的股价信息
    newid_three_month = '{0}_{1}{2:0>2}{3:0>2}'.format(report["StockCode"], newdate_three_month.year,
                                                       newdate_three_month.month, newdate_three_month.day)
    res = collection2.find_one({"_id": newid_three_month})
    count = 0
    while res == None:
        newdate_three_month += timedelta_minus_one_day
        newid_three_month = '{0}_{1}{2:0>2}{3:0>2}'.format(report["StockCode"], newdate_three_month.year,
                                                           newdate_three_month.month, newdate_three_month.day)
        res = collection2.find_one({"_id": newid_three_month})
        count += 1
        if count > 7:
            break
    if res != None:
        price_three_month = res["price"]

    #获取一个月后的股价信息
    newid_one_month = '{0}_{1}{2:0>2}{3:0>2}'.format(report["StockCode"], newdate_one_month.year,
                                                     newdate_one_month.month, newdate_one_month.day)
    res = collection2.find_one({"_id": newid_one_month})
    count = 0
    while res == None:
        newdate_one_month += timedelta_minus_one_day
        newid_one_month = '{0}_{1}{2:0>2}{3:0>2}'.format(report["StockCode"], newdate_one_month.year,
                                                           newdate_one_month.month, newdate_one_month.day)
        res = collection2.find_one({"_id": newid_one_month})
        count += 1
        if count > 7:
            break
    if res != None:
        price_one_month = res["price"]

    #获取当前的股价信息
    id_now = '{0}_{1}{2:0>2}{3:0>2}'.format(report["StockCode"], date.year, date.month, date.day)
    res = collection2.find_one({"_id": id_now})
    count = 0
    while res == None:
        date += timedelta_minus_one_day
        id_now = '{0}_{1}{2:0>2}{3:0>2}'.format(report["StockCode"], date.year, date.month, date.day)
        res = collection2.find_one({"_id": id_now})
        count += 1
        if count > 7:
            break
    if res != None:
        price_now = res["price"]

    # 筛选数据
    if price_now == 0:
        continue
    if report["Content"] == "":
        continue

    rate_six_month = (price_half_year-price_now)/price_now
    label_six_month = 1 if rate_six_month >= 0.05 else 0 if rate_six_month > -0.05 else -1

    rate_three_month = (price_three_month - price_now) / price_now
    label_three_month = 1 if rate_three_month >= 0.05 else 0 if rate_three_month > -0.05 else -1

    rate_one_month = (price_one_month - price_now) / price_now
    label_one_month = 1 if rate_one_month >= 0.05 else 0 if rate_one_month > -0.05 else -1

    #标记误差点
    if price_half_year == 0:
        label_six_month = 100
    if price_three_month == 0:
        label_three_month = 100
    if price_one_month == 0:
        label_one_month =100

    report['label_six_month'] = label_six_month
    report['label_three_month'] = label_three_month
    report['label_one_month'] = label_one_month

    #加一步分词操作
    seg_list = pseg.cut(report['Content'])
    seg_clean_list = []
    # print(", ".join(seg_list))
    for w in seg_list:
        if w.flag == 'm':
            continue
        if w.flag == 'x':
            continue
        seg_clean_list.append(w.word)
    report["seg_conent"] = seg_clean_list

    del report["Content"]

    list.append(report)
    ins_count += 1

    if ins_count>100:
        collection_ins.insert_many(list)
        print report["_id"]
        ins_count = 0
        list = []
    # collection_ins.insert(report)
    # print  report
