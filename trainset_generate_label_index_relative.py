# -*- coding: utf-8 -*-

from pymongo import *
import datetime

client = MongoClient('localhost', 27017)
db = client.AnalystReport
collection1 = db.LabeledSegReport
collection2 = db.StockPrice

collection_ins = db.RatedSegReport

timedelta_half_year = datetime.timedelta(180)
timedelta_three_month = datetime.timedelta(90)
timedelta_one_month = datetime.timedelta(30)
timedelta_minus_one_day = datetime.timedelta(-1)

list = []
ins_count = 0

for report in collection1.find({}):
    date = report['Date']

    # 股价涨跌幅
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

    # 获取三个月后的股价信息
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

    # 获取一个月后的股价信息
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

    # 获取当前的股价信息
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

    price_rate_six_month = (price_half_year - price_now) / price_now
    price_rate_three_month = (price_three_month - price_now) / price_now
    price_rate_one_month = (price_one_month - price_now) / price_now

    report['price_rate_six_month'] = price_rate_six_month
    report['price_rate_three_month'] = price_rate_three_month
    report['price_rate_one_month'] = price_rate_one_month

    # HS300
    newdate_half_year = date + timedelta_half_year
    newdate_three_month = date + timedelta_three_month
    newdate_one_month = date + timedelta_one_month

    price_half_year = 0
    price_three_month = 0
    price_one_month = 0
    price_now = 0

    # 获取半年后的股价信息
    newid_half_year = 'HS300_{0}{1:0>2}{2:0>2}'.format(newdate_half_year.year,
                                                     newdate_half_year.month, newdate_half_year.day)
    res = collection2.find_one({"_id": newid_half_year})
    count = 0
    while res == None:
        newdate_half_year += timedelta_minus_one_day
        newid_half_year = 'HS300_{0}{1:0>2}{2:0>2}'.format(newdate_half_year.year,
                                                         newdate_half_year.month, newdate_half_year.day)
        res = collection2.find_one({"_id": newid_half_year})
        count += 1
        if count > 7:
            break
    if res != None:
        price_half_year = res["price"]

    # 获取三个月后的股价信息
    newid_three_month = 'HS300_{0}{1:0>2}{2:0>2}'.format(newdate_three_month.year,
                                                       newdate_three_month.month, newdate_three_month.day)
    res = collection2.find_one({"_id": newid_three_month})
    count = 0
    while res == None:
        newdate_three_month += timedelta_minus_one_day
        newid_three_month = 'HS300_{0}{1:0>2}{2:0>2}'.format(newdate_three_month.year,
                                                           newdate_three_month.month, newdate_three_month.day)
        res = collection2.find_one({"_id": newid_three_month})
        count += 1
        if count > 7:
            break
    if res != None:
        price_three_month = res["price"]

    # 获取一个月后的股价信息
    newid_one_month = 'HS300_{0}{1:0>2}{2:0>2}'.format(newdate_one_month.year,
                                                     newdate_one_month.month, newdate_one_month.day)
    res = collection2.find_one({"_id": newid_one_month})
    count = 0
    while res == None:
        newdate_one_month += timedelta_minus_one_day
        newid_one_month = 'HS300_{0}{1:0>2}{2:0>2}'.format(newdate_one_month.year,
                                                         newdate_one_month.month, newdate_one_month.day)
        res = collection2.find_one({"_id": newid_one_month})
        count += 1
        if count > 7:
            break
    if res != None:
        price_one_month = res["price"]

    # 获取当前的股价信息
    id_now = 'HS300_{0}{1:0>2}{2:0>2}'.format(date.year, date.month, date.day)
    res = collection2.find_one({"_id": id_now})
    count = 0
    while res == None:
        date += timedelta_minus_one_day
        id_now = 'HS300_{0}{1:0>2}{2:0>2}'.format(date.year, date.month, date.day)
        res = collection2.find_one({"_id": id_now})
        count += 1
        if count > 7:
            break
    if res != None:
        price_now = res["price"]

    # 筛选数据
    if price_now == 0:
        continue

    hs300_rate_six_month = (price_half_year - price_now) / price_now
    hs300_rate_three_month = (price_three_month - price_now) / price_now
    hs300_rate_one_month = (price_one_month - price_now) / price_now

    report['hs300_rate_six_month'] = hs300_rate_six_month
    report['hs300_rate_three_month'] = hs300_rate_three_month
    report['hs300_rate_one_month'] = hs300_rate_one_month

    # ZZ500
    newdate_half_year = date + timedelta_half_year
    newdate_three_month = date + timedelta_three_month
    newdate_one_month = date + timedelta_one_month

    price_half_year = 0
    price_three_month = 0
    price_one_month = 0
    price_now = 0

    # 获取半年后的股价信息
    newid_half_year = 'ZZ500_{0}{1:0>2}{2:0>2}'.format(newdate_half_year.year,
                                                     newdate_half_year.month, newdate_half_year.day)
    res = collection2.find_one({"_id": newid_half_year})
    count = 0
    while res == None:
        newdate_half_year += timedelta_minus_one_day
        newid_half_year = 'ZZ500_{0}{1:0>2}{2:0>2}'.format(newdate_half_year.year,
                                                         newdate_half_year.month, newdate_half_year.day)
        res = collection2.find_one({"_id": newid_half_year})
        count += 1
        if count > 7:
            break
    if res != None:
        price_half_year = res["price"]

    # 获取三个月后的股价信息
    newid_three_month = 'ZZ500_{0}{1:0>2}{2:0>2}'.format(newdate_three_month.year,
                                                       newdate_three_month.month, newdate_three_month.day)
    res = collection2.find_one({"_id": newid_three_month})
    count = 0
    while res == None:
        newdate_three_month += timedelta_minus_one_day
        newid_three_month = 'ZZ500_{0}{1:0>2}{2:0>2}'.format(newdate_three_month.year,
                                                           newdate_three_month.month, newdate_three_month.day)
        res = collection2.find_one({"_id": newid_three_month})
        count += 1
        if count > 7:
            break
    if res != None:
        price_three_month = res["price"]

    # 获取一个月后的股价信息
    newid_one_month = 'ZZ500_{0}{1:0>2}{2:0>2}'.format(newdate_one_month.year,
                                                     newdate_one_month.month, newdate_one_month.day)
    res = collection2.find_one({"_id": newid_one_month})
    count = 0
    while res == None:
        newdate_one_month += timedelta_minus_one_day
        newid_one_month = 'ZZ500_{0}{1:0>2}{2:0>2}'.format(newdate_one_month.year,
                                                         newdate_one_month.month, newdate_one_month.day)
        res = collection2.find_one({"_id": newid_one_month})
        count += 1
        if count > 7:
            break
    if res != None:
        price_one_month = res["price"]

    # 获取当前的股价信息
    id_now = 'ZZ500_{0}{1:0>2}{2:0>2}'.format(date.year, date.month, date.day)
    res = collection2.find_one({"_id": id_now})
    count = 0
    while res == None:
        date += timedelta_minus_one_day
        id_now = 'ZZ500_{0}{1:0>2}{2:0>2}'.format(date.year, date.month, date.day)
        res = collection2.find_one({"_id": id_now})
        count += 1
        if count > 7:
            break
    if res != None:
        price_now = res["price"]

    # 筛选数据
    if price_now == 0:
        continue

    zz500_rate_six_month = (price_half_year - price_now) / price_now
    zz500_rate_three_month = (price_three_month - price_now) / price_now
    zz500_rate_one_month = (price_one_month - price_now) / price_now

    report['zz500_rate_six_month'] = zz500_rate_six_month
    report['zz500_rate_three_month'] = zz500_rate_three_month
    report['zz500_rate_one_month'] = zz500_rate_one_month

    list.append(report)
    ins_count += 1

    if ins_count > 100:
        collection_ins.insert_many(list)
        print report["_id"]
        ins_count = 0
        list = []