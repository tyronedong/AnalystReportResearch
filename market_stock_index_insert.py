# -*- coding: utf-8 -*-

from pymongo import *
import csv
import datetime

client = MongoClient('localhost', 27017)
db = client.AnalystReport
collection_ins = db.StockPrice

hushen_300_file = file('./data/000300.csv', 'rb')
hushen_300_file_csvreader = csv.reader(hushen_300_file)

for line in hushen_300_file_csvreader:
    dict = {}
    date = line[0]
    if date == 'Date':
        continue
    dict['_id'] = 'HS300_{0}'.format(date.replace('-',''))
    dict['StockCode']='HS300'
    dict['date'] = datetime.datetime.strptime(date,'%Y-%m-%d')
    dict['price'] = float(line[4])
    collection_ins.insert(dict)
    # print line

zhongzheng_500_file = file('./data/000905.csv', 'rb')
zhongzheng_500_file_csvreader = csv.reader(zhongzheng_500_file)

for line in zhongzheng_500_file_csvreader:
    dict = {}
    date = line[0]
    if date.find('-') == -1:
        continue
    dict['_id'] = 'ZZ500_{0}'.format(date.replace('-', ''))
    dict['StockCode'] = 'ZZ500'
    dict['date'] = datetime.datetime.strptime(date, '%Y-%m-%d')
    dict['price'] = float(line[3])
    collection_ins.insert(dict)
#print type(hushen_300_file_csvreader)
print 'finish'