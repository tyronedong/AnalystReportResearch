# -*- coding: utf-8 -*-
import xlrd
from pymongo import *

def store_report():
    client = MongoClient('localhost', 27017)
    db = client.StockReport
    ins_collection = db.ReportInfo

    report_excel_data = xlrd.open_workbook('tabel_2_report.xlsx')
    table = report_excel_data.sheet_by_index(0)

    nrows = table.nrows
    for i in range(1, nrows):
        # init variable
        rpt_d_info = {}
        row = table.row_values(i)

        # clean unvalid rows
        if row[3] == 0:
            continue
        if row[13] == u'EN':
            continue

        rpt_d_info['_id'] = row[0]
        rpt_d_info['Date'] = xlrd.xldate.xldate_as_datetime(row[7], 0)
        rpt_d_info['ORG_CL'] = int(row[6])
        rpt_d_info['Title'] = row[8]
        rpt_d_info['Keywords'] = row[9]
        rpt_d_info['Authors'] = row[10].split(',')
        rpt_d_info['Pages'] = int(row[11])
        rpt_d_info['Byte_Size'] = int(row[12])
        author_id_list = []
        author_id_list.append(row[14])
        if(row[15]!=u''):
            author_id_list.append(row[15])
            if (row[16] != u''):
                author_id_list.append(row[16])
        rpt_d_info['Authors_ID'] = author_id_list
        # date = xlrd.xldate.xldate_as_datetime(row[1], 0)
        ins_collection.insert_one(rpt_d_info)
        print rpt_d_info['Date'], row[0]

def store_brokerage():
    client = MongoClient('localhost', 27017)
    db = client.StockReport
    ins_collection = db.BrokerageInfo

    report_excel_data = xlrd.open_workbook('tabel_4_org.xlsx')
    table = report_excel_data.sheet_by_index(0)

    nrows = table.nrows
    for i in range(1, nrows):
        # init variable
        rpt_d_info = {}
        row = table.row_values(i)

        # clean unvalid rows
        if row[4] == 0:
            continue
        # if row[13] == u'EN':
        #     continue

        rpt_d_info['_id'] = row[0]
        # rpt_d_info['Date'] = xlrd.xldate.xldate_as_datetime(row[7], 0)
        rpt_d_info['ORG_CL'] = int(row[7])
        rpt_d_info['SName_CN'] = row[8]
        rpt_d_info['FName_CN'] = row[16]
        rpt_d_info['SName_EN'] = row[9]
        rpt_d_info['FName_EN'] = row[10]
        rpt_d_info['ORG_URL'] = row[17]
        rpt_d_info['ORG_BI'] = row[19]
        # date = xlrd.xldate.xldate_as_datetime(row[1], 0)
        ins_collection.insert_one(rpt_d_info)
        print row[0]

def store_report_cls():
    client = MongoClient('localhost', 27017)
    db = client.StockReport
    ins_collection = db.ReportCLSInfo

    report_excel_data = xlrd.open_workbook('tabel_5_cls.xlsx')
    table = report_excel_data.sheet_by_index(0)

    nrows = table.nrows
    for i in range(1, nrows):
        # init variable
        rpt_d_info = {}
        row = table.row_values(i)

        # clean unvalid rows
        if row[4] == 0:
            continue
        # if row[13] == u'EN':
        #     continue

        rpt_d_info['_id'] = row[0]
        # rpt_d_info['Date'] = xlrd.xldate.xldate_as_datetime(row[7], 0)
        rpt_d_info['Report_GUID'] = row[6]
        rpt_d_info['Report_Type'] = row[15]
        rpt_d_info['SEC_CD'] = row[8]
        # date = xlrd.xldate.xldate_as_datetime(row[1], 0)
        ins_collection.insert_one(rpt_d_info)
        print row[0]

def store_security():
    client = MongoClient('localhost', 27017)
    db = client.StockReport
    ins_collection = db.SecurityInfo

    report_excel_data = xlrd.open_workbook('tabel_1_sec.xlsx')
    table = report_excel_data.sheet_by_index(0)

    nrows = table.nrows
    for i in range(1, nrows):
        # init variable
        rpt_d_info = {}
        row = table.row_values(i)

        # clean unvalid rows
        if row[3] == 0:
            continue
        # if row[13] == u'EN':
        #     continue

        rpt_d_info['_id'] = row[8]
        # rpt_d_info['Date'] = xlrd.xldate.xldate_as_datetime(row[7], 0)
        rpt_d_info['SEC_CD'] = row[8]
        rpt_d_info['MKT_CL'] = row[10]
        rpt_d_info['VAR_CL'] = row[11]
        rpt_d_info['SName_CN'] = row[13]
        rpt_d_info['FName_CN'] = row[12]
        rpt_d_info['SName_EN'] = row[14]
        rpt_d_info['FName_EN'] = row[16]
        # date = xlrd.xldate.xldate_as_datetime(row[1], 0)
        ins_collection.insert_one(rpt_d_info)
        print row[0]

def store_person():
    client = MongoClient('localhost', 27017)
    db = client.StockReport
    ins_collection = db.PersonInfo

    report_excel_data = xlrd.open_workbook('tabel_3_person.xlsx')
    table = report_excel_data.sheet_by_index(0)

    nrows = table.nrows
    for i in range(1, nrows):
        # init variable
        rpt_d_info = {}
        row = table.row_values(i)

        # clean unvalid rows
        if row[4] == 0:
            continue
        # if row[13] == u'EN':
        #     continue

        rpt_d_info['_id'] = row[0]
        # rpt_d_info['Date'] = xlrd.xldate.xldate_as_datetime(row[7], 0)
        rpt_d_info['Name_CN'] = row[6]
        rpt_d_info['Name_EN'] = row[9]
        rpt_d_info['Sex'] = row[8]
        rpt_d_info['Email'] = row[24]
        rpt_d_info['Company'] = row[25]
        # date = xlrd.xldate.xldate_as_datetime(row[1], 0)
        ins_collection.insert_one(rpt_d_info)
        print row[0]

# store_person()
