from pymongo import *
import jieba
import jieba.posseg as pseg
import re

jieba.load_userdict('./data/userdict_mix_label.txt')
jieba.enable_parallel(8)

client = MongoClient('localhost', 27017)
db = client.AnalystReport
collection1 = db.Report

seg_sentences_file = open('./data/seg_sentences2.txt', mode='w')

# juhao = unicode('。'.encode('utf-8'))
# tanhao = unicode('！'.encode('utf-8'))
# wenhao = unicode('？'.encode('utf-8'))
count = 0
write_count = 0
write_list = []
for report in collection1.find({}):
    #加一步分词操作
    # sentences = re.split('。！？', report['Content'])
    # for sentence in sentences:
    print count, report['_id']
    count += 1

    seg_list = pseg.cut(report['Content'])
    seg_sentence_str = ''
    # print(", ".join(seg_list))
    for w in seg_list:
        if w.flag == 'm':
            continue
        if w.flag == 'x':
            if w.word == u'。' or w.word == u'！' or w.word == u'？':
                seg_sentence_str += '\n'
                write_list.append(seg_sentence_str.encode('utf-8'))
                if write_count > 1000:
                    seg_sentences_file.writelines(write_list)
                    write_list = []
                    write_count = 0
                # seg_sentences_file.write(seg_sentence_str.encode('utf-8'))
                write_count += 1
                seg_sentence_str = ''
            continue
        seg_sentence_str += w.word
        seg_sentence_str += ' '


    # collection_ins.insert(report)
    # print  report
seg_sentences_file.flush()
seg_sentences_file.close()
