# -*- coding: utf-8 -*-

import jieba
import jieba.posseg as pseg

jieba.load_userdict('./data/userdict_mix_label.txt')
jieba.load_userdict('./data/phrases/user_dict_phrases.txt')

seg_list = pseg.cut(u'')