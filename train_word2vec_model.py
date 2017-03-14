# -*- coding: utf-8 -*-
import gensim
import os

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                line = unicode(line, 'utf-8')
                yield line.split()

sentences = MySentences('./data/sentences')
model = gensim.models.Word2Vec(sentences, size=100, workers=8)

# print model[u'结算']
model.save('./Word2Vec/word2vec.model')
