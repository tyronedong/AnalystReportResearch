# -*- coding: utf-8 -*-
from gensim.models import Doc2Vec
import os

DOC2VEC_DIR = '../model/'
WRITE_FILE = '../model/array.txt'

lines = []
doc2vecModel = Doc2Vec.load(os.path.join(DOC2VEC_DIR, 'doc2vec.model'))
for idx, word in enumerate(doc2vecModel.vocab):
    # word_embeddings_dict[word] = np.asarray(doc2vecModel[word], dtype='float32')
    line = word
    line += ' '
    for item in doc2vecModel[word]:
        line += str(item)
        line += ' '
    line += '\n'
    lines.append(line.encode('utf-8'))

w_file = open(WRITE_FILE, mode='w')
w_file.writelines(lines)
w_file.flush()
w_file.close()
