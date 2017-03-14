# -*- coding: utf-8 -*-
import os
import gc

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                line = unicode(line, 'utf-8')
                yield line.split()

sentences = MySentences('./data/sentences')

word_total = 0
bi_word_total = 0
tri_word_total = 0
word_dict = {}
bi_word_dict = {}
tri_word_dict = {}
count = 0
for sentence in sentences:
    last_word = ''
    last_last_word = ''
    for word in sentence:
        # 处理单个word
        if word_dict.has_key(word):
            word_dict[word] += 1
        else:
            word_dict[word] = 1
        word_total += 1

        if last_word == '':
            last_word = word
            continue

        # # 处理bi_word
        # bi_word = last_word + '_' + word
        # if bi_word_dict.has_key(bi_word):
        #     bi_word_dict[bi_word] += 1
        # else:
        #     bi_word_dict[bi_word] = 1
        # bi_word_total += 1


        if last_last_word == '':
            last_last_word = last_word
            continue
        # 处理tri_word
        tri_word= last_last_word + '_' + last_word + '_' + word
        if tri_word_dict.has_key(tri_word):
            tri_word_dict[tri_word] += 1
        else:
            tri_word_dict[tri_word] = 1
        tri_word_total += 1

        last_last_word = last_word
        last_word = word
    # if count > 10000:
    #     break
    print 'loaded sentences: ', count
    count += 1

# print 'calculating bi ps...'
# bi_word_option_dict = {}
# for (bi_word, num) in bi_word_dict.items():
#     if num < 10:
#         continue
#     words = bi_word.split('_')
#     if len(words) is not 2:
#         continue
#     p_w1 = word_dict.get(words[0])*1.0/word_total
#     p_w2 = word_dict.get(words[1])*1.0/word_total
#     p_w1w2 = bi_word_dict.get(bi_word)*1.0/bi_word_total
#
#     bi_word_option_dict[bi_word] = p_w1w2 - p_w1*p_w2
#     # print bi_word, (p_w1w2-p_w1*p_w2)
#
# # bi_word_option_dict = sorted(bi_word_option_dict.items(), lambda x, y: cmp(x[1], y[1]))
# #
# # for (word, num) in bi_word_option_dict:
# #     print word, num
# print 'sorting bi...'
# list = sorted(bi_word_option_dict.items(), key=lambda d: d[1], reverse=True)
# output_count = 0
# print 'writing bi to disk...'
# output_file = file('./data/phrases/bi_words_filter_10.txt', 'w')
# for (word, val) in list:
#     output_file.write(word.encode('utf-8'))
#     output_file.write('\n')
#     if val < 0:
#         print 'threashold', val
#         break
#     # output_count += 1
#     # if output_count > 10000:
#     #     print 'threshold: ', num
#     #     break
#
# output_file.flush()
# output_file.close()
#
# del bi_word_option_dict
# del list
# del bi_word_dict
# gc.collect()

print 'calculating tri ps...'
tri_word_option_dict = {}
for (tri_word, num) in tri_word_dict.items():
    if num < 10:
        continue
    words = tri_word.split('_')
    if len(words) is not 3:
        continue
    p_w1 = word_dict.get(words[0])*1.0/word_total
    p_w2 = word_dict.get(words[1])*1.0/word_total
    p_w3 = word_dict.get(words[2])*1.0/word_total
    p_w1w2w3 = tri_word_dict.get(tri_word)*1.0/tri_word_total

    tri_word_option_dict[tri_word] = p_w1w2w3/(p_w1*p_w2*p_w3)
    # print bi_word, (p_w1w2-p_w1*p_w2)

# bi_word_option_dict = sorted(bi_word_option_dict.items(), lambda x, y: cmp(x[1], y[1]))
#
# for (word, num) in bi_word_option_dict:
#     print word, num
print 'sorting tri...'
list = sorted(tri_word_option_dict.items(), key=lambda d: d[1], reverse=True)
output_count = 0
print 'writing tri to disk...'
output_file = file('./data/phrases/tri_words_filter_10.txt', 'w')
for (word, val) in list:
    if val < 100:
        print 'min threashold', val
        break
    output_file.write(word.encode('utf-8'))
    output_file.write('\n')
    # if val < 0:
    #     print 'threashold', val
    #     break
    # output_count += 1
    # if output_count > 10000:
    #     print 'threshold: ', num
    #     break

output_file.flush()
output_file.close()

print 'finish.'
