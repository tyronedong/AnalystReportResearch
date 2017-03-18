# -*- coding: utf-8 -*-
# 从word2phrase的输出文件中提取词组

data_file = file('./data/phrases/seg_phrases_4.txt')

phrases = set()
count = 0
for line in data_file:
    print count
    # print line
    words = line.split()
    for word in words:
        if '_' in word:
            new_phrase = word.replace('_', '')
            if (len(word)-len(new_phrase)) > 3:
                continue
            # phrases.append(word.replace('_', ''))
            phrases.add(new_phrase)
    # if count > 1000:
    #     break
    count += 1

print len(phrases)
list = [(i+' n\n') for i in phrases]
print len(list)
# print list
output_file = file('./data/phrases/user_dict_phrases.txt', mode='w')
output_file.writelines(list)
output_file.flush()
output_file.close()
