
file = open('./data/userdict_mix.txt')
lines = file.readlines()

new_lines = []
for line in lines:
    l = line.replace('\n','')
    ll = l + ' nt\n'
    new_lines.append(ll)

writefile = open('./data/userdict_mix_label.txt','w')
writefile.writelines(new_lines)
writefile.flush()

file.close()
writefile.close()
