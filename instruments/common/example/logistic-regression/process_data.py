# coding: utf-8
import sys

arg = int(sys.argv[1])
input_name = 'day_{}'.format(arg)
td1_name = 'train_data_{}.csv'.format(arg * 2)
tl1_name = 'train_label_{}.csv'.format(arg * 2)
ed1_name = 'eval_data_{}.csv'.format(arg * 2)
el1_name = 'eval_label_{}.csv'.format(arg * 2)
td2_name = 'train_data_{}.csv'.format(arg * 2 + 1)
tl2_name = 'train_label_{}.csv'.format(arg * 2 + 1)
ed2_name = 'eval_data_{}.csv'.format(arg * 2 + 1)
el2_name = 'eval_label_{}.csv'.format(arg * 2 + 1)

with open(input_name, 'r') as f, \
    open(td1_name, 'w') as td1,\
    open(tl1_name, 'w') as tl1,\
    open(ed1_name, 'w') as ed1,\
    open(el1_name, 'w') as el1,\
    open(td2_name, 'w') as td2,\
    open(tl2_name, 'w') as tl2,\
    open(ed2_name, 'w') as ed2,\
    open(el2_name, 'w') as el2:
    for index, line in enumerate(f.xreadlines()):
        parts = [ p if p != '' else '-1' for p in line.split('\t')[:13] ]
        ((tl1 if (index / 2) % 20 != 0 else el1) if index % 2 == 0 else (tl2 if (index / 2) % 20 != 0 else el2)).write(parts[0] + '\n')
        ((td1 if (index / 2) % 20 != 0 else ed1) if index % 2 == 0 else (td2 if (index / 2) % 20 != 0 else ed2)).write(','.join(parts[1:]) + '\n')

