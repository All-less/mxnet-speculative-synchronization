# coding: utf-8
import sys

import scipy as sp
import scipy.sparse


counter = 0
prev = 0

with open('/home/ubuntu/com-friendster.ungraph.txt', 'r') as graph:
    for i, line in enumerate(graph):
        try:
            if line.startswith('#'):
                continue
            src = int(line.split()[0])
            if src != prev:
                prev = src
                counter += 1
                if counter % 100000 == 0:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                if counter == 10000000:
                    print(src)
                    break
        except:
            print('Error occurred when processing line {}.\n{}'.format(i, line))
            raise
