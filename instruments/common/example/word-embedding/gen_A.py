# coding: utf-8
"""
This script convert output from decode.cpp to a numpy array.
"""
import numpy as np

A = []

with open('cooccurrence.shuf.text', 'r') as f:
    for index, line in enumerate(f.xreadlines()):
        parts = line.split()
        A.append([ int(parts[0]), int(parts[1]), float(parts[2]) ])
        if index % 600000 == 0:
            print index / 600000

np.save(open('cooccurrence.shuf.npy', 'wb'), np.array(A))
