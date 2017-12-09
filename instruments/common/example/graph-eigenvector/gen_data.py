# coding: utf-8
"""
This script generate adjacency matrix from friendster dataset.
"""
import sys

import scipy as sp
import scipy.sparse

#           node_id         eigen_value
# 10K-th :  27979           5.89382978
# 100K-th:  259257          7.82133753
# 1M-th :   2720146
# 10M-th :  25509957
end_id = 259257
data, row_id, col_id = [], [], []
mapping = {}
counter = 0


def check_mapping(node):
    global counter
    if node not in mapping:
        mapping[node] = counter
        counter += 1

def insert_edge(src, dst):
    data.append(1)
    row_id.append(mapping[src])
    col_id.append(mapping[dst])

with open('/home/ubuntu/friendster/com-friendster.ungraph.txt', 'r') as graph:
    for i, line in enumerate(graph):
        try:
            if line.startswith('#'):
                continue
            if i % 1000000 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            src, dst = line.split()
            src, dst = int(src), int(dst)
            if src > end_id:
                break
            if dst > end_id:
                continue
            check_mapping(src)
            check_mapping(dst)
            insert_edge(src, dst)
            insert_edge(dst, src)
        except:
            print('Error occurred when processing line {}.\n{}'.format(i, line))
            raise
    matrix = sp.sparse.csr_matrix((data, (row_id, col_id)))
    sp.sparse.save_npz('/home/ubuntu/friendster/friendster-100K.npz', matrix)
