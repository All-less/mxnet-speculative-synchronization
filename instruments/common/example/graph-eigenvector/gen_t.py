# coding: utf-8
import sys

import scipy as sp
import scipy.sparse


A = sp.sparse.load_npz('friendster-100K.npz')
dim = A.shape[0]

T = sp.sparse.csr_matrix((dim, dim), dtype='float64')
for i in range(dim):
    if i % 10000 == 0:
        print(i)
        sys.stdout.flush()
    T += A[i].T * A[i]

sp.sparse.save_npz('friendster-100K-T.npz', T)

