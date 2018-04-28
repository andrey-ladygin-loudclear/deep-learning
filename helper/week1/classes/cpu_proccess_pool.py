import os
from multiprocessing import Pool

def tokenize(iter):
    print('tokenize', os.getpid(), iter)

rows = range(100)

with Pool(processes=os.cpu_count()) as pool:
    pool.map(tokenize, rows, 10)