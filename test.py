from functools import partial
from multiprocessing import Pool, cpu_count
import time
def mm(i, j):
    time.sleep((100 - i)/10)
    return 1

pool = Pool()
a = pool.starmap(mm, zip(range(100), range(100)))
pool.close()

for i in a:
    print(i)