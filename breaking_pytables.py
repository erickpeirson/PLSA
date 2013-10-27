import tables
import numpy as np
from multiprocessing import Pool

num_Z = 10
num_W = 2000
num_D = 4

def create_array():
    f = tables.openFile("./asdf.h5", "w")
    g = f.createGroup("/", "g")
    tw = f.createEArray("/g", "tw", atom=tables.Float64Atom(), expectedrows=num_Z, shape=(num_Z, 0))
    for i in xrange(num_W):
        tw.append(np.array([[i for x in xrange(0, 10)]]).transpose())

    f.flush()
    f.close()

def read_array():
    f = tables.openFile("./asdf.h5", "a")
    tw = f.root.g.tw
    for i in xrange(0, 100):
        print tw[:, 900]

def parallel_read(d):
    f = tables.openFile("./asdf.h5", "a")
    tw = f.root.g.tw
    for w in xrange(num_W):
        t = tw[:, w]
        print str(d) + "," + str(w) + ": " + str(t)

def read_array_parallel():
    pool = Pool(4)
    TASKS = [ d for d in xrange(0, num_D) ]
    jobs = pool.imap(parallel_read, TASKS)
    pool.close()
    pool.join()
    finished = False
    while not finished:
        try:
            jobs.next()
        except StopIteration:
            finished = True