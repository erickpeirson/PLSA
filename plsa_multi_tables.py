"""Methods for pLSA using Multiprocessing and PyTables."""

import time
import random
from multiprocessing import Pool, Queue, Manager
import string
import tables
import os
import numpy as np
import lxml.etree as ET
import Levenshtein as L
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as sw
from nltk.corpus import words as wds
from corpora import *
import tables

def normalize(vec):
    """Normalize a vector to be a probablistic representation.
    
    Args
        vec (numpy array) - one-dimensional array.
    
    Returns
        none.
    """
    s = sum(vec)
    if abs(s) == 0: print vec
    assert(abs(s) != 0.0) # the sum must not be 0
    for i in range(len(vec)):
        assert(vec[i] >= 0) # element must be >= 0
        vec[i] = vec[i] * 1.0 / s

    # TODO: raise some exceptions, instead of just asserting?

def do_estep(args):
    """Given a document d, Multiplies document-topic and topic-word vectors for 
    each word. For multiprocessing. 
    
    Args
        args (tuple):
            0: d (int) - index of a document.
            1: num_W (int) - number of words in the corpus model.
            2: num_Z (int) - number of topics in the topic model.
            3: hdf5_path (str) - path to the HDF5 repository.
            4: rqueue (Queue) - a multiprocessing.Queue.Queue object to which
                results should be sent.

    Returns
        True.
    """
    d, num_W, num_Z, hdf5_path, rqueue = args
    result = np.zeros( [ num_W, num_Z ] )

    # Load data.
    f = tables.openFile(hdf5_path, 'r') # TODO: check that path was provided.

    document_topic = f.root.g.document_topic
    topic_word = f.root.g.topic_word
    
    dt = document_topic[d, :]   # Read from disk.

    chunk_size = 100 # TODO: Experiment with different chunk sizes, to optimize.
    for i in xrange(0, num_W, chunk_size):
        if num_W > i + chunk_size:  # To handle remainder.
            chunk_end = chunk_size
        else:
            chunk_end = num_W - i
        
        tw_slice = topic_word[:, i:i+chunk_end]
        
        for x in xrange(0, chunk_end):
            w = i + x
            tw = tw_slice[:, x]
            probs = dt * tw     # Here's where the magic happens.
            
            if sum(probs) == 0.0:
                pass
            else:
                normalize(probs)
                result[w] = probs

    f.close()
    rqueue.put((d, result))

    return True

def update_topic(args):
    """Accepts the result of each E-step subprocess, and updates the topic
    probability matrix accordingly.
                0: d (int) - index of a document.
            1: result (array-like) - vector of topic probabilities for that 
                document.
    
    Args
        args (tuple)
            0: hdf5_path (str) - path to the HDF5 repository.
            1: rqueue (Queue) - multiprocessing.Queue.Queue object from which 
                results should be retrieved.
    
    Returns
        True.
    """
    
    hdf5_path, rqueue = args
    d, result = rqueue.get()    # Blocks until a new result is ready.
                                # result is a matrix of P(z, w) for document d.

    f = tables.openFile(hdf5_path, 'a')
    f.root.g.topic[d, :] = result
    f.flush()
    f.close()
    
    return True

def do_mstep_a(args):
    """First part of the M-step, for multiprocessing.

    Given a topic z, calculates a new topic-word probability from term 
    frequencies (corpus) and the topic-probability matrix.

    Args
        args (tuple)
            0: z (int) - index of a topic.
            1: num_W (int) - number of words in the corpus model.
            2: num_D (int) - number of documents in the corpus.
            3: hdf5_path (str) - path to the HDF5 repository.
            4: rqueue (Queue) - a multiprocessing.Queue.Queue object to which
                results should be sent.
        
    Returns
        True.
    """
    
    z, num_W, num_D, hdf5_path, rqueue  = args
    result = np.zeros( [ num_W ] )  # TODO: do I need this?

    # Load data.
    f = tables.openFile(hdf5_path, 'r') # TODO: check that path was provided.
    document_word = f.root.g.document_word
    topic = f.root.g.topic
    
    chunk_size = 100 # TODO: Experiment with different chunk sizes, to optimize.
    for i in xrange(0, num_D, chunk_size):
        if num_D > i + chunk_size:  # To deal with remainder.
            chunk_end = chunk_size
        else:
            chunk_end = num_D - i
    
        dw_slice = document_word[i:i+chunk_end, :]  # Read from disk.
        tp_slice = topic[i:i+chunk_end, :, z]   # Read from disk.
    
        for x in xrange(0, chunk_end):
            for w in xrange(0, num_W):
                count = dw_slice[x, w]
                tp = tp_slice[x, w]
                result[w] += count * tp

    normalize(result)   # Probabilistic interpretation.

    f.close()

    rqueue.put((z, result))
#    update_topic_word(z, result, hdf5_path)
    return True

def update_topic_word(args):
    """Accepts the result of each M-step (part A) subprocess, and updates the 
    topic-word probability matrix accordingly.
    
    Args
        args (tuple)
            0: hdf5_path (str) - path to the HDF5 repository.
            1: rqueue (Queue) - multiprocessing.Queue.Queue object from which 
                results should be retrieved.
        
    Returns
        True.
    """
    
    hdf5_path, rqueue  = args
    z, result = rqueue.get()    # Blocks until a new result is ready.
                                # result is a vector of word probabilities for
                                #  topic z.

    f = tables.openFile(hdf5_path, 'a')

    f.root.g.topic_word[z, : ] = result

    f.flush()
    f.close()
    return True

def do_mstep_b(args):
    """Second part of the M-step, for multiprocessing.

    Given a document d, calculate a new document-topic probability based on the 
    term-document matrix (corpus) and the topic-probability matrix.

    Args
        args (tuple):
            0: d (int) - index of a document.
            1: num_Z (int) - number of topics in the topic model.
            2: num_W (int) - number of words in the corpus model.
            3: hdf5_path (str) - path to the HDF5 repository.
            4: rqueue (Queue) - a multiprocessing.Queue.Queue object to which
                results should be sent.
    Returns
        True.
    """
    
    d, num_Z, num_W, hdf5_path, rqueue = args
    result = np.zeros( [ num_Z ] )

    # Load data.
    f = tables.openFile(hdf5_path, 'r') # TODO: check that path was provided.
    document_word = f.root.g.document_word
    topic = f.root.g.topic
    
    chunk_size = 100 # TODO: Experiment with different chunk sizes, to optimize.
    for i in xrange(0, num_W, chunk_size):
        if num_W > i + chunk_size:  # To handle remainder.
            chunk_end = chunk_size
        else:
            chunk_end = num_W - i
    
        dw_slice = document_word[d, i:i+chunk_end]  # Read from disk.
        tp_slice = topic[d, i:i+chunk_end]  # Read from disk.

        for x in xrange(0, chunk_end):
            count = dw_slice[x]
            for z in xrange(0, num_Z):
                tp = tp_slice[x, z]
                result[z] += count * tp # Magic.

    normalize(result)   # Probabilitistic interpretation.
    f.close()
    
    rqueue.put((d, result))
#    update_document_topic(d, result, hdf5_path)
    return True

def update_document_topic(args):
    """Accepts the result of each M-step (part B) subprocess, and updates the 
    document_topic probability matrix accordingly.
    
    Args
        args (tuple)
            0: hdf5_path (str) - path to the HDF5 repository.
            1: rqueue (Queue) - multiprocessing.Queue.Queue object from which 
                results should be retrieved.
        
    Returns
        True.
    
    """
    hdf5_path, rqueue = args
    d, result = rqueue.get()    # Blocks until a new result is ready.
                                # result is a vector of topic probabilities for
                                #  document d.
    
    f = tables.openFile(hdf5_path, 'a')
    
    f.root.g.document_topic[d, : ] = result
    f.flush()
    f.close()
    return True

def setup_random_tables(num_D, num_W, num_Z, path):
    """Sets up HDF5 repository using PyTables with random corpus and probability
    matrices, for testing purposes.
    
    Args
        num_D (int) - number of documents in corpus.
        num_W (int) - number of words in the corpus model.
        num_Z (int) - number of topics in the topic model.
        path (str) - path to directory where HDF5 repository will be created.
        
    Returns
        string. path to HDF5 repository.
    """

    # Each set of test data gets a new HDF5 repository.
    id = ''.join(random.choice(string.ascii_uppercase) for x in xrange(4))
    r_path = path + "/" + id + ".h5"

    f = tables.openFile(r_path, "w")
    g = f.createGroup("/", "g")

    # Generate some data. Use EArray to avoid hogging memory.
    print "generate random word frequencies for each document"
    d_w = f.createEArray("/g", "document_word", atom=tables.Float64Atom(), expectedrows=num_D, shape=(0, num_W))
    for i in xrange(num_D):
        d_w.append(np.random.random(size=(1, num_W)))
    f.flush()

    print "generate a random document_topic probability matrix"
    d_t = f.createEArray("/g", "document_topic", atom=tables.Float64Atom(), expectedrows=num_D, shape=(0, num_Z))
    for i in xrange(num_D):
        d_t.append(np.random.random(size=(1, num_Z)))
    f.flush()

    print "generate a random topic_word probability matrix"
    t_w = f.createEArray("/g", "topic_word", atom=tables.Float64Atom(), expectedrows=num_Z, shape=(num_Z, 0))
    for i in xrange(num_W):
        t_w.append(np.random.random(size=(num_Z, 1)))
    f.flush()

    print "generate a random topic probability matrix"
    t_ = f.createEArray("/g", "topic", atom=tables.Float64Atom(), expectedrows=num_D, shape=(0, num_W, num_Z))
    for i in xrange(num_D):
        t_.append(np.random.random(size=(1, num_W, num_Z)))
    f.flush()

    f.close()

    print "done."
    return r_path

def teardown_random_tables(path):
    """Deletes the HDF5 repository created by setup_random_tables().
    
    Args
        path (str) - full path to HDF5 repository.
    
    Returns
        bool. True if file is deleted successfully.
    """

    return os.remove(path)

def performance_test(num_D=10, num_W=200, num_Z=10, processes=4, path="./"):
    """For testing performance of EM algorithm under various conditions, using
    PyTables and multiprocessing. Generates random data based on args, sets up 
    PyTables, and runs the EM algorithm.
    
    Args
        num_D (int) - number of documents in corpus.
        num_W (int) - number of words in the corpus model.
        num_Z (int) - number of topics in the topic model.
        processes (int) - number of parallel processes.
    
    Returns
        float. total iteration time.
        float. average duration of E-step chunk.
        float. average duration of M-step part A chunk.
        float. average duration of M-step part B chunk.
    """

    print "starting performance test with:"
    print "\t" + str(num_D) + " documents"
    print "\t" + str(num_W) + " words"
    print "\t" + str(num_Z) + " topics"

    print "set up tables with data"
    
    # Generate data and create HDF5 repository.
    r_path = setup_random_tables(num_D, num_W, num_Z, path)

    print "start single iteration"
    print "-"*40
    variance, i_duration, e_duration, ma_duration, mb_duration = iterate(r_path, num_D, num_W, num_Z, processes, verbose=True)
    print "-"*40
    
    print "clean up"
    teardown_random_tables(r_path)

    print "test complete"

    return i_duration, e_duration, ma_duration, mb_duration


def iterate(hdf5_path, num_D, num_W, num_Z, processes=4, verbose=False):
    """Do a single iteration, and appends document-topic probability 
    variance to self.variances.
    """
    
    start_e = time.time()
    
    m = Manager()
    rqueue = m.Queue()

    # E-step.
    pool = Pool(processes)  # For doing all of the maths.
    TASKS = [ (d, num_W, num_Z, hdf5_path, rqueue) for d in xrange(0, num_D) ]
    jobs = pool.imap(do_estep, TASKS)
    pool.close()
    
    qpool = Pool(1)         # For processing results.
    QTASKS = [ (hdf5_path, rqueue) for d in xrange(0, num_D) ]
    qjobs = qpool.imap(update_topic, QTASKS)
    qpool.close()
    
    pool.join()
    qpool.join()
    
    finished = False    # TODO: Figure out why I have to do this... grrrr...
    while not finished:
        try:
            jobs.next()
        except StopIteration:
            finished = True

    if verbose:
        e_duration = time.time() - start_e
        print "finished e-step in " + str(e_duration)

    # M-step part A
    start_ma = time.time()

    pool = Pool(processes)  # For doing the work.
    TASKS = [ (z, num_W, num_Z, hdf5_path, rqueue) for z in xrange(0, num_Z) ]
    jobs = pool.imap(do_mstep_a, TASKS)
    pool.close()

    qpool = Pool(1)         # For processing results.
    QTASKS = [ (hdf5_path, rqueue) for z in xrange(0, num_Z) ]
    qjobs = qpool.imap(update_topic_word, QTASKS)
    qpool.close()

    pool.join()
    qpool.join()
    
    finished = False
    while not finished:
        try:
            jobs.next()
        except StopIteration:
            finished = True

    if verbose:
        ma_duration = time.time() - start_ma
        print "finished m-step part a in " + str(ma_duration)

    # M-step part B
    start_mb = time.time()

    pool = Pool(processes)  # For doing the work.
    TASKS = [ ( d, num_Z, num_W, hdf5_path, rqueue ) for d in xrange(0, num_D) ]
    jobs = pool.imap(do_mstep_b, TASKS)
    pool.close()

    qpool = Pool(1)         # For processing results.
    QTASKS = [ (hdf5_path, rqueue) for d in xrange(0, num_D) ]
    qjobs = qpool.imap(update_document_topic, QTASKS)
    qpool.close()

    pool.join()
    qpool.join()

    finished = False
    while not finished:
        try:
            jobs.next()
        except StopIteration:
            finished = True

    if verbose:
        mb_duration = time.time() - start_mb
        print "finished m-step part b in " + str(mb_duration)

        i_duration = time.time() - start_e
        print "iteration took " + str(i_duration)

    # Keep track of variance in document_topic probability, as an
    #  indication of progress in the model training process. Usually
    #  sigmoid.
    f = tables.openFile(hdf5_path, 'a') # TODO: check that path was provided.
    document_topic = f.root.g.document_topic
    variance = np.var(document_topic)
    f.close()

    if verbose:
        return variance, i_duration, e_duration, ma_duration, mb_duration
    return variance


class pLSA:
    """Provides methods for building a pLSA model from a bag-of-words corpus.
    
    TODO: 
        * Add an interface to the vocabulary table.
        * Add methods to prettily view results."""
    def __init__(self, hdf5_path, num_D, num_W, num_Z=10):
        """Prepare tables for the EM algorithm. 
        
        Args
            hdf5_path (str) - path to an HDF5 repository. It is assumed that
                this repository contains a matrix /g/document_word with shape
                ( num_D , num_W ), and a table /g/vocabulary with index/string
                pairs (where index corresponds to a column in /g/document_word).
            num_D (int) - number of documents in the corpus. Specifically, the
                number of rows in the matrix /g/document_word.
            num_W (int) - number of words in the corpus' vocabulary. 
                Specifically, the number of columns in /g/document_word.
            num_Z (int) - the number of topics to be generated.
        
        Returns
            nothing.
        """
        self.hdf5_path = hdf5_path
        self.num_D = num_D
        self.num_W = num_W
        self.num_Z = num_Z
        self.iteration = 0
        self.variance_log = []
        
        return
        
    def from_data(self):
        f = tables.openFile(self.hdf5_path, 'a')
        
        print "generate a random document_topic probability matrix"
        d_t = f.createEArray("/g", "document_topic", atom=tables.Float64Atom(), expectedrows=self.num_D, shape=(0, self.num_Z))
        for i in xrange(self.num_D):
            d_t.append(np.random.random(size=(1, self.num_Z)))
        f.flush()

        print "generate a random topic_word probability matrix"
        t_w = f.createEArray("/g", "topic_word", atom=tables.Float64Atom(), expectedrows=self.num_Z, shape=(self.num_Z, 0))
        for i in xrange(self.num_W):
            t_w.append(np.random.random(size=(self.num_Z, 1)))
        f.flush()

        print "generate a random topic probability matrix"
        t_ = f.createEArray("/g", "topic", atom=tables.Float64Atom(), expectedrows=self.num_D, shape=(0, self.num_W, self.num_Z))
        for i in xrange(self.num_D):
            t_.append(np.random.random(size=(1, self.num_W, self.num_Z)))
        f.flush()
        
        f.close()
        return

    
    def train(self, max_iter=10, processes=4):
        """Train the pLSA model using the EM algorithm.

        Multiprocessing approach.
        The EM algorithm is divided into three parts:
            E-step: update P( z | d, w )
            M-step (part A): update P( w | z )
            M-step (part B): update P( z | d )

        Each step is divided into sub-tasks based on the highest iteration
        level and distributed to workers. For example, the E-step is divided 
        into N(d) sub-tasks. Each worker calls a handler (e.g. update_topic, for
        E-step) that updates the appropriate probability matrix, which keeps 
        results vectors out of the main process.
        
        Args
            max_iter (int) - Maximum number of iterations to perform on this
                training run.
            processes (int) - Number of parallel processes to spawn at a time.
    
        Returns
            list (int) - Variance in document-topic probability at the end of
                each iteration.
        """

        start = time.time()

        for local_iteration in xrange(0, max_iter):
            # TODO: check for asymptote in self.variance_log, based on a
            #   specifiable delta threshold.
            
            self.variance_log.append(iterate(self.hdf5_path, self.num_D, self.num_W, self.num_Z, processes))
            self.iteration += 1     # Global counter for the model.
            print "finished iteration "  + str(sef.iteration)
            
        print "training complete. " + str(max_iter) + " iterations in " + str(time.time() - start) + " seconds."
        return self.variance_log

if __name__ == "__main__":
    pass
