import time
import random
from multiprocessing import Pool
import string
import tables
import os
import numpy as np

def normalize(vec):
    """Normalize a vector to be a probablistic representation.
    
    Args
        vec (numpy array) - one-dimensional array.
    
    Returns
        none.
    """
    s = sum(vec)
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

    Returns
        True.

    Raises
        nothing.
    """

    d, num_W, num_Z = args
    result = np.zeros( [ num_W, num_Z ] )
    
    dt = document_topic[d, :] # Read from disk.

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
                print 'exit' # TODO: Raise an exception or something.
            else:
                normalize(probs)
                result[w] = probs

    update_topic(d, result)
    return True

def update_topic(d, result):
    """Accepts the result of each E-step subprocess, and updates the topic
    probability matrix accordingly.
    
    Args
        d (int) - index of a document.
        result (array-like) - vector of topic probabilities for that document.
    
    Returns
        True.
    
    Notes
        topic and f are both assumed to be globals.
    """
    global topic, f
    
    topic[d, :] = result
    f.flush()   # Out of paranoia, if nothing else.
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
            
    Returns
        True.

    Raises
        nothing.
    """
    
    z, num_W, num_D  = args
    result = np.zeros( [ num_W ] )
    
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

    update_topic_word(z, result)
    return True

def update_topic_word(z, result):
    """Accepts the result of each M-step (part A) subprocess, and updates the 
    topic-word probability matrix accordingly.
    
    Args
        z (int) - index (row) of a topic in the topic_word matrix.
        result (array-like) - vector of word probabilities for that topic.
        
    Returns
        True.
    
    Notes
        topic_word and f are both assumed to be globals.
    """
    global topic_word, f
    
    topic_word[z, : ] = result
    f.flush()
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

    Returns


    Raises
        nothing.
    """
    
    d, num_Z, num_W = args
    result = np.zeros( [ num_Z ] )
    
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

    update_document_topic(d, result)
    return True

def update_document_topic(d, result):
    """Accepts the result of each M-step (part B) subprocess, and updates the 
    document_topic probability matrix accordingly.
    
    Args
        d (int) - index (row) of a document in the document_topic matrix.
        result (array-like) - vector of topic probabilities for that document.
        
    Returns
        True.
    
    """
    global document_topic, f
    
    document_topic[d, : ] = result
    f.flush()   # Paranoia.
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
        
    Notes
        TODO: Use tables.EArray instead of tables.Array, and add data 
            incrementally, to cut down on large memory overhead.
    """

    # Each set of test data gets a new HDF5 repository.
    id = ''.join(random.choice(string.ascii_uppercase) for x in xrange(4))
    r_path = path + "/" + id + ".h5"

    f = tables.openFile(r_path, "w")
    g = f.createGroup("/", "g")

    # Generate some data.
    d_w = f.createArray("/g", "document_word", np.random.random( size = ( num_D, num_W)))
    f.flush()

    d_t = f.createArray("/g", "document_topic", np.random.random( size = ( num_D, num_Z)))
    f.flush()
    
    t_w = f.createArray("/g", "topic_word", np.random.random( size = ( num_Z, num_W)))
    f.flush()
    
    t_ = f.createArray("/g", "topic", np.random.random( size = ( num_D, num_W, num_Z )))
    f.flush()
    
    f.close()

    return r_path

def teardown_random_tables(path):
    """Deletes the HDF5 repository created by setup_random_tables().
    
    Args
        path (str) - full path to HDF5 repository.
    
    Returns
        bool. True if file is deleted successfully.
    """

    return os.remove(path)

def performance_test(num_D=10, num_W=200, num_Z=10, processes=4, path="./", data="asdf"):
    """For testing performance of EM algorithm under various conditions, using
    PyTables and multiprocessing. Generates random data based on args, sets up 
    PyTables, and runs the EM algorithm.
    
    Args
        num_D (int) - number of documents in corpus.
        num_W (int) - number of words in the corpus model.
        num_Z (int) - number of topics in the topic model.
        processes (int) - number of parallel processes.
    
    Returns
        tuple. 
            float. total iteration time.
            float. average duration of E-step chunk.
            float. average duration of M-step part A chunk.
            float. average duration of M-step part B chunk.
    """

    global document_word, document_topic, topic_word, topic, f

    print "starting performance test with:"
    print "\t" + str(num_D) + " documents"
    print "\t" + str(num_W) + " words"
    print "\t" + str(num_Z) + " topics"

    print "set up tables with data"
    
    # Generate data and create HDF5 repository.
    r_path = setup_random_tables(num_D, num_W, num_Z, path)
    
    f = tables.openFile(r_path, 'a')

    # To compare performance to non-PyTables approach.
    if data == "tables":    # Uses PyTables (yay!).
        document_word = f.root.g.document_word
        document_topic = f.root.g.document_topic
        topic_word = f.root.g.topic_word
        topic = f.root.g.topic
    else:   # Just use numpy arrays in memory (ack!).
        document_word = np.random.random( size = ( num_D, num_W) )
        document_topic = np.random.random( size = ( num_D, num_Z) )
        topic_word = np.random.random( size = ( num_Z, num_W ) )
        topic = np.random.random( size = ( num_D, num_W, num_Z ) )

    print "start iteration"
    iteration_start = time.time()

    # E-step.
    pool = Pool(processes)
    TASKS = [ (d, num_W, num_Z) for d in xrange(0, num_D) ]
    jobs = pool.imap(do_estep, TASKS)
    pool.close()
    pool.join()
    
    finished = False
    while not finished:
        try:
            jobs.next()
        except StopIteration:
            finished = True

    estep_duration = time.time() - iteration_start
    print "E-step took " + str( estep_duration )

    # M-step part A
    m_a_start = time.time()

    pool = Pool(processes)
    TASKS = [ (z, num_W, num_Z ) for z in xrange(0, num_Z) ]
    jobs = pool.imap(do_mstep_a, TASKS)
    pool.close()
    pool.join()

    finished = False
    while not finished:
        try:
            jobs.next()
        except StopIteration:
            finished = True

    mstep_a_duration = time.time() - m_a_start
    print "M-step A took " + str( mstep_a_duration )

    # M-step part B
    m_b_start = time.time()

    pool = Pool(processes)
    TASKS = [ ( d, num_Z, num_W ) for d in xrange(0, num_D) ]
    jobs = pool.imap(do_mstep_b, TASKS)
    pool.close()
    pool.join()

    finished = False
    while not finished:
        try:
            jobs.next()
        except StopIteration:
            finished = True

    mstep_b_duration = time.time() - m_b_start
    print "M-step B took " + str( mstep_b_duration )

    iteration_time = time.time() - iteration_start
    print "finished iteration in " + str( iteration_time )

    print "clean up"
    f.close()
    teardown_random_tables(r_path)

    print "test complete"

    return iteration_time, estep_duration, mstep_a_duration, mstep_b_duration

class PLSA:
    def __init__(self, hdf5_path, num_D, num_W, num_Z=10):
        self.hdf5_path = hdf5_path
        self.num_D = num_D
        self.num_W = num_W
        self.num_Z = num_Z
        self.iteration = 0
        self.variance_log = []
    
    def train(self, max_iter=10, processes=4):
        """Train the pLSA model using the EM algorithm.
        
        Args
            max_iter (int) - Maximum number of iterations to perform on this
                training run.
            processes (int) - Number of parallel processes to spawn at a time.
    
        Returns
            list (int) - Variance in document-topic probability at the end of
                each iteration.
        """

        global document_word, document_topic, topic_word, topic, f

        start = time.time()

        # Multiprocessing approach.
        # The EM algorithm is divided into three parts:
        #   E-step: update P(z | d, w)
        #   M-step (part A): update P(w | z)
        #   M-step (part B): update P(z | d)
        #
        # Each step is divided into sub-tasks based on the highest iteration
        #   level, which are distributed to workers using multiprocessing. For
        #   example, the E-step is divided into N(d) sub-tasks. Each worker
        #   calls a handler (e.g. update_topic, for E-step) that updates the
        #   appropriate probability matrix, which keeps results vectors out of
        #   the main process (out of paranoia, if nothing else).

        for local_iteration in xrange(0, max_iter):
            # TODO: check for asymptote in self.variance_log, based on a
            #   specifiable delta threshold.
            
            
            # Load data.
            f = tables.openFile(self.hdf5_path, 'a') # TODO: check that path was provided.

            document_word = f.root.g.document_word
            document_topic = f.root.g.document_topic
            topic_word = f.root.g.topic_word
            topic = f.root.g.topic
            
            # E-step.
            pool = Pool(processes)
            TASKS = [ (d, self.num_W, self.num_Z) for d in xrange(0, self.num_D) ]
            jobs = pool.imap(do_estep, TASKS)
            pool.close()
            pool.join()
            
            finished = False
            while not finished:
                try:
                    jobs.next()
                except StopIteration:
                    finished = True

            # M-step part A
            m_a_start = time.time()

            pool = Pool(processes)
            TASKS = [ (z, self.num_W, self.num_Z ) for z in xrange(0, self.num_Z) ]
            jobs = pool.imap(do_mstep_a, TASKS)
            pool.close()
            pool.join()

            finished = False
            while not finished:
                try:
                    jobs.next()
                except StopIteration:
                    finished = True

            # M-step part B
            m_b_start = time.time()

            pool = Pool(processes)
            TASKS = [ ( d, self.num_Z, self.num_W ) for d in xrange(0, self.num_D) ]
            jobs = pool.imap(do_mstep_b, TASKS)
            pool.close()
            pool.join()

            finished = False
            while not finished:
                try:
                    jobs.next()
                except StopIteration:
                    finished = True

            # Keep track of variance in document_topic probability, as an
            #  indication of progress in the model training process. Usually
            #  sigmoid.
            variance = np.var(document_topic)
            self.variance_log.append(variance)

            print "finished iteration " + str(local_iteration+1) + " (" + str(self.iteration + 1) + " overall) of " + str(max_iter)
            print "document_topic probability variance: " + str(variance)

            self.iteration += 1     # Global counter for the model.
        
            f.flush()
            f.close()
            
        print "training complete. " + str(max_iter) + " iterations in " + str(time.time() - start) + " seconds."
        return self.variance_log

