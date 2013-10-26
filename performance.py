"""Methods for testing the performance of various data-handling approaches."""

from plsa import *
import random
import string
import tables
import os

def setup_random(num_D, num_W, num_Z):
    """Sets up SQL tables with random corpus and probability matrices for
    testing purposes.
    
    Args
        num_D (int) - number of documents in corpus.
        num_W (int) - number of words in the corpus model.
        num_Z (int) - number of topics in the topic model.
        
    Returns
        bool.
    """
    
    connection = psycopg2.connect(host="127.0.0.1", user="erickpeirson", database="erickpeirson")
    cur = connection.cursor()
    
    start = time.time()
    
    # Generate some random data
    print "generating some random data"
    document_word = np.random.random( size = ( num_D, num_W) )
    document_topic = np.random.random( size = ( num_D, num_Z) )
    topic_word = np.random.random( size = ( num_Z, num_W ) )
    topic = np.random.random( size = ( num_D, num_W, num_Z ) )

    # Create tables
    print "creating tables"
    cur.execute("CREATE TABLE document_word(document integer, term_frequency float["+str(num_W)+"]);")
    for i in xrange(num_D):
        cur.execute("INSERT INTO document_word VALUES ('"+str(i)+"', '{" + ", ".join( [ str(w) for w in document_word[i]] ) + "}' );")

    cur.execute("CREATE TABLE document_topic(document integer, topic float["+str(num_Z)+"]);")
    cur.execute("CREATE TABLE topic_word(topic integer, word float["+str(num_W)+"]);")
    cur.execute("CREATE TABLE topic(document integer, word_topic float["+str(num_W)+"]["+str(num_Z)+"]);")

    # Populate with data
    print "populating tables with data"
    for i in xrange(num_D):
       cur.execute("INSERT INTO document_topic VALUES ('"+str(i)+"', '{" + ", ".join( [ str(w) for w in document_topic[i]] ) + "}' );")
    for i in xrange(num_Z):
       cur.execute("INSERT INTO topic_word VALUES ('"+str(i)+"', '{" + ", ".join( [ str(w) for w in topic_word[i]] ) + "}' );")
    for i in xrange(num_D):
        vals = []
        for x in xrange(num_W):
            vals.append("{" + ", ".join ( [ str(b) for b in topic[i][x] ] ) + "}")
        cur.execute("INSERT INTO topic VALUES ('"+str(i)+"', '{" + ", ".join(vals) + "}');")

    connection.commit() # TODO: set up some exception handling.

    print "set up random dataset in " + str( time.time() - start )

    return True

def teardown_random():
    """Drops all of the testing tables created by setup_random().
    
    Args
        none.
    
    Returns
        none.
    """

    connection = psycopg2.connect(host="127.0.0.1", user="erickpeirson", database="erickpeirson")
    cur = connection.cursor()

    print "tearing down testing tables."

    cur.execute("DROP TABLE document_word;")
    cur.execute("DROP TABLE document_topic;")
    cur.execute("DROP TABLE topic_word;")
    cur.execute("DROP TABLE topic;")
    connection.commit() # TODO: some exception handling.

    print "all testing tables dropped."



def benchtest(num_D=10, num_W=200, num_Z=10):
    """For testing performance of EM algorithm under various conditions, using
    a PostgreSQL database. Generates random data based on args, sets up SQL 
    tables, and runs the EM algorithm.
    
    Args
        num_D (int) - number of documents in corpus.
        num_W (int) - number of words in the corpus model.
        num_Z (int) - number of topics in the topic model.
    
    Returns
        tuple. 
            float. total iteration time.
            float. duration of E-step.
            float. duration of M-step part A.
            float. duration of M-step part B.
    """


    print "starting performance test with " + str(num_D) + " documents, " + str(num_W) + " words, and " + str(num_Z) + " topics."
        
    
    setup_random(num_D, num_W, num_Z) # Sets up a bunch of tables with fake data.

    print "-" * 80
    print "starting iteration."
    
    iteration_start = time.time()
    
    # E-step
    for d in xrange(1, num_D):
        do_estep((d, num_W, num_Z, 1, 'asdf'))
    estep_duration = time.time() - iteration_start
    print "E-step took " + str( estep_duration )

    # M-step part A
    m_a_start = time.time()
    for z in xrange(1, num_Z):
        do_mstep_a((z, num_W, num_D, 1, 'asdf'))
    mstep_a_duration = time.time() - m_a_start
    print "M-step A took " + str( mstep_a_duration )

    # M-step part B
    m_b_start = time.time()
    for d in xrange(1, num_D):
        do_mstep_b((d, num_Z, num_W, 1, 'asdf'))
    mstep_b_duration = time.time() - m_b_start
    print "M-step B took " + str( mstep_b_duration )

    iteration_time = time.time() - iteration_start
    print "finished iteration in " + str( iteration_time )

    print "-" * 80
    
    teardown_random()   # Drops all of the tables used for testing.

    print "done."

    return iteration_time, estep_duration, mstep_a_duration, mstep_b_duration


def benchtest_multi(num_D=10, num_W=200, num_Z=10, processes=4):
    """For testing performance of EM algorithm under various conditions, using
    a PostgreSQL database and multiprocessing. Generates random data based on 
    args, sets up SQL tables, and runs the EM algorithm.
    
    Args
        num_D (int) - number of documents in corpus.
        num_W (int) - number of words in the corpus model.
        num_Z (int) - number of topics in the topic model.
        processes (int) - number of parallel processes.
    
    Returns
        tuple. 
            float. total iteration time.
            float. duration of E-step.
            float. duration of M-step part A.
            float. duration of M-step part B.
    """
    
    global connection, cur

    print "starting benchtest with " + str(num_D) + " documents, " + str(num_W) + " words, and " + str(num_Z) + " topics."
    
    connection = psycopg2.connect(host="127.0.0.1", user="erickpeirson", database="erickpeirson")
    cur = connection.cursor()
    
    setup_random(num_D, num_W, num_Z) # Sets up a bunch of tables with fake data.

    print "-" * 80
    print "starting iteration."
    
    iteration_start = time.time()
    
    # E-step
    pool = Pool(processes)
    TASKS = [ (d, num_W, num_Z, 2, 'asdf') for d in xrange(1, num_D) ]
    jobs = pool.imap(do_estep, TASKS)
    pool.close()
    pool.join()
    
    finished = False
    while not finished:
        try:
            jobs.next()
        except Exception as E:
            finished = True
            print E

    estep_duration = time.time() - iteration_start
    print "E-step took " + str( estep_duration )

    # M-step part A
    m_a_start = time.time()

    pool = Pool(processes)
    TASKS = [ (z, num_W, num_Z, 2, 'asdf') for z in xrange(1, num_Z) ]
    jobs = pool.imap(do_mstep_a, TASKS)
    pool.close()
    pool.join()

    finished = False
    while not finished:
        try:
            jobs.next()
        except:
            finished = True

    mstep_a_duration = time.time() - m_a_start
    print "M-step A took " + str( mstep_a_duration )

    # M-step part B
    m_b_start = time.time()

    pool = Pool(processes)
    TASKS = [ ( d, num_Z, num_W, 2, 'asdf') for d in xrange(1, num_D) ]
    jobs = pool.imap(do_mstep_b, TASKS)
    pool.close()
    pool.join()

    finished = False
    while not finished:
        try:
            jobs.next()
        except:
            finished = True

    mstep_b_duration = time.time() - m_b_start
    print "M-step B took " + str( mstep_b_duration )

    iteration_time = time.time() - iteration_start
    print "finished iteration in " + str( iteration_time )

    print "-" * 80
    
    teardown_random()   # Drops all of the tables used for testing.

    print "done."
    

    return iteration_time, estep_duration, mstep_a_duration, mstep_b_duration


def benchtest_memcache(num_D=10, num_W=200, num_Z=10):
    """For testing performance of EM algorithm under various conditions, using
    a PostgreSQL database and memcache. Generates random data based on args, 
    sets up SQL tables, and runs the EM algorithm.
    
    Args
        num_D (int) - number of documents in corpus.
        num_W (int) - number of words in the corpus model.
        num_Z (int) - number of topics in the topic model.
    
    Returns
        tuple. 
            float. total iteration time.
            float. average duration of E-step chunk.
            float. average duration of M-step part A chunk.
            float. average duration of M-step part B chunk.
    """

    print "starting benchtest with " + str(num_D) + " documents, " + str(num_W) + " words, and " + str(num_Z) + " topics."
        
    
    setup_random(num_D, num_W, num_Z) # Sets up a bunch of tables with fake data.

    id = ''.join(random.choice(string.ascii_uppercase) for x in xrange(4))

    print "-" * 80
    print "starting iteration."
    
    iteration_start = time.time()

    # E-step
    for d in xrange(1, num_D):
        do_estep_mc((d, num_W, num_Z, 3, id))
    estep_duration = time.time() - iteration_start
    print "E-step took " + str( estep_duration )

    # M-step part A
    m_a_start = time.time()
    for z in xrange(1, num_Z):
        do_mstep_a((z, num_W, num_D, 3, id))
    mstep_a_duration = time.time() - m_a_start
    print "M-step A took " + str( mstep_a_duration )

    # M-step part B
    m_b_start = time.time()
    for d in xrange(1, num_D):
        do_mstep_b((d, num_Z, num_W, 3, id))
    mstep_b_duration = time.time() - m_b_start
    print "M-step B took " + str( mstep_b_duration )

    iteration_time = time.time() - iteration_start
    print "finished iteration in " + str( iteration_time )

    print "-" * 80

    teardown_random()   # Drops all of the tables used for testing.

    print "done."



#    return iteration_time, estep_duration, mstep_a_duration, mstep_b_duration
    return 

def benchtest_memcache_multi(num_D=10, num_W=200, num_Z=10, processes=4):
    """For testing performance of EM algorithm under various conditions, using
    a PostgreSQL database, memcache, and multiprocessing. Generates random data 
    based on args, sets up SQL tables, and runs the EM algorithm.
    
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

    global connection, cur

    print "starting benchtest with " + str(num_D) + " documents, " + str(num_W) + " words, and " + str(num_Z) + " topics."
    
    connection = psycopg2.connect(host="127.0.0.1", user="erickpeirson", database="erickpeirson")
    cur = connection.cursor()
    
    setup_random(num_D, num_W, num_Z) # Sets up a bunch of tables with fake data.

    id = ''.join(random.choice(string.ascii_uppercase) for x in xrange(4))

    print "-" * 80
    print "starting iteration."
    
    iteration_start = time.time()
    
    # E-step
    pool = Pool(processes)
    TASKS = [ (d, num_W, num_Z, 4, id) for d in xrange(1, num_D) ]
    jobs = pool.imap(do_estep_mc, TASKS)
    pool.close()
    pool.join()
    
    finished = False
    while not finished:
        try:
            jobs.next()
        except Exception as E:
            finished = True
            print E

    estep_duration = time.time() - iteration_start
    print "E-step took " + str( estep_duration )

    # M-step part A
    m_a_start = time.time()

    pool = Pool(processes)
    TASKS = [ (z, num_W, num_Z, 4, id) for z in xrange(1, num_Z) ]
    jobs = pool.imap(do_mstep_a_mc, TASKS)
    pool.close()
    pool.join()

    finished = False
    while not finished:
        try:
            jobs.next()
        except:
            finished = True

    mstep_a_duration = time.time() - m_a_start
    print "M-step A took " + str( mstep_a_duration )

    # M-step part B
    m_b_start = time.time()

    pool = Pool(processes)
    TASKS = [ ( d, num_Z, num_W, 4, id) for d in xrange(1, num_D) ]
    jobs = pool.imap(do_mstep_b_mc, TASKS)
    pool.close()
    pool.join()

    finished = False
    while not finished:
        try:
            jobs.next()
        except:
            finished = True

    mstep_b_duration = time.time() - m_b_start
    print "M-step B took " + str( mstep_b_duration )

    iteration_time = time.time() - iteration_start
    print "finished iteration in " + str( iteration_time )

    print "-" * 80
    
    teardown_random()   # Drops all of the tables used for testing.

    print "done."

    return iteration_time, estep_duration, mstep_a_duration, mstep_b_duration




