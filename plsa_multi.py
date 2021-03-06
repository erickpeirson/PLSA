import re
import numpy as np
import time
import random
from multiprocessing import Pool


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

# These get passed out to workers.
def do_estep(d):
    result = np.zeros([vocabulary_size, number_of_topics])
    
    for w in range(vocabulary_size):
        prob = document_topic_prob[d, :] * topic_word_prob[:, w]
        if sum(prob) == 0.0:
            print 'exit'
        else:
            normalize(prob)
        result[w] = prob
    return result

def do_mstep_a(t):
    result = np.zeros([ vocabulary_size ])

    for w_index in range(vocabulary_size):
        s = 0
        for d_index in range(number_of_documents):
            count = term_doc_matrix[d_index][w_index]
            s = s + count * topic_prob[d_index, w_index, t]
        result[w_index] = s
    normalize(result)
    return result

def do_mstep_b(d):
    result = np.zeros( [ number_of_topics ])
    for z in range(number_of_topics):
        s = 0
        for w_index in range(vocabulary_size):
            count = term_doc_matrix[d][w_index]
            s = s + count * topic_prob[d, w_index, z]
        result[z] = s
    normalize(result)
    return result

np.set_printoptions(threshold='nan')


def test_iteration(num_D, num_W, num_Z, processes=4, verbose=False):
    """Do a single iteration."""
    
    global term_doc_matrix, document_topic_prob, topic_word_prob, topic_prob, vocabulary_size, number_of_topics, number_of_documents
    
    number_of_documents = num_D
    vocabulary_size = num_W
    number_of_topics = num_Z
    
    print "generating some random data"
    term_doc_matrix = np.random.random( size = ( num_D, num_W) )
    document_topic_prob = np.random.random( size = ( num_D, num_Z) )
    topic_word_prob = np.random.random( size = ( num_Z, num_W ) )
    topic_prob = np.random.random( size = ( num_D, num_W, num_Z ) )
    
    
    iteration_start = time.time()
    
    # E-step
    pool = Pool(processes)
    TASKS = range(0, num_D)
    jobs = pool.imap(do_estep, TASKS)
    pool.close()
    pool.join()
    
    finished = False
    while not finished:
        try:
            jobs.next()
        except Exception as E:
            finished = True

    estep_duration = time.time() - iteration_start
    print "E-step took " + str( estep_duration )

    # M-step part A
    m_a_start = time.time()

    pool = Pool(processes)
    TASKS = range(0, num_Z)
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
    TASKS = range(0, num_D)
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
    

    return iteration_time, estep_duration, mstep_a_duration, mstep_b_duration

class MultiCorpus(object):

    '''
    A collection of documents, with multiprocessing pLSA method.
    '''

    def __init__(self):
        '''
        Initialize empty document list.
        '''
        self.documents = []


    def add_document(self, document):
        '''
        Add a document to the corpus.
        '''
        self.documents.append(document)


    def build_vocabulary(self):
        '''
        Construct a list of unique words in the corpus.
        '''
        # ** ADD ** #
        # exclude words that appear in 90%+ of the documents
        # exclude words that are too (in)frequent
        discrete_set = set()
        for document in self.documents:
            for word in document.words:
                discrete_set.add(word)
        self.vocabulary = list(discrete_set)
        
        


    def plsa(self, nt, max_iter, processes=4):
        '''
        Model topics using multiprocessing.
        
        Args
            nt (int): number of topic
            max_iter (int): maximum number of iterations
            processes (int): maximum number of parallel processes (default=4)
            
        '''
        print "EM iteration begins. Num topics: " + str(nt) + "; Iterations: " + str(max_iter) + "; Processes: " + str(processes)
        
        global vocabulary_size, number_of_documents, number_of_topics, document_topic_prob, topic_word_prob, term_doc_matrix, topic_prob
        
        # Get vocabulary and number of documents.
        self.build_vocabulary()
        number_of_documents = len(self.documents)
        vocabulary_size = len(self.vocabulary)
        number_of_topics = nt
        
        # build term-doc matrix
        term_doc_matrix = np.zeros([number_of_documents, vocabulary_size], dtype = np.int)
        for d_index, doc in enumerate(self.documents):
            term_count = np.zeros(vocabulary_size, dtype = np.int)
            for word in doc.words:
                if word in self.vocabulary:
                    w_index = self.vocabulary.index(word)
                    term_count[w_index] = term_count[w_index] + 1
            term_doc_matrix[d_index] = term_count

        # Create the counter arrays.
        document_topic_prob = np.zeros([number_of_documents, number_of_topics], dtype=np.float) # P(z | d)
        topic_word_prob = np.zeros([number_of_topics, vocabulary_size], dtype=np.float) # P(w | z)
        topic_prob = np.zeros([number_of_documents, vocabulary_size, number_of_topics], dtype=np.float) # P(z | d, w)

        # Initialize
        print "Initializing..."
        
        # randomly assign values
        document_topic_prob = np.random.random(size = (number_of_documents, number_of_topics))
        for d_index in range(number_of_documents):
            normalize(document_topic_prob[d_index]) # normalize for each document
        topic_word_prob = np.random.random(size = (number_of_topics, vocabulary_size))
        for z in range(number_of_topics):
            normalize(topic_word_prob[z]) # normalize for each topic


        # Run the EM algorithm using multiprocessing
        for iteration in range(max_iter):
            start = time.time()

            # e step
            topic_prob = []
            pool = Pool(processes)
            TASKS = []
            for d_index in range(number_of_documents):
                TASKS.append(d_index)
            jobs = pool.imap(do_estep, TASKS)
            pool.close()
            pool.join()

            finished = False
            while not finished:
                try:
                    topic_prob.append(jobs.next())
                except Exception as e:
                    finished = True
            topic_prob = np.asarray(topic_prob)


            # m step - first part
            pool = Pool(processes)
            topic_word_prob = []
            TASKS = []
            for z_index in range(number_of_topics):
                TASKS.append(z_index)
            jobs = pool.imap(do_mstep_a, TASKS)

            pool.close()
            pool.join()
            
            finished = False
            while not finished:
                try:
                    topic_word_prob.append(jobs.next())
                except:
                    finished = True
            topic_word_prob = np.asarray(topic_word_prob)


            # m step - second part
            pool = Pool(processes)
            document_topic_prob = []
            TASKS = []

            for d_index in range(number_of_documents):
                TASKS.append(d_index)
            jobs = pool.imap(do_mstep_b, TASKS)
            pool.close()
            pool.join()


            finished = False
            while not finished:
                try:
                    document_topic_prob.append(jobs.next())
                except:
                    finished = True

            document_topic_prob = np.asarray(document_topic_prob)

            print "iteration " + str(iteration) + " completed in " + str(time.time() - start) + " seconds."
            print "document probability variance: " + str( np.var(document_topic_prob))

        self.topic_word_prob = topic_word_prob
        self.document_topic_prob = document_topic_prob

