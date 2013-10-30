import random
import unittest
import string
import tables as t
import numpy as np
from PLSA.plsa_multi_tables import *
from PLSA.corpora import *

class TestIteration(unittest.TestCase):

    def setUp(self):
        path = "./test_corpus.h5"
        multi_path = "./test_corpus_working.h5"
        self.num_W = 26
        self.num_D = 100
        self.num_Z = 20

        # Generate a vocabulary.
        vdict = {}
        i = 0
        for l in string.letters[0:26]:
            vdict[i] = l
            i += 1

        f = t.openFile(path, "w")
        g = f.createGroup("/", "g")
        v = f.createTable(g, "vocabulary", Word)
        for key, value in vdict.iteritems():
            w = v.row
            w['string'] = value
            w['index'] = key
            w.append()
            v.flush()
        f.flush()

        # Generate a random corpus with integer word-counts.
        dist = [ 0 for i in xrange(20) ] + [ 1, 1, 1, 2, 2, 3 ]
        documents = [ np.array([ dist[random.randint(0, 25)] for x in xrange(self.num_W-1) ]+[1]).reshape(1, 26) for i in xrange(self.num_D) ]
        maxs = [ np.max(doc) for doc in documents ]
        print min(maxs)

        dw = f.createEArray("/g", "document_word", atom=tables.Float64Atom(), expectedrows=20, shape=(0, 26))
        for d in documents:
            dw.append(d)
            dw.flush()
        f.flush()

        # Generate random probability matrices.

        d_t = f.createEArray("/g", "document_topic", atom=tables.Float64Atom(), expectedrows=self.num_D, shape=(0, self.num_Z))
        for i in xrange(self.num_D):
            vec = np.random.random(size=(self.num_Z, 1))
            normalize(vec)
            d_t.append(vec.reshape(1, self.num_Z))
        f.flush()


        t_w = f.createEArray("/g", "topic_word", atom=tables.Float64Atom(), expectedrows=self.num_Z, shape=(self.num_Z, 0))
        for i in xrange(self.num_W):
            vec = np.random.random(size=(self.num_Z, 1))
            normalize(vec)
            t_w.append(vec)
        f.flush()

        t_ = f.createEArray("/g", "topic", atom=tables.Float64Atom(), expectedrows=self.num_D, shape=(0, self.num_W, self.num_Z))
        for i in xrange(self.num_D):
            mat = []
            for x in xrange(self.num_W):
                vec = np.random.random(size=(self.num_Z, 1))
                normalize(vec)
                mat.append(vec.reshape(self.num_Z))
            a = np.array([mat])
            t_.append(a)
        f.flush()
        f.close()

        # Make a copy, for comparison.
        t.copyFile(path, multi_path)

    def test_consistency():
        """Make sure that we get the same results using multiprocessing as we 
        would otherwise."""
        
        # Sample matrices before EM.
        f = t.openFile(path, 'r')
        dt = f.root.g.document_topic
        t_ = f.root.g.topic
        tw = f.root.g.topic_word

        f2 = t.openFile(multi_path, 'r')
        dt2 = f2.root.g.document_topic
        t_2 = f2.root.g.topic
        tw2 = f.root.g.topic_word
        
        before = { 'dt': dt[0,1], 't_': t_[0,1,2], 'tw': tw[0,1] }
        before2 = { 'dt': dt2[0,1], 't_': t_2[0,1,2], 'tw': tw2[0,1] }

        f.close()
        f2.close()


        # Run without multiprocessing.
        m = Manager()
        q = m.Queue()

        # E-Step
        qpool = Pool(1) # Just to handle results.
        QTASKS = [ (path, q) for d in xrange(0, self.num_D) ]
        qjobs = qpool.imap(update_topic, QTASKS)
        qpool.close()

        for i in xrange(self.num_D):
            do_estep((i, self.num_W, self.num_Z, path, q))

        qpool.join()

        # M-Step A
        qpool = Pool(1) # Just to handle results.
        QTASKS = [ (path, q) for d in xrange(0, self.num_Z) ]
        qjobs = qpool.imap(update_topic_word, QTASKS)
        qpool.close()

        for i in xrange(self.num_Z):
            do_mstep_a((i, self.num_W, self.num_Z, path, q))

        qpool.join()

        # M-Step B 
        qpool = Pool(1) # Just to handle results.
        QTASKS = [ (path, q) for d in xrange(0, self.num_D) ]
        qjobs = qpool.imap(update_document_topic, QTASKS)
        qpool.close()

        for i in xrange(self.num_D):
            do_mstep_b((i, self.num_Z, self.num_W, path, q))

        qpool.join()

        # Run with multiprocessing
        iterate(multi_path, self.num_D, self.num_W, self.num_Z, 4)

        # Examine outcome.
        f = t.openFile(path, 'r')
        dt = f.root.g.document_topic
        t_ = f.root.g.topic
        tw = f.root.g.topic_word

        f2 = t.openFile(multi_path, 'r')
        dt2 = f2.root.g.document_topic
        t_2 = f2.root.g.topic
        tw2 = f.root.g.topic_word
        
        after = { 'dt': dt[0,1], 't_': t_[0,1,2], 'tw': tw[0,1] }
        after2 = { 'dt': dt2[0,1], 't_': t_2[0,1,2], 'tw': tw2[0,1] }

        f.close()
        f2.close()

        self.assertTrue(before['dt'] != after['dt'])
        self.assertTrue(before['t_'] != after['t_'])
        self.assertTrue(before['tw'] != after['tw'])
        self.assertTrue(before2['dt'] != after2['dt'])
        self.assertTrue(before2['t_'] != after2['t_'])
        self.assertTrue(before2['tw'] != after2['tw'])

        self.assertEqual(after['dt'], after2['dt'])
        self.assertEqual(after['t_'], after2['t_'])
        self.assertEqual(after['tw'], after2['tw'])



if __name__ == '__main__':
    unittest.main()
    
