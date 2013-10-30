"""Classes and helper methods for corpus prep."""

import time
import random
from multiprocessing import Pool
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

def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)
   

class Word(tables.IsDescription):
    index = tables.Int64Col()
    string = tables.StringCol(40) 
    
class Corpus(object):
    def __init__(self, hdf5_path):
        """
        Args
            hdf5_path (str) - path to the HDF5 repository (will be created or
                overwritten).
        """
        self.num_D = 0
        self.num_W = 0
        
        self.hdf5_path = hdf5_path
        file = tables.openFile(hdf5_path, "w")
        group = file.createGroup("/", "g")
    
        vocabulary = file.createTable(group, 'vocabulary', Word)
        fuzzy_vocabulary = file.createTable(group, 'fuzzy_vocabulary', Word)
        
        file.flush()
        file.close()
    
        return
    
    def add_word(self, new_word, fuzzy=True):
        """Adds a new word to the Corpus vocabulary. Tries to minimize duplicate
        entries due to OCR errors. The candidate word, new_word, is evaluated
        using the following checks:
            * Is the word already in the vocabulary?
            * If not, is the word in WordNet? If yes: add the word.
            * If not, is the word similar to an existing word? If yes: map onto
                existing word.
            * Is the word in the NTLK stoplist? If so, don't add it.
            * Otherwise, add the word.
            
        The reason to check WordNet before looking for similar words is to
        minimize false-positives. E.g. so that "weight" and "eight" aren't
        collapsed into the same word.
        
        Args
            new_word (str) - string representation of word to add to vocabulary.
        
        Returns
            int. index of word in vocabulary.
            
        Notes
            TODO: Check against German and French word lists.
        """
        
        file = tables.openFile(self.hdf5_path, "a")
        vocabulary = file.root.g.vocabulary
        fuzzy_vocabulary = file.root.g.fuzzy_vocabulary
        
        new = False
        
        # Already in vocabulary?
        exists = new_word in [ x['string'] for x in vocabulary ]
        if not exists:
            synset = wn.synsets(new_word)   # Check WordNet.
            if len(synset) > 0 or new_word in wds.words():
                new = True
            else:   # Look for something very similar...
                    # TODO: Use a better similarity metric.
                matches = [ x['index'] for x in vocabulary if 0 < L.distance(x['string'], new_word) < 2 ]
                if len(matches) > 0:
                    word = fuzzy_vocabulary.row
                    word['string'] = new_word
                    word['index'] = matches[0]  # Use the first match.
                    word.append()
                    fuzzy_vocabulary.flush()
                
                    index = matches[0]
                else:
                    new = True
        else:
            index = self.word_index(new_word)
    
        if new_word in sw.words():  # Apply stoplist.
            index = False
        elif new:
            index = self.num_W
            word = vocabulary.row
            word['index'] = index
            word['string'] = new_word
            word.append()
            vocabulary.flush()            

            word = fuzzy_vocabulary.row
            word['index'] = index
            word['string'] = new_word
            word.append()
            fuzzy_vocabulary.flush()
            
            self.num_W += 1
        file.close()
    
        return index

    def word(self, index):
        """Returns the string representation of a word, given its index.
        
        Args
            index (int) - index of word.
        
        Returns
            string. string representation of word, or
            None. if no match is found.
        """
        
        file = tables.openFile(self.hdf5_path, "a")
        vocabulary = file.root.g.vocabulary        
        
        result = [ x['string'] for x in vocabulary.where("index == "+str(index))]
        file.close()
        
        try:
            return result[0]
        except IndexError:
            return None
        
    def word_index(self, string):
        """Returns the index of a word, given its string representation.
        
        Args
            string (str) - string representation of word.
        
        Returns
            int. first index of word matching string, or
            None. if no match is found.
        """
        file = tables.openFile(self.hdf5_path, "a")
        vocabulary = file.root.g.vocabulary        
        
        result = [ x['index'] for x in vocabulary.where("string == '"+str(string)+"'")]
        file.close()
        
        try:
            return result[0]
        except IndexError:
            return None

class DFRCorpus(Corpus):
    """Class for managing JSTOR Data-for-Research datasets."""
    
    def __init__(self, hdf5_path):
        self.files = []
        super(DFRCorpus, self).__init__(hdf5_path)
    
        return

    def add_file(self, path):
        """Adds a filepath to the Corpus documents list.
        
        Args
            path (str) - path to a text file.
        
        Returns
            None. if the file can be opened.
            False. if the file cannot be opened (e.g. if it doesn't exist).
        """

        try:
            with open(path):
                self.files.append(path)
            return None
        except IOError:
            return False

    def build(self, fuzzy=True, min_len=4):
        """Sequentially opens each file in self.documents, and adds any new
        words to the vocabulary table."""

        self.num_D = len(self.files)    # So that we can use the extendable
                                        #  dimension for words, not documents.
                                        # TODO: when PyTables supports multiple
                                        #  extendable dimensions, let the
                                        #  corpus grow along that axis.
        
        f = tables.openFile(self.hdf5_path, 'a')
        dw = f.createEArray("/g", "document_word", atom=tables.Float64Atom(), expectedrows=self.num_D, shape=(self.num_D, 0))
        
        for i in xrange(len(self.files)):
            with open(self.files[i], 'r') as file:
                root = ET.fromstring(file.read().replace("&", "&amp;"))

                for elem in root:
                    word = elem.text.strip(" ")
                    if len(word) >= min_len:
                        index = self.add_word(word, fuzzy)
                        if index >= dw.shape[1]:
                            dw.append(np.zeros([self.num_D, 1]))
                            dw.flush()
                            f.flush()
                        dw[i, index] = elem.attrib['weight']
        f.flush()
        f.close()

        return