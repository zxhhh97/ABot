import os.path as osp
import os,sys
from pprint import pprint
import time
from itertools import count,islice
from collections import defaultdict,Counter
from gensim import corpora, models, similarities
from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
import logging
import numpy as np
import pandas as pd
from six import iteritems
import pandas as pd
import random 

class content_sim(object):
    def __init__(self,dic):
        self.nod2ind=self.node2index(dic)
        pass

    def node2index(self,dic):
        dic_n2i={}
        for i,k in enumerate(dic.keys()):
            dic_n2i[k]=i
        return dic_n2i

    def get_dictionary(self,texts):
        dictionary = corpora.Dictionary(texts)
        once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
        dictionary.filter_tokens(once_ids)  #
        dictionary.compactify() 
        dictionary.save('./sample.dict') 
        return dictionary

    
    def load_corpus(self,dic,CORPUS_PATH='./corpus.mm'):
        if not os.path.isfile(CORPUS_PATH):
            print('Creating corpus')
            corpus=list(dic.values())
            print('corpus created')
            corpora.MmCorpus.serialize(CORPUS_PATH, corpus)
        print('Loading corpus')
        #corpus = corpora.MmCorpus(CORPUS_PATH)
        return corpus


    def load_tfidf(self,corpus,TFIDF_PATH='./out.tfidf_model'):
        if not os.path.isfile(TFIDF_PATH):
            print('Creating TF-IDF')
            tfidf = models.TfidfModel(corpus)
            print('TF-IDF created')
            tfidf.save(TFIDF_PATH)
        print('Loading TF-IDF model')
        tfidf = models.TfidfModel.load(TFIDF_PATH)
        return tfidf

    def get_sims(self,id1,id2,index,corpus_md):
        a=self.nod2ind[id1]
        b=self.nod2ind[id2]
        return index[corpus_md[a]][b]

    def gen_true_edges(self,dic,G):
        keys_list=dic.keys()
        for n1,n2 in G.edges():
            if n1 in keys_list and n2 in keys_list:
                if dic[n1] and dic[n2]:
                    yield (n1,n2)


    def randomly_choose_false_edges(self,dic, true_edges, num):
        nodes=list(dic.keys())
        tmp_list = list()
        all_flag = False
        for _ in range(num):
            trial = 0
            while True:
                x = nodes[random.randint(0, len(nodes) - 1)]
                y = nodes[random.randint(0, len(nodes) - 1)]
                trial += 1
                if trial >= 1000:
                    all_flag = True
                    break
                if x != y and (x, y) not in true_edges and (y, x) not in true_edges:
                    tmp_list.append((x, y))
                    break
            if all_flag:
                break
        return tmp_list
