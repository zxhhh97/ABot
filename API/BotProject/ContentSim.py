"""
by zxh
"""
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
import random 

class content_sim(object):
    def __init__(self,dic,rootpath,model='lsi'):
        self.rootpath=rootpath
        self.nod2ind=self.node2index(dic)
        self.model=self.get_model(model)

    def get_model(self,model):
        dic={'lsi':models.LsiModel,'tfidf':models.TfidfModel}
        return dic[model]

    def node2index(self,dic):
        dic_n2i={}
        for i,k in enumerate(dic.keys()):
            dic_n2i[k]=i
        return dic_n2i

    def get_dictionary(self,texts,min_freq=1,savepath=None):
        # todo: load dic
        dictionary = corpora.Dictionary(texts)
        once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == min_freq]
        dictionary.filter_tokens(once_ids)  #
        dictionary.compactify()
        if savepath:
            dictionary.save(savepath) 
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

    def get_sims(self,id1,id2,index,corpus_model):
        a=self.nod2ind[id1]
        b=self.nod2ind[id2]
        return index[corpus_model[a]][b]

    def sim_list(self,edge_list,path,label,index,corpus_model):
        SIMS=np.zeros((len(edge_list),5))
        embs=self.load_embs()
        for i,edge in enumerate(edge_list):
            n1,n2=edge[0],edge[1]
            s_cont=self.get_sims(n1,n2,index,corpus_model)
            s_embs=self.get_embsim(embs,n1,n2)
            SIMS[i,:]=[n1,n2,s_cont,s_embs,label]
        result=pd.DataFrame(columns=['node1','node2','sim_content','sim_embs','label'],data=SIMS)
        result.to_csv(path)
        return result

    def sim_list2(self,edge_list,dic,path,label,index,corpus_model):
        SIMS=np.zeros((len(edge_list),4))
        for i,edge in enumerate(edge_list):
            n1,n2=edge[0],edge[1]
            s_cont=self.get_sims(n1,n2,index,corpus_model)
            SIMS[i,:]=[n1,n2,s_cont,label]
        result=pd.DataFrame(columns=['node1','node2','sim_content','label'],data=SIMS)
        result.to_csv(path)
        return result

    def get_embsim(self,embs,node1,node2):
        vector1 = embs[int(node1)]
        vector2 = embs[int(node2)]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    def load_embs(self):
        pwd=os.path.join(self.rootpath,'embs.npy') 
        embs=np.load(pwd,allow_pickle=True)
        embs=embs.item()
        print('embs len:',len(embs))
        return embs