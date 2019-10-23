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

sys.path.append(r'./BotProject')
from Readoffline import read_tweet
from ContentSim import content_sim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

rootpath='./BotProject/pre_data'  ##

tw=read_tweet(rootpath) 
G_now,G_future = tw.get_graph()
name2node=tw.name2node
#dic_content=tw.dic_content2tf(num_files=1,num_features=10)
dic_content2=tw.dic_contents_sample(num_files=10)
texts=list(dic_content2.values())

def node2ind(dic):
    node2index={}
    for i,k in enumerate(dic.keys()):
        node2index[k]=i
    return node2index

def get_dictionary():
    dictionary = corpora.Dictionary(texts)
    once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
    dictionary.filter_tokens(once_ids)  #
    dictionary.compactify() 
    dictionary.save('./sample.dict') 
    return dictionary

dictionary=get_dictionary()
n2i=node2ind(dic_content2)
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('./corpus.mm', corpus) 


## GET Corpus 

CORPUS_PATH='./corpus.mm'
def load_corpus(dic):
    if not os.path.isfile(CORPUS_PATH):
        print('Creating corpus')
        corpus=list(dic.values())
        print('corpus created')
        corpora.MmCorpus.serialize(CORPUS_PATH, corpus)
    print('Loading corpus')
    #corpus = corpora.MmCorpus(CORPUS_PATH)
    return corpus

TFIDF_PATH='./out.tfidf_model'
def load_tfidf(corpus):
    if not os.path.isfile(TFIDF_PATH):
        print('Creating TF-IDF')
        tfidf = models.TfidfModel(corpus)
        print('TF-IDF created')
        tfidf.save(TFIDF_PATH)
    print('Loading TF-IDF model')
    tfidf = models.TfidfModel.load(TFIDF_PATH)
    return tfidf

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=20)
#tfidf=load_tfidf(corpus)
#corpus_tfidf = tfidf[corpus]
corpus_lsi=lsi[corpus]

print('-----begin calculating index-----')
#index = similarities.Similarity(corpus_tfidf)
index = similarities.MatrixSimilarity(corpus_lsi)
index.save('./sample.index')
print('-------end------')

def sims(id1,id2,dic):
    a=n2i[id1]
    b=n2i[id2]
    return index[corpus_lsi[a]][b]

def gen_true_edges(dic):
    keys_list=dic.keys()
    for n1,n2 in G_future.edges():
        if n1 in keys_list and n2 in keys_list:
            if dic[n1] and dic[n2]:
                yield (n1,n2)


def randomly_choose_false_edges(dic, true_edges, num):
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

print('begin compare node pairs')
start=time.perf_counter()
edges=list(gen_true_edges(dic_content2))
false_edges=randomly_choose_false_edges(dic_content2, G_future.edges(), 100)

print('number of pairs:',len(edges),len(false_edges))
def sim_list(edge_list,dic,path):
    SIMS=np.zeros((len(edge_list),3))
    for i in range(len(edge_list)):
        n1,n2=edge_list[i]
        s=sims(n1,n2,dic)
        SIMS[i,:]=[n1,n2,s]
    result=pd.DataFrame(columns=['node1','node2','sim_content'],data=SIMS)
    result.to_csv(path)
    return SIMS

sim1=sim_list(edges,dic_content2,'./T_edges_con_sim.csv')
sim2=sim_list(false_edges,dic_content2,'./F_edges_con_sim.csv')
print(time.perf_counter()-start)