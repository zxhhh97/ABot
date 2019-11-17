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

from ReadOffline import read_tweet,read_graph
from ContentSim import content_sim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

rootpath='./BotProject/pre_data'  ## raw files
start=time.perf_counter()

## Reading files
tw=read_tweet(rootpath) 
rg=read_graph(rootpath)
G_now,G_future = rg.G_now,rg.G_future
name2node=rg.get_name2node()

def mk_midpath(midpath): 
    if not osp.isdir(midpath):
        os.makedirs(midpath)

midpath='./midfile'
path_corpdict='corpus_node2ind.npy'
mk_midpath(midpath)

def get_corpus_dictionary(path1='sample.dict',path2='corpus_node2ind.npy',path3='text.txt'):
    path1=osp.join(midpath,path1)
    path2=osp.join(midpath,path2)
    path3=osp.join(midpath,path3)
    if not(os.path.exists(path1) and  os.path.exists(path2) and os.path.exists(path3)):
        dic_content=tw.dic_contents(name2node,num_files=None) ## most time consuming
        texts=list(dic_content.values())
        file= open(path3, 'w')  
        for fp in texts:
            file.write(str(fp))
            file.write('\n')
        file.close()
        node2ind={}
        for i,k in enumerate(dic_content.keys()):
            node2ind[k]=i
        np.save(path2,node2ind)
        cs=content_sim(dic_content,rootpath)
        dictionary=cs.get_dictionary(texts,min_freq=1,savepath=path1)
    else:
        dictionary=corpora.Dictionary.load(path1)
        node2ind = np.load(path2,allow_pickle=True).item()
        with open(path3, 'r') as file:
            texts = file.readlines()
            for i,text in enumerate(texts):
                texts[i]=text.strip().strip("']|['").split("', '")
    return  (texts,node2ind,dictionary)
    ## GET Corpus 
def get_model_corpus(dictionary,path='./corpus.mm'):
    path1=osp.join(midpath,'corpus.mm')
    if not os.path.exists(path1):
        corpus = [dictionary.doc2bow(text) for text in texts]
        corpora.MmCorpus.serialize(path1, corpus) 
    else:
        print('loading corpus')
        corpus = corpora.MmCorpus(path1)
    path2=osp.join(midpath,'./corpus_lsi.model')
    if not os.path.exists(path2):
        lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=10)
        corpus_lsi=lsi[corpus]
        corpus_lsi.save(path2)
    else:
        corpus_lsi=models.LsiModel.load(path2)
    return (corpus,corpus_lsi)

def get_sim_index(corpus_lsi,path='./sample.index'):
    path=osp.join(midpath,path)
    if not os.path.exists(path):
        print('-----begin calculating index-----')
        index = similarities.MatrixSimilarity(corpus_lsi)
        index.save(path)
        print('-------end------')
    else:
        print('-----loading index-----')
        index=similarities.Similarity.load(path)
    print('-------end------')
    return index
    

texts,node2ind,dictionary=get_corpus_dictionary()
corpus,corpus_lsi=get_model_corpus(dictionary=dictionary)
index=get_sim_index(corpus_lsi)

print('begin compare node pairs')

true_edges=rg.gen_true_edges(node2ind)
false_edges=rg.randomly_choose_false_edges(node2ind, num=len(true_edges))
edges=true_edges+false_edges
#true_edges=rg.gen_pure_edges(node2ind)
#false_edges=rg.randomly_choose_false_edges(node2ind, num=len(true_edges))

print('number of pairs:',len(true_edges),len(false_edges))
cs=content_sim(node2ind,rootpath)
#dfsim=cs.sim_list(edges,osp.join(midpath,'./edges_sim.csv'),0,index,corpus_lsi)
dfsim1=cs.sim_list(true_edges,osp.join(midpath,'./T_edges_sim.csv'),1,index,corpus_lsi)
dfsim2=cs.sim_list(false_edges,osp.join(midpath,'./F_edges_sim.csv'),0,index,corpus_lsi)
dfsim=pd.concat([dfsim1,dfsim2],axis=0)
dfsim.to_csv(osp.join(midpath,'./edges_sim.csv'))
print(time.perf_counter()-start)