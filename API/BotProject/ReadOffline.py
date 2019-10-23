'''
author:xiaohan 
update:19/10/21
'''

import os.path as osp
import os
from pprint import pprint
import time
from itertools import count,islice
from collections import defaultdict,Counter
import networkx as nx
#from gensim import corpora, models, similarities
#from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
import numpy as np
import random
import networkx as nx

class read_tweet(object):
    def __init__(self,rootpath='./BotProject/pre_data',min_tf=2):
        filename = "Tweets-withoutwords/2010_10_14/tweet_result_0_.txt"
        self.path_sample_text=osp.join(rootpath,filename)
        self.path_table=osp.join(rootpath,"WordTable.txt")
        self.rootpath=rootpath
        #self.start=start
        #self.end=end
        self.min_tf=min_tf
        self.read_wordtable()
        self.name2node=self.get_name2node()
        
        
    def __iter_txtfiles(self,path="Tweets-withoutwords/"):
        rootpath=osp.join(self.rootpath,path)
        print("Reading text files in:",rootpath)
        for root, dirs, files in os.walk(rootpath, topdown=True):
            for fname in filter(lambda fname: fname.endswith('_.txt'), files):
                yield os.path.join(root, fname)
        
    # generator for reading text
    def __iter_text(self,path):
        with open(path, 'r') as f:
            for line in f:
                yield line.strip('\n')
    
    # called by read_wordtable
    # generator for reading wordtable
    def __iter_wordtable(self,min_tf=2):  
        itr=self.__iter_text(self.path_table)
        next(itr)
        for text in itr:
            ID,tf,word=text.split('\t')
            ID,tf=map(int,[ID,tf])
            if tf>min_tf and tf<10000:
                yield (ID,tf,word)
                
    # called by __init__         
    def read_wordtable(self):
        print('begin reading WordTable')
        min_tf=self.min_tf
        #dic={'id':ID,'tf':tf,'word':word}
        self.id2tf={}
        self.id2word={}
        self.word2tf={}
        self.id2newid={}
        for i,tmp in enumerate(self.__iter_wordtable(min_tf=min_tf)):
            self.id2tf[tmp[0]]=tmp[1]
            self.id2word[tmp[0]]=tmp[2]
            self.word2tf[tmp[2]]=tmp[1]
            self.id2newid[tmp[0]]=i
        print('num_words:',len(self.id2tf))
        print('get wordtable dic:id2tf, word2tf, id2word')
        
    ## called by dic_contents        
    def __iter_block(self,path):
        itr=self.__iter_text(path)
        for text in itr:
            if not text == '':
                block=[text] #block[0]=name
                for attr in itr:
                    block.append(attr.strip().split(' '))
                    if len(block)>7 and attr == '': #reach the end of a tweet block
                        yield (block[0],block[6]) # name & content
                        break
   
    def dic_contents(self,num_files=None):
        name_content = defaultdict(list)
        filefolder=islice(self.__iter_txtfiles(),0,num_files)
        print('-----Read tweets .txt-----')
        filefolder=[self.path_sample_text]
        start = time.perf_counter()
        for path in filefolder:
            print('Reading:',path)
            sample= islice(self.__iter_block(path),0,None)
            for k,v in sample:
                if k in self.name2node:
                    if v==['']:
                        v=[]
                    else:
                        v=[int(x) for x in v]
                    name_content[self.name2node[k]]+=v
                print('processing time:',time.perf_counter()-start)
        print('-----End of reading tweets-----')
        return name_content

    def dic_contents_sample(self,num_files=None):
        node_content = defaultdict(list)
        filefolder=islice(self.__iter_txtfiles(),0,num_files)
        print('-----Read tweets .txt-----')
        #filefolder=[self.path_sample_text]
        start = time.perf_counter()
        for path in filefolder:
            sample= islice(self.__iter_block(path),0,None)
            for k,v in sample:
                if k in self.name2node:
                    if v==['']:
                        v=[]
                    node_content[self.name2node[k]]+=v
            print('processing time:',time.perf_counter()-start)
        return node_content

    def __count_word(self,word_list,n=10):
        cont=Counter(word_list).most_common(n)
        # Replace name with node id 
        cont=[(self.id2newid[cnt[0]],cnt[1]) for cnt in cont if cnt[0] in self.id2tf]
        return cont
    
    def dic_content2tf(self,num_files=None,num_features=10):
        print('num_files=',num_files)
        name_ct=self.dic_contents(num_files=num_files)
        node_tf=defaultdict(list)
        for k,v in name_ct.items():
            node_tf[k]=self.__count_word(v,n=num_features)
        print('number of users:',len(node_tf))
        return node_tf
    
    def get_graph(self,filename='graph_cb.txt'):
        #filename='graph_cb.txt'
        path=os.path.join(self.rootpath,filename)
        G_now=nx.DiGraph()
        G_future=nx.DiGraph()
        G=nx.DiGraph()
        itr_graph = self.__read_graph(path)
        edge_label = np.array(list(itr_graph))
        G.add_edges_from(zip(edge_label[:,0],edge_label[:,1]))
        print('total edge number:',G.number_of_nodes())
        for n1,n2,t in edge_label:
            if t==1:
                G_now.add_edge(n1,n2)
            else:
                G_future.add_edge(n1,n2)  
        print('edge number at time=1:',G_now.number_of_edges(),'\nedge number at time>1:',G_future.number_of_edges())
        return (G_now, G_future)

    
    def __read_graph(self,path):
        with open(path,'r') as f:
            for line in f:
                if line:
                    yield list(map(int,line.strip('\n').split()))

    def get_name2node(self):
        file1='user_map.txt'
        file2='user_list.txt'
        print(os.path.abspath('.'))
        file1=os.path.join(self.rootpath,file1)
        file2=os.path.join(self.rootpath,file2)
        name2node={}
        id2node={}
        node2name={}
        with open(file2,'r') as f:
            i=0
            while True:
                line=f.readline()
                if not line:
                    break
                id2node[line.strip()]=i
                i+=1
        with open(file1,'r') as f:
            while True:
                line=f.readline()
                if not line:
                    break
                tmp=line.strip().split(' ')
                if tmp[0] in id2node:
                    name2node[tmp[1]]=id2node[tmp[0]]
                    node2name[id2node[tmp[0]]]=tmp[1]
                else:
                    name2node[tmp[1]]=None
        return name2node #{"name"='node_id'}