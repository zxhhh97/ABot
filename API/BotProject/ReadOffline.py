'''
author:xiaohan 
update:19/10/21
'''

import os.path as osp
import os
from pprint import pprint
import time
import pandas as pd
from itertools import count,islice
from collections import defaultdict,Counter
import networkx as nx
#from gensim import corpora, models, similarities
#from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
import numpy as np
import random


class read_graph(object):
    def __init__(self,rootpath='./BotProject/pre_data',g_file='graph_cb.txt',usermap='user_map.txt',userlist='user_list.txt'):
        self.rootpath=rootpath       
        self.files={"graph":g_file,"name2id":usermap,"id2node":userlist}
        self.G_now,self.G_future,self.G=self.graphs()
        self.name2node=self.get_name2node()
        self.embs=self.load_embs()

    def __read_graph(self,path):
        with open(path,'r') as f:
            for line in f:
                if line:
                    yield list(map(int,line.strip('\n').split()))

    def graphs(self):
        #filename='graph_cb.txt'
        path=os.path.join(self.rootpath,self.files["graph"])
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
        print('edge number at time=1:',G_now.number_of_edges(),
        '\nedge number at time>1:',G_future.number_of_edges())
        return (G_now, G_future,G)

    def get_name2node(self):
        print(os.path.abspath('.'))
        file_name2id=os.path.join(self.rootpath,self.files['name2id'])
        file_id2node=os.path.join(self.rootpath,self.files['id2node'])
        id2node={}
        node2name={}
        name2node={}
        print('begin reading:',file_id2node)
        with open(file_id2node,'r') as f:
            i=0
            while True:
                line=f.readline()
                if not line:
                    break
                id2node[line.strip()]=i
                i+=1
        print('begin reading:',file_name2id)
        with open(file_name2id,'r') as f:
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

    #def query(self,df,value,src):
    #    return df.loc[df[src[0]]==value,src[1]].values[0]

    def gen_true_edges(self,dic):
        nodes=dic.keys() #content dic
        #nodes=self.df['node'].values
        G=self.G_future
        G2=self.G_now
        candidates=set(list(G2.nodes()))
        edges=list(G.edges())
        tmp_list=[]
        #all_flag = False
        for n1,n2 in edges:
            if n1 in nodes and n2 in nodes:
                if dic[n1] and dic[n2] and n1 in candidates and n2 in candidates and (n2,n1) not in tmp_list:
                    tmp_list.append((n1,n2))
        return tmp_list

    def gen_pure_edges(self,dic):
        nodes=dic.keys() #content dic
        G=self.G
        edges=list(G.edges())
        tmp_list=[]
        for n1,n2 in edges:
            if n1 in nodes and n2 in nodes:
                if dic[n1] and dic[n2]:
                    tmp_list.append((n1,n2))
        return tmp_list
    

    def randomly_choose_false_edges(self,dic,num=100):
        true_edges=self.G.edges()
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
                if x != y and (x, y) not in true_edges and (y, x) not in true_edges and not (x,y) in tmp_list and (y,x) not in tmp_list :
                    if x in self.embs and y in self.embs:
                        tmp_list.append((x, y))
                    break
            if all_flag:
                break
        return tmp_list
    
    def load_embs(self):
        pwd=os.path.join(self.rootpath,'embs.npy') 
        embs=np.load(pwd,allow_pickle=True)
        embs=embs.item()
        return embs

class read_tweet(object):
    def __init__(self,rootpath='./BotProject/pre_data',min_tf=2):
        self.path_table=osp.join(rootpath,"WordTable.txt")
        self.rootpath=rootpath
        self.min_tf=min_tf
        self.read_wordtable()
        
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
        
    def __iter_text2(self,path):
        with open(path, 'rb') as f:
            for line in f:
                yield str(line).lstrip("b'").rstrip("\\r\\n'").strip()
    ## called by dic_contents        
    def __iter_block(self,path):
        itr=self.__iter_text2(path)
        for text in itr:
            if not text == '':
                block=[text] #block[0]=name
                for attr in itr:
                    block.append(attr.strip().split(' '))
                    if len(block)>7 and attr == '': #reach the end of a tweet block
                        yield (block[0],block[6]) # name & content
                        break
   

    def dic_contents(self,name2node,num_files=None):
        node_content = defaultdict(list)
        filefolder=islice(self.__iter_txtfiles(),0,num_files)
        print('-----Read tweets .txt-----')
        start = time.perf_counter()
        for path in filefolder:
            print('Reading:',path)
            sample= islice(self.__iter_block(path),0,None)
            for k,v in sample:
                if k in name2node:
                    if v==['']:
                        v=[]
                    node_content[name2node[k]]+=v
            print('processing time:',time.perf_counter()-start)
        print('-----End of reading tweets-----')
        return node_content

    def __count_word(self,word_list,n=10):
        cont=Counter(word_list).most_common(n)
        # Replace name with node id 
        cont=[(self.id2newid[cnt[0]],cnt[1]) for cnt in cont if cnt[0] in self.id2tf]
        return cont
    