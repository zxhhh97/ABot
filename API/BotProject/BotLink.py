'''
author:xiaohan 
update:19/10/21
'''
import pandas as pd
import numpy as np
import warnings, time
from sklearn import metrics
import json,os
import networkx as nx


class bot_follow(object):
    def __init__(self):
        self.rootpath='./BotProject/pre_data'
        self.G,_,_=self.get_graph()
        print('begin nodes()')
        self.nodes=self.G.nodes()
        print('begin name2node')
        self.name2node,self.node2name=self.get_name2node()
        print('begin embs')
        self.embs=self.get_embs()
        print('get Nodes')
        self.Nodes=self.embs.keys()

    def to_jsonstr(self, json_data):
        json_obj = {}
        json_obj['data'] = json_data
        json_obj['code'] = 0
        json_obj['msg'] = 'ok'
        json_str = json.dumps(json_obj, ensure_ascii=False)
        return json_str

    def check_name(self,node,embs):
        user_map,_=self.get_name2node()  # return dict:{'aaa'=123
        #embs=self.get_embs()
        if  node in user_map.values():
            return True
        else:
            return False

    def get_score_dict(self,embs,user_map,bot_node,name_list):
        #user_map=self.get_name2node()  # return dict:{'aaa'=123
        scoredict={}
        recomdict={}
        for name in name_list:
            node=user_map[name]
            if self.check_name(node,embs):
                scoredict[name]=round(float(self.get_score(embs,bot_node,node)),4)
                print('name:',name,'node:',node)
            else:
                scoredict[name]=-1
                print('name:',name,'node: None')
        min_sco=max(0.5,np.median(list(scoredict.values())))
        for k,v in scoredict.items():
            recomdict[k]=0
            if v>min_sco:
                recomdict[k]=1
        return (scoredict,recomdict)

    def get_name2node(self):
        file1='user_map.txt'
        file2='user_list.txt'
        file1=os.path.join(self.rootpath,file1)
        file2=os.path.join(self.rootpath,file2)
        name2node={}
        node2name={}
        id2node={}
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
                    node2name[id2node[tmp[0]]]= tmp[1]
                else:
                    name2node[tmp[1]]=None
        return (name2node,node2name) #{"name"='node_id'}
        
    def get_embs(self,filename='embs.npy'):
        #filename='embs.npy'
        pwd=os.path.join(self.rootpath,filename)
        embs=np.load(pwd,allow_pickle=True)
        embs=embs.item()
        return embs

    def get_score(self,embs, node1, node2):
        vector1 = embs[int(node1)]
        vector2 = embs[int(node2)]
        return np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2))
    
    def get_label(self,G,bot,node):
        lbl = 0
        if node in list(G.predecessors(bot)):
            lbl=1

        result = lbl
        return result
    
    def get_graph(self,filename='graph_cb.txt'):
        #filename='graph_cb.txt'
        path=os.path.join(self.rootpath,filename)
        #G_now=nx.DiGraph()
        #G_future=nx.DiGraph()
        G=nx.DiGraph()
        itr_graph = self.__read_graph(path)
        edge_label = np.array(list(itr_graph))
        G.add_edges_from(zip(edge_label[:,0],edge_label[:,1]))
        print('total nodes number:',G.number_of_nodes())
        #for n1,n2,t in edge_label:
        #    if t==1:
        #        G_now.add_edge(n1,n2)
        #    else:
        #        G_future.add_edge(n1,n2)  
        #print('edge number at time=1:',G_now.number_of_edges(),'\nedge number at time>1:',G_future.number_of_edges())
        #return (G,G_now, G_future)
        return (G,0,0)

    
    def __read_graph(self,path):
        with open(path,'r') as f:
            for line in f:
                if line:
                    yield list(map(int,line.strip('\n').split()))
