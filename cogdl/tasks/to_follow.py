import random,os,argparse
from collections import defaultdict

import pandas as pd
import copy
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models.keyedvectors import Vocab
from six import iteritems
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from tqdm import tqdm

from cogdl import options
from cogdl.datasets import build_dataset
from cogdl.models import build_model

import cogdl.tasks.link_prediction
from . import register_task
from .link_prediction import LinkPrediction

def generate_pairs(walks, vocab):
    pairs = []
    skip_window = 2
    for walk in walks:
        for i in range(len(walk)):
            for j in range(1, skip_window + 1):
                if i - j >= 0:
                    pairs.append((vocab[walk[i]].index, vocab[walk[i - j]].index))
                if i + j < len(walk):
                    pairs.append((vocab[walk[i]].index, vocab[walk[i + j]].index))
    return pairs


def generate_vocab(walks):
    index2word = []
    raw_vocab = defaultdict(int)

    for walk in walks:
        for word in walk:
            raw_vocab[word] += 1

    vocab = {}
    for word, v in iteritems(raw_vocab):
        vocab[word] = Vocab(count=v, index=len(index2word))
        index2word.append(word)

    index2word.sort(key=lambda word: vocab[word].count, reverse=True)
    for i, word in enumerate(index2word):
        vocab[word].index = i

    return vocab, index2word


def divide_data(input_list, division_rate):
    local_division = len(input_list) * np.cumsum(np.array(division_rate))
    random.shuffle(input_list)
    return [
        input_list[
            int(round(local_division[i - 1]))
            if i > 0
            else 0 : int(round(local_division[i]))
        ]
        for i in range(len(local_division))
    ]

def seperate_data(input_list,time_list):
    train_list=[]
    test_list=[]
    for i in range(len(time_list)):
        if time_list[i]==1:
            train_list.append(input_list[i])
        else:
            test_list.append(input_list[i])
    return [train_list,test_list]


def randomly_choose_false_edges(nodes, true_edges, num):
    true_edges_set = set(true_edges)
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
            if x != y and (x, y) not in true_edges_set and (y, x) not in true_edges_set:
                tmp_list.append((x, y))
                break
        if all_flag:
            break
    return tmp_list


def gen_node_pairs(train_data, test_data):
    G = nx.DiGraph()
    G.add_edges_from(train_data)
    # RWG = RWGraph(G)

    # base_walks = RWG.simulate_walks(20, 10)
    # vocab, index2word = generate_vocab(base_walks)
    # train_pairs = generate_pairs(base_walks, vocab)

    training_nodes = set(list(G.nodes()))
    #valid_true_data = []
    test_true_data = []
    #for u, v in valid_data:
    #    if u in training_nodes and v in training_nodes:
    #        valid_true_data.append((u, v))
    for u, v in test_data:
        if u in training_nodes and v in training_nodes:
            test_true_data.append((u, v))

    #valid_false_data = randomly_choose_false_edges(
    #    list(training_nodes), train_data, len(valid_data)
    #)
    test_false_data = randomly_choose_false_edges(
        list(training_nodes), train_data, len(test_data)
    )
    return (test_true_data, test_false_data)


def get_score(embs, node1, node2):
    # vector1 = embs[int(node1)].cpu().detach().numpy()
    # vector2 = embs[int(node2)].cpu().detach().numpy()
    vector1 = embs[int(node1)]
    vector2 = embs[int(node2)]
    return np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )


def evaluate(embs, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    for edge in true_edges:
        true_list.append(1)
        prediction_list.append(get_score(embs, edge[0], edge[1]))
    for edge in false_edges:
        true_list.append(0)
        prediction_list.append(get_score(embs, edge[0], edge[1]))

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-len(true_edges)]

    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), auc(rs, ps)

def get_description(G,id0,node):
    if node in following(G,id0) and node in follower(G,id0):
        flag="<->bot"
    elif node in follower(G,id0):
        flag="->bot"
    elif node in following(G,id0):
        flag="<-bot"
    elif is_follower_of(G,follower(G,id0),node):
        flag="-> x -> bot"
    elif is_following_of(G,following(G,id0),node):
        flag="<- x <-bot"
    elif is_follower_of(G,following(G,id0),node):
        flag="-> x <-bot"
    elif is_following_of(G,follower(G,id0),node):
        flag="<- x ->bot"
    else:
        flag='None'
    return flag

def following(G, node):
    result=list(G.successors(node))
    return result

def follower(G, node):
    result=list(G.predecessors(node))
    return result

def is_follower_of(G,a_set,target):
    lst = False
    for node in a_set:
        if target in follower(G,node):
            lst = True
            break
    return lst

def is_following_of(G,a_set,target):
    lst = False
    for node in a_set:
        if target in following(G,node):
            lst = True
            break
    return lst

def get_label(edge_list,edge_attr,bot,node):
    lbl=0
    following_state='None'
    follower_state="None"
    if (node,bot) in edge_list:
        idx=edge_list.index((node,bot))
        label=edge_attr[idx]
        lbl=1
        if label == 1:
            follower_state='existed follower'
        else:
            follower_state= "future follower"
    if (bot,node) in edge_list:
        idx=edge_list.index((bot,node))
        label=edge_attr[idx]
        if label==1:
            following_state="existed following"
        else:
            following_state="future following"
    result=[lbl,following_state,follower_state]
    return result


def get_label2(G_now,G_future,bot,node):
    lbl=0
    if node in follower(G_future,bot):
        lbl=1
    current_state=get_description(G_now,bot,node)
    future_state=get_description(G_future,bot,node)
    result=[lbl,current_state,future_state]
    return result

def get_num_follower_you_know(G,bot,node):
    a=follower(G,bot)
    b=following(G,node)
    return len(a)+len(b)-len(set(a+b))


@register_task("to_follow")
class ToFollow(LinkPrediction):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--hidden-size", type=int, default=128)
        parser.add_argument("--negative-ratio", type=int, default=5)
        #parser.add_argument("--ego-id", type=int, default=0)
        #parser.add_argument("--candidates", type=int, default=0)
        parser.add_argument('--infile', type=str, help='txt file, the first line is the id of bot(int), the second line lists the candidates for evaluation, separated by space', default='infile.txt')
        # fmt: on
    
    def __init__(self, args):
        super(ToFollow, self).__init__(args)

        dataset = build_dataset(args)
        data = dataset[0]
        self.data = data.cuda()
        if hasattr(dataset, 'num_features'):
            args.num_features = dataset.num_features
        model = build_model(args)
        self.model = model
        self.patience = args.patience
        self.max_epoch = args.max_epoch
        f=open(args.infile,'r')
        self.bot=int(f.readline())
        self.candidates=list(map(int,f.readline().split(' ')[:]))
        f.close()
        print('infile.txt has been read')

        edge_list = self.data.edge_index.cpu().numpy()
        edge_attr = self.data.edge_attr.cpu().numpy()[0]
        edge_list = list(zip(edge_list[0], edge_list[1]))
        self.pro_edge_list=edge_list
        self.pro_edge_attr=edge_attr
        
        #self.train_data, self.valid_data, self.test_data = divide_data(edge_list, [1, 0.0, 0.0])#change
        self.train_data,self.test_data=seperate_data(edge_list,edge_attr)
        #self.valid_data, self.test_data = gen_node_pairs(self.train_data, self.valid_data, self.test_data)
        self.test_data = gen_node_pairs(self.train_data, self.test_data)

    def train(self):
        G = nx.DiGraph()
        G.add_edges_from(self.train_data)
        G_future=nx.DiGraph()
        G_future.add_edges_from(self.train_data)
        G_future.add_edges_from(self.test_data[0])
        print('number of nodes(now):',G.number_of_nodes())
        print('number of edges(now):',G.number_of_edges())
        print('number of nodes(future):',G_future.number_of_nodes())
        print('number of edges(future):',G_future.number_of_edges())
        pwd = os.getcwd()
        pwd=os.path.join(pwd,'cogdl/data','twitter-dynamic-net','processed','embs.npy') 
        if os.path.exists(pwd):
            print('embs already exists')
            embs=np.load(pwd,allow_pickle=True)
            embs=embs.item()
        else:
            print('begin training for embs')
            embeddings = self.model.train(G)
        #np.save('embeddings.npy',embeddings)
            embs = dict()
            for vid, node in enumerate(G.nodes()):
                embs[node] = embeddings[vid]
            np.save(pwd,embs) 
        for node,emb in embs.items():
            if np.all(emb == 0):
                lis_emb=[embs[n] for n in G.predecessors(node)]
                embs[node]=np.mean(lis_emb,axis=0)

        scores=dict()
        true_label=dict()
        current_state=dict()
        future_state=dict()
        for node in self.candidates:
            scores[node]=get_score(embs, self.bot, node)
            true_label[node],current_state[node],future_state[node]=get_label2(G,G_future,self.bot,node)
        rank=pd.DataFrame([scores,true_label,current_state,future_state],index=['score','label','old_state','new_state'])
        
        rank=rank.T
        rank=rank.sort_values(by='score',ascending=False)
        # save result to files
        np.save(os.path.join(os.getcwd(),'cogdl/data','twitter-dynamic-net','processed','rank.npy'),rank)
        rank.to_csv(os.path.join(os.getcwd(),'cogdl/data','twitter-dynamic-net','processed','rank.csv'))
       
        roc_auc, f1_score, pr_auc = evaluate(embs, self.test_data[0], self.test_data[1])
        print(
            f"Test ROC-AUC = {roc_auc:.4f}, F1 = {f1_score:.4f}, PR-AUC = {pr_auc:.4f}"
        )
        accuracy=dict( ROC_AUC=roc_auc,PR_AUC=pr_auc,F1=f1_score)
        
        return [rank,accuracy]
        