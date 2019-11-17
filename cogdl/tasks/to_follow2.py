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

def seperate_data(input_list,time_list,sep_point=3):
    train_list=[]
    test_list=[]
    for i in range(len(time_list)):
        if time_list[i]<=sep_point:
            train_list.append(input_list[i])
        else:
            test_list.append(input_list[i])
    return [train_list,test_list]


def gen_node_pairs(edgelist, time_stamp,time):
    dict_time={}
    for i,e in enumerate(edgelist):
        dict_time[tuple(e)]=time_stamp[i]
    cnt_T=[]
    cnt_F=[]
    for i,t in enumerate(time_stamp):
        u,v=(edgelist[i])
        stamp=dict_time.get((v,u),-2)
        if t<=time:
            if stamp>time:
                cnt_T.append((u,v))
            elif stamp<-1:
                cnt_F.append((u,v))
    cnt_F=random.sample(cnt_F,len(cnt_T))
    return (cnt_T,cnt_F)


def get_score(embs, node1, node2):
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


@register_task("to_follow2")
class ToFollow2(LinkPrediction):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--hidden-size", type=int, default=128)
        parser.add_argument("--negative-ratio", type=int, default=5)
        parser.add_argument("--sep_point", type=int, default=3)
    
    def __init__(self, args):
        super(ToFollow2, self).__init__(args)
        dataset = build_dataset(args)
        data = dataset[0]
        self.data = data.cuda()
        if hasattr(dataset, 'num_features'):
            args.num_features = dataset.num_features
        model = build_model(args)
        self.model = model
        self.patience = args.patience
        self.max_epoch = args.max_epoch
        self.sep_point=args.sep_point
        edge_list = self.data.edge_index.cpu().numpy()
        edge_attr = self.data.edge_attr.cpu().numpy()[0]
        edge_list = list(zip(edge_list[0], edge_list[1]))
        self.edge_list=edge_list
        self.edge_attr=edge_attr
        
        #self.train_data, self.valid_data, self.test_data = divide_data(edge_list, [1, 0.0, 0.0])#change
        self.train_data,self.test_data=seperate_data(self.edge_list,self.edge_attr,self.sep_point)
        #self.valid_data, self.test_data = gen_node_pairs(self.train_data, self.valid_data, self.test_data)
        self.test_data = gen_node_pairs(self.edge_list, self.edge_attr,self.sep_point)

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
        pwd=os.path.join(pwd,'cogdl/data','twitter-ltc','processed','embs'+str(self.sep_point)+'.npy') 
        print('begin training for embs')
        embeddings = self.model.train(G)
        embs = dict()
        for vid, node in enumerate(G.nodes()):
            embs[node] = embeddings[vid]
        np.save(pwd,embs) 
    
        roc_auc, f1_score, pr_auc = evaluate(embs, self.test_data[0], self.test_data[1])
        print(
            f"Test ROC-AUC = {roc_auc:.4f}, F1 = {f1_score:.4f}, PR-AUC = {pr_auc:.4f}"
        )
        return dict(
            ROC_AUC=roc_auc,
            PR_AUC=pr_auc,
            F1=f1_score,
        )
        
        