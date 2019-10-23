import sys
import os
import os.path as osp
from itertools import repeat
import numpy as np
import scipy.sparse as sp
import torch
from torch_sparse import coalesce

import cogdl.transforms as T
from cogdl.data import Data, Dataset, download_url, extract_gz, extract_rar

from . import register_dataset


class twitter(Dataset):
    r"""networks from http://arnetminer.org/lab-datasets/ twitter datasets
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"DynamicNet"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`cogdl.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`cogdl.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    # url =https://snap.stanford.edu/data/higgs-social_network.edgelist.gz
    

    def __init__(self, 
                 root, 
                 name,
                 url,
                 transform=None, 
                 re_transform=None, 
                 pre_transform=None):
        self.name = name
        self.url = url
        super(twitter, self).__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        #return ['higgs-social_network.edgelist.gz']
        return ['graph_cb.txt']

    @property
    def processed_file_names(self):
        return "data.pt"

    def get(self, idx):
        assert idx == 0
        return self.data

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_rar(path, self.raw_dir)
        os.unlink(path)

    def read_txt_label(self,path, start=0, num=3,end=None, dtype=None):
        with open(path,'r') as f:
            src = f.read().split('\n')[:-1]
            print('edge number: ', len(src))
            result = np.zeros((num, len(src)))
            for i, line in enumerate(src):
                result[:, i] = list(map(int, line.strip().split(' ')[start:end]))
        result = torch.from_numpy(result).to(dtype)
        return result

    def process(self):
        edge=self.read_txt_label(osp.join(self.raw_dir, '{}.txt'.format(self.name)),dtype=torch.int)
        edge_index=edge[:-1,:]
        edge_attr=edge[-1:,:]
        data = Data(edge_index=edge_index,edge_attr=edge_attr, x=None, y=None)
        #data = Data(edge_index=edge_index, x=None, y=None)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(data, self.processed_paths[0])


@register_dataset('dynamicnet')
class DynamicNet(twitter):
    def __init__(self):
        url = 'http://arnetminer.org/lab-datasets/tweet/twitter_network.rar'
        dataset, filename = 'twitter-dynamic-net','graph_cb'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
        super(DynamicNet, self).__init__(path, filename, url)
        