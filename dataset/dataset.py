import numpy as np
from tqdm import tqdm
import cv2
import os
import pandas as pd

import torch
from torch.utils.data import Dataset

from utils.flow_utils import readFlowFile, calc_flow
from utils.image_utils import set_contrast, sharpen_img, add_noise

class array2tensor(object):
    """converts a numpy array to a torch tensor"""
        
    def __call__(self, array):
        
        ## numpy: H x W x C => torch: C x H x W
        if len(array.shape) > 3:
            array = array.transpose((0, 3, 1, 2)).astype(np.float32)
        else:
            array = array.transpose((2, 0, 1)).astype(np.float32)
        tensor = torch.from_numpy(array)
        return tensor


def tensor2array(tensor):
    """converts a torch tensor to a numpy array"""
        
    array = tensor.cpu().detach().numpy()
    
    ## torch: C x H x W => numpy: H x W x C
    if len(array.shape) > 3:
        array = array.transpose((0, 2, 3, 1)).astype(np.float32)
    else:
        array = array.transpose((1, 2, 0)).astype(np.float32)
    return torch.from_numpy(array)


def get_fold(k, protocol):
    train_idx  = proto[f'Fold {k}'] == 0
    train_fold = protocol[train_idx][['subject', 'file', 'emotion']].values
    test_fold  = protocol[~train_idx][['subject', 'file', 'emotion']].values
    return train_fold, test_fold
    
def train_test_folds(k, proto, X_inp_img, X_tar_img, X_flow, y):
    
    train_idx = proto[f'Fold {k}'] == 0

    X_inp_img_train = X_inp_img[train_idx]
    X_tar_img_train = X_tar_img[train_idx]
    X_flow_train = X_flow[train_idx]
    y_train = y[train_idx]
    
    X_inp_img_test = X_inp_img[~train_idx]
    X_tar_img_test = X_tar_img[~train_idx]
    X_flow_test = X_flow[~train_idx]
    y_test = y[~train_idx]
    
    train_fold = (X_inp_img_train, X_tar_img_train, X_flow_train, y_train)
    test_fold  = (X_inp_img_test, X_tar_img_test, X_flow_test, y_test)
    
    return train_fold, test_fold



def load_protocol(file_path, num_folds=10, sep=';'):
    columns = ['sub_id', 'seq_id', 'label'] + [f'Fold {i}' for i in range(1, num_folds+1)]
    protocol = pd.read_csv(file_path, sep=sep, names=columns)
    protocol.columns = columns
    protocol['seq_id'] = protocol['seq_id'].apply(lambda x: str(x).rjust(3, '0'))
    return protocol


def load_emotions_file(file_path, sep=';'):
    emotions_df = pd.read_csv(file_path, sep=sep, names=['sub_id', 'seq_id', 'label'], index_col=False)
    emotions_df['seq_id'] = emotions_df['seq_id'].apply(lambda x: str(x).rjust(3, '0'))
    return emotions_df

    
class eMotionGANDataset(Dataset):
    """
    Main dataset for training & validating eMotionGAN.

    Args:
        data_file_path: path to data_file.
        flow_args: parameters to calculate optical flow.
        folds: training/testing folds.

    """
    def __init__(self, data_file_path, folds, flow_args):
        
        self.data_file_path = data_file_path
        self.folds = folds
        self.flow_args = flow_args
        self.files = self.load_files_paths()
        

    def load_files_paths(self):
        
        paths = []        
        with open(self.data_file_path, 'r') as f:
            files = f.readlines()
            
        for seq in tqdm(self.folds, desc=f"loading files..... "):
            seq = seq.astype('str')
            for file in files:
                file = file.strip()
                dataset_0, sub, pose, emotion, algo, *_ = file.split(',') 
                if dataset_0 != dataset:
                    continue
                seq_0 = np.asarray([sub, pose, emotion])
                if all(seq_0 == seq):
                    if algo == self.flow_params['algo']:
                            paths.append(file)

        return paths
    
  
    def __len__(self):
        return len(self.files)
  

    def __getitem__(self, index):
        
        file = self.files[index]
        dataset, sub, pose, emotion, flow_type, inp_img_path, tar_img_path, _, _, step, inp_flow_path, _, tar_flow_path = file.split(',') 
        
        inp_flow = readFlowFile(inp_flow_path)
        tar_flow = readFlowFile(tar_flow_path)

        inp_img = cv2.imread(inp_img_path, 0)
        tar_img = cv2.imread(tar_img_path, 0)
        inp_img = cv2.cvtColor(inp_img, cv2.COLOR_GRAY2RGB)
        tar_img = cv2.cvtColor(tar_img, cv2.COLOR_GRAY2RGB)
        
        if np.random.rand() > 0.9:
            inp_img = set_contrast(inp_img)
            tar_img = set_contrast(tar_img) 
            
        if np.random.rand() > 0.9:
            inp_img = sharpen_img(inp_img)
            tar_img = sharpen_img(tar_img)
                            
        ## add some noise to images
        if np.random.rand() > 0.9:
            inp_img = add_noise(inp_img)
            tar_img = add_noise(tar_img)
        
        ## scale
        inp_img = inp_img / 255.0
        tar_img = tar_img / 255.0
        
        ## clipping
        if self.flow_args['max_mag']:
            inp_flow[..., 0][inp_flow[..., 0] > self.flow_args['max_mag']] = self.flow_args['max_mag']
            tar_flow[..., 0][tar_flow[..., 0] > self.flow_args['max_mag']] = self.flow_args['max_mag']
            
        ## normalize flow
        if self.flow_args['max_mag']:
            inp_flow[..., 0] = inp_flow[..., 0] / self.flow_args['max_mag']
            tar_flow[..., 0] = tar_flow[..., 0] / self.flow_args['max_mag']
        else:
            inp_flow[..., 0] = inp_flow[..., 0] / inp_flow[..., 0].max()
            tar_flow[..., 0] = tar_flow[..., 0] / tar_flow[..., 0].max()
        inp_flow[..., 1] = inp_flow[..., 1] / (2 * np.pi)
        tar_flow[..., 1] = tar_flow[..., 1] / (2 * np.pi)
            
        ## remove noise
        if self.flow_args['min_mag']:
            tar_flow[tar_flow[..., 0] < self.flow_args['min_mag']] = 0.0
            
        ## resize
        if self.flow_args['shape']:
            inp_flow = cv2.resize(inp_flow, self.flow_args['shape'], interpolation=cv2.INTER_LINEAR)
            tar_flow = cv2.resize(tar_flow, self.flow_args['shape'], interpolation=cv2.INTER_LINEAR)
            inp_img  = cv2.resize(inp_img, self.flow_args['shape'], interpolation=cv2.INTER_LINEAR)
            tar_img  = cv2.resize(tar_img, self.flow_args['shape'], interpolation=cv2.INTER_LINEAR)
            
        ## scale to [-1, 1]
        inp_flow = (-2. * inp_flow + 1.)
        tar_flow = (-2. * tar_flow + 1.)
        inp_img = (-2. * inp_img + 1.)
        tar_img = (-2. * tar_img + 1.)
        
        ## to tensors
        inp_img  = array2tensor()(inp_img)
        tar_img  = array2tensor()(tar_img)
        inp_flow = array2tensor()(inp_flow)
        tar_flow = array2tensor()(tar_flow)
        label = torch.tensor(int(emotion), dtype=torch.float32)

        napex = 3 < int(step) <= 5
        
        return inp_img, tar_img, inp_flow, tar_flow, label, napex




def load_flow_img_dataset(data_path, emotions, flow_algo, max_mag=None, min_mag=None, normalize=False, shape=None, start=0):
    
    """to load optical flow & images dataset"""

    X_flow, X_inp_img, X_tar_img, labels = [], [], [], []

    sep = os.path.sep
    fname = data_path.split(sep)[-1]

    for e, (sub_id, seq_id, label) in enumerate(tqdm(emotions.values, desc=f'load {fname :.<20}', colour='green')):
        
        path = sep.join([data_path, sub_id, seq_id, ''])
        files = [f for f in sorted(os.listdir(path))]
        
        neutral = cv2.imread(files[0], 0)
        apex = cv2.imread(files[-1], 0)
        flow = calc_flow(neutral, apex, flow_algo)
        flow = flow_to_polar(flow, max_mag=max_mag, min_mag=min_mag, normalize=normalize, shape=shape)

        neutral = cv2.resize(neutral, shape, interpolation=cv2.INTER_LINEAR)
        neutral = cv2.cvtColor(neutral, cv2.COLOR_GRAY2RGB) / 255.0
        apex = cv2.resize(apex, shape, interpolation=cv2.INTER_LINEAR)
        apex = cv2.cvtColor(apex, cv2.COLOR_GRAY2RGB) / 255.0

        X_flow.append(-2.0*flow+1.0)
        X_inp_img.append(-2.0*neutral+1.0)
        X_tar_img.append(-2.0*apex+1.0)
        labels.append(label)

    X_flow = np.array(X_flow)
    X_flow = array2tensor()(X_flow)
    
    X_tar_img = np.array(X_tar_img)
    X_tar_img = array2tensor()(X_tar_img)
    
    X_inp_img = np.array(X_inp_img)
    X_inp_img = array2tensor()(X_inp_img)
    
    return X_inp_img, X_tar_img, X_flow, torch.tensor(labels, dtype=torch.float32)