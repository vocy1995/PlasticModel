import gc
import torch
import string
import numpy as np
import pandas as pd

from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

LABEL_LIST = ['NYLON', 'PBAT', 'PBS', 'PBSA', 'PCL', 'PE', 'PEA', 'PES', 'PET', 'PHA', 'PHB', 'PLA', 'PU', 'NEGATIVE']

def get_csv_file(path):
    
    read_data = pd.read_csv(path)
    
    seq_data = read_data['seq']
    labal_data = read_data['label']
    
    del read_data
    gc.collect()
    
    return seq_data, labal_data

class PlasticDataset:
    def __init__(self, seq, label, encoding):
        self.seq = seq
        self.label = label
        self.encoding = encoding
        
    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, idx):
        seq = self.encoding(self.seq[idx])
        label = torch.FloatTensor(make_multi_label(self.label[idx]))
        
        seq = np.array([seq])
        seq = torch.FloatTensor(seq)
        
        return seq, label

    
class PlasticDatasetBRNN:
    def __init__(self, seq, label, encoding):
        self.seq = seq
        self.label = label
        self.encoding = encoding
        
    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, idx):
        label = torch.FloatTensor(make_multi_label(self.label[idx]))
        seq = self.encoding(self.seq[idx])   
        seq = torch.FloatTensor(seq)
        return seq, label
    
def make_multi_label(data):
    data = data.translate(str.maketrans('', '', string.punctuation))
    data = data.split(' ')
    label_np = np.zeros(len((LABEL_LIST)))
    
    for i in range(len(data)):
        label_np[int(data[i])] = 1
    return label_np
        