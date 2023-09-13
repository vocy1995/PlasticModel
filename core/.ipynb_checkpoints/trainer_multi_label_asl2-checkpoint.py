import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import math
import string
import numpy as np 

from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from dataset.PBCdataset_multi_label import PlasticDataset, get_csv_file, PlasticDatasetBRNN
from config.cfg_multi_label import opt

from .aslloss import AsymmetricLossOptimized

#PET, PU, PHA list
PET_PU_PHA_INDEX = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 32, 36, 37, 39, 40, 42, 43, 44, 48, 49, 50, 51, 59, 60, 61, 62, 63, 64, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 93, 94, 97, 99, 100, 101, 107, 111, 112, 113, 115, 116, 118, 119, 120, 121, 122, 123, 124, 125, 126, 129, 131, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179]

class Trainer():
    
    def train_model(self, loader, model, optimizer, class_weight, batch_size):
        train_loss = 0
        
        # multi label Loss
        # class_weight = torch.FloatTensor(class_weight)
        # class_weight = class_weight.cuda()
        # criterion = nn.MultiLabelSoftMarginLoss(weight = class_weight)
        
        #ASLLoss
        criterion = AsymmetricLossOptimized(
            gamma_neg=4, gamma_pos=1,
            clip=0,
            disable_torch_grad_focal_loss=True,
            eps=1e-5,
        )

        count = 0
        model.train()
        for tr_i, data in enumerate(loader):
            torch.cuda.empty_cache()
            seq, label = data
            seq = seq.cuda()
            label = label.cuda()

            
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss = train_loss / len(loader)

        return model, train_loss

    def val_model(self, loader, model, class_weight, batch_size):
        val_loss = 0
        
        # multi label Loss
        # class_weight = torch.FloatTensor(class_weight)
        # class_weight = class_weight.cuda()
        # criterion = nn.MultiLabelSoftMarginLoss(weight = class_weight)
        
        #ASLLoss
        criterion = AsymmetricLossOptimized(
            gamma_neg=4, gamma_pos=1,
            clip=0,
            disable_torch_grad_focal_loss=True,
            eps=1e-5,
        )
        model.eval()
        for va_i, data in enumerate(loader):
            torch.cuda.empty_cache()
            seq, label = data
            seq = seq.cuda()
            label = label.cuda()

            output = model(seq)
            
            loss = criterion(output, label)
            val_loss += loss.item()

        val_loss = val_loss / len(loader)

        return model, val_loss

    def test_model(self, loader, model, class_weight, batch_size):
        test_pred_list = list()
        test_label_list = list()
        weight_pred_list = list()
        sigmoid = nn.Sigmoid()
        model.eval()
        for te_i, data in enumerate(loader):
            if te_i not in PET_PU_PHA_INDEX:
                continue
                
            te_weight = np.array([class_weight])
            te_weight = torch.FloatTensor(te_weight)
            torch.cuda.empty_cache()
            seq, label = data

            seq = seq.cuda()
            label = label.cuda()

            pred = model(seq)

            sig_pred = sigmoid(pred)

            weighed_pred = pred * te_weight.cuda()
            weighed_sig_pred = sigmoid(weighed_pred)
            prediction = sig_pred.detach().cpu() >= torch.FloatTensor([0.5])
            weighed_sig_pred = weighed_sig_pred.detach().cpu() >= torch.FloatTensor([0.5])

            test_pred_list.extend(prediction.tolist())
            test_label_list.extend(label.tolist())
            weight_pred_list.extend(weighed_sig_pred.tolist())

        return model, test_pred_list, weight_pred_list, test_label_list
    
    def set_data(self, count, batch_size, mth):
        
        train_path = f'./data/multi_data/cnn_brnn/1_train_data.csv'
        val_path = f'./data/multi_data/cnn_brnn/1_val_data.csv'
        test_path = f'./data/multi_data/cnn_brnn/1_test_data.csv'
        
        train_seq, train_label = get_csv_file(train_path)
        val_seq, val_label = get_csv_file(val_path)
        test_seq, test_label = get_csv_file(test_path)
    
        traindataset = PlasticDataset(train_seq, train_label, mth)
        valdataset = PlasticDataset(val_seq, val_label, mth)
        testdataset = PlasticDataset(test_seq, test_label, mth)
        
        train_dataloader = DataLoader(traindataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 1, drop_last = True)
        val_dataloader = DataLoader(valdataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 1, drop_last = True)
        test_dataloader = DataLoader(testdataset, batch_size = 2, shuffle = True, pin_memory = True, num_workers = 1, drop_last = True)
        
        train_count = self.count_class_data(train_label)
        val_count = self.count_class_data(val_label)
        evl_count = self.count_class_data(test_label)

        tr_weight = compute_class_weight(class_weight = 'balanced', classes = np.unique(train_count), y = train_count)
        val_weight = compute_class_weight(class_weight = 'balanced', classes = np.unique(val_count), y = val_count)
 
        return train_dataloader, val_dataloader, test_dataloader, tr_weight, val_weight
    
    def set_data_brnn(self, count, batch_size, mth):

        train_path = f'./data/multi_data/cnn_brnn/1_train_data.csv'
        val_path = f'./data/multi_data/cnn_brnn/1_val_data.csv'
        test_path = f'./data/multi_data/cnn_brnn/1_test_data.csv'

        train_seq, train_label = get_csv_file(train_path)
        val_seq, val_label = get_csv_file(val_path)
        test_seq, test_label = get_csv_file(test_path)
        
        traindataset = PlasticDatasetBRNN(train_seq, train_label, mth)
        valdataset = PlasticDatasetBRNN(val_seq, val_label, mth)
        testdataset = PlasticDatasetBRNN(test_seq, test_label, mth)

        train_dataloader = DataLoader(traindataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 1)
        val_dataloader = DataLoader(valdataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 1)
        test_dataloader = DataLoader(testdataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 1)
        
        train_count = self.count_class_data(train_label)
        val_count = self.count_class_data(val_label)
        evl_count = self.count_class_data(test_label)

        tr_weight = compute_class_weight(class_weight = 'balanced', classes = np.unique(train_count), y = train_count)
        val_weight = compute_class_weight(class_weight = 'balanced', classes = np.unique(val_count), y = val_count)
        te_weight = compute_class_weight(class_weight = 'balanced', classes = np.unique(evl_count), y = evl_count)
        
        return train_dataloader, val_dataloader, test_dataloader, tr_weight, val_weight, te_weight
        
    def count_class_data(self, csv_data):
        class_data = ['NYLON', 'PBAT', 'PBS', 'PBSA', 'PCL', 'PE', 'PEA', 'PES', 'PET', 'PHA', 'PHB', 'PLA', 'PU', 'PVA', 'NEGATIVE']
        count_data = np.zeros(len(class_data), dtype=np.int64)
        for i in csv_data:
            output_string = i.translate(str.maketrans('', '', string.punctuation))
            output_string = output_string.split(" ")

            for i in output_string:
                count_data[int(i)] += 1
        
        label_count = 0
        label_list = list()
        for y in count_data:
            label_list.extend([label_count] * int(y))
            label_count += 1
            
        return label_list
    