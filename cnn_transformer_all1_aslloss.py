import os
import gc

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from core.PBCmodel_transformer_all1 import PlasticTransformer
from sklearn.metrics import roc_auc_score
from config.cfg_multi_label import opt

from tools.save_result import save_AUC, save_loss_plot, save_pred_result, create_logger
from core.trainer_multi_label_asl import Trainer

from sklearn.metrics import f1_score
from datetime import datetime

def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    
def train_model(batch_size, lr):
    epoch = opt.epoch
    method_list = opt.method_list
    kr_size_list = opt.kr_size_list
    method_name_list = opt.method_name_list
    trainer = Trainer()

    for mth, kr_size, mth_name in zip(method_list, kr_size_list, method_name_list):
        fold_count = 1
        print(f'method : {mth_name}')
        check_val_loss = 1000
        
        #make result path
        model_path = f'{RESULT_MAIN_PATH}CNN_ALL1_ASL/'
        method_path = f'{model_path}{mth_name}/'
        fold_path = f'{method_path}{fold_count}_Fold/'
        hyperparameter_path = f'{fold_path}{lr}_batchsize_{batch_size}/'
        auc_path = f'{hyperparameter_path}AUC/'
        loss_path = f'{hyperparameter_path}Loss/'
        log_path = f'{hyperparameter_path}Log/'
        save_model_path = f'{hyperparameter_path}Model/'
        pred_path = f'{hyperparameter_path}PredResult/'
        
        make_dir(RESULT_MAIN_PATH)
        make_dir(model_path)
        make_dir(method_path)
        make_dir(fold_path)
        make_dir(hyperparameter_path)
        make_dir(auc_path)
        make_dir(loss_path)
        make_dir(log_path)
        make_dir(save_model_path)
        make_dir(pred_path)

        #make dataloader and class weight
        train_dataloader, val_dataloader, test_dataloader, tr_weight, val_weight = trainer.set_data(fold_count, batch_size, mth)

        model = PlasticTransformer(kr_size).cuda()

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 2)
        
        loss_save_name = f'{fold_count}_Fold_{mth_name}_Loss.png'
        epoch_list = list()
        train_loss_list = list()
        val_loss_list = list()
        test_loss_list = list()
        
        val_check_count = 0
        for epoch_idx in range(1, epoch + 1):
            auc_save_name = f'Epoch_{epoch_idx}_{mth_name}_AUC.png'
            weighted_auc_save_name = f'Epoch_{epoch_idx}_{mth_name}_weighted_AUC.png'
            log_save_name = f'{fold_count}_Fold_{mth_name}.log'
            pred_file_name = f'{fold_count}_Fold_{mth_name}_result.csv'
            weighted_pred_file_name = f'{fold_count}_Fold_{mth_name}_weighted_result.csv'
            model_save_name = f'{fold_count}_Fold_{mth_name}.pth'

            epoch_list.append(epoch_idx)
            model.train()            
            model, train_loss = trainer.train_model(train_dataloader, model, optimizer, tr_weight, batch_size)
            
            model.eval()
            with torch.no_grad():
                model, val_loss = trainer.val_model(val_dataloader, model, val_weight, batch_size)
                model, test_pred_list, weight_pred_list, test_label_list = trainer.test_model(test_dataloader, model, te_weight, batch_size)
                
                scheduler.step(val_loss)
    
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            
            F1 = f1_score(test_pred_list, test_label_list, average = 'samples')
            wt_F1 = f1_score(weight_pred_list, test_label_list, average = 'samples')

            cur_lr = optimizer.param_groups[0]['lr']

            #make log file
            fileHandler, streamHandler, logger = create_logger(log_path + log_save_name)
            logger.info(f'{fold_count}Fold Epoch : {epoch_idx} | Train_loss : {train_loss:.4f} | Val_loss : {val_loss:.4f} | F1 : {F1:.4f} | wt_F1 : {wt_F1:.4f} | lr : {cur_lr}')
            fileHandler.close()
            logger.removeHandler(fileHandler)
            logger.removeHandler(streamHandler)
            
            #save prediction data as csv file
            save_pred_result(test_pred_list, test_label_list, pred_path + pred_file_name)

            if check_val_loss > val_loss:# and count_val_loss != 2:
                check_val_loss = val_loss
                torch.save(model.state_dict(), save_model_path + model_save_name)
                val_check_count = 0

            del test_pred_list, test_label_list, train_loss, val_loss
                
        #make loss plot 
        save_loss_plot(train_loss_list, val_loss_list, epoch_list, loss_path + loss_save_name)
        fold_count += 1
            
        del train_loss_list, val_loss_list, epoch_list
        gc.collect()
        torch.cuda.empty_cache()
        print()

def main():
    today = str(datetime.now())
    today = today[:10]
    RESULT_MAIN_PATH = f'result_4class/{today}/'
    
    batch_size_list = [16, 32, 64]
    lr_list = [1e-4, 1e-5, 1e-6]
    
    for bt_size in batch_size_list:
        for lr in lr_list:
            train_model(bt_size, lr)    
if __name__ == "__main__":
    main()
    