import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve
from sklearn import metrics
from sklearn.utils.class_weight import compute_sample_weight

def save_AUC(label, pred, file_name, weight = False):

    fpr, tpr, _ = metrics.roc_curve(label,  pred, pos_label = 1, sample_weight = weight)
    auc = metrics.roc_auc_score(label, pred, sample_weight = weight)
            
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.xlabel('False Positive Rate (Positive label : 1)')
    plt.ylabel('True Positive Rate (Positive label : 1)')
    plt.title('ROC Curve')
    plt.legend(loc=4)
    plt.savefig(file_name)
    plt.close()
    
def sk_save_AUC(model, pred, label, file_name):
    plot_roc_curve(model, pred, label)
    plt.savefig(file_name)
    plt.close()
    
def save_loss_plot(train, val, epoch, file_name):

    plt.plot(epoch, train, label = 'trapn_loss')
    plt.plot(epoch, val, label = 'val_loss')
    
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc = 'upper right')
    plt.title('Train Val Test Loss')
    plt.savefig(file_name)
    plt.close()
    
def save_cof_matrix(conf_matrix, file_name):
    
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Answer', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig(file_name)
    plt.close()

def create_logger(file_path):
    import logging
    import logging.handlers

    logger = logging.getLogger(__name__)

    formatter = logging.Formatter('[%(asctime)s] - %(message)s')

    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(file_path)

    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    logger.setLevel(level=logging.DEBUG)

    return fileHandler, streamHandler, logger

def gene_save_pred_result(seq, pred, label, path):
    import pandas as pd
    
    csv_dict = {"seq" : seq, "prediction" : pred, "label" : label}
    df_data = pd.DataFrame(csv_dict)
    
    df_data.to_csv(path, index=False)
    
def save_pred_result(pred, label, path):
    import pandas as pd
    
    csv_dict = {"prediction" : pred, "label" : label}
    df_data = pd.DataFrame(csv_dict)
    
    df_data.to_csv(path, index=False)