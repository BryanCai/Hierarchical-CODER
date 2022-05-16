import csv
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
from cal_sim import similarity
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def load_all_rel(data_path):
    with open(data_path) as f:
        csvreader = csv.reader(f)
        listA = []
        listB = []
        label = []
        for idx, row in tqdm(enumerate(csvreader)):
            if idx == 0:
                first_row = row
                continue
            listA.append(row[5])
            listB.append(row[6])
            label.append(row[9])
    label = [1 if lab=='similarity' else 0 for lab in label]
    return listA, listB, label

def eval(model_name_or_path, tokenizer_name, data_path, out_path):
    if model_name_or_path[-4:] == '.pth':
        model = torch.load(model_name_or_path)
    else:
        model = AutoModel.from_pretrained(model_name_or_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    listA, listB, label = load_all_rel(data_path)
    print('start prediction')
    pred_sim = similarity(listA, listB, model, tokenizer)
    fpr, tpr, thresholds = roc_curve(label, pred_sim)
    auc_score = auc(fpr, tpr)
    print(model_name_or_path, 'AUC', auc_score)
    plt.plot(fpr, tpr)
    plt.savefig(out_path)
    return None

if __name__ == '__main__':
    eval(
        model_name_or_path='GanjinZero/coder_eng', 
        tokenizer_name='GanjinZero/coder_eng', 
        data_path='/media/sdb1/Zengsihang/Hier_CODER/data_processing/files/AllRelationPairs.csv',
        out_path='coder.png'   
    )


