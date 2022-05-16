import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm
import random
import string
import time
import pickle
import gc
import argparse

batch_size = 64
device = torch.device("cuda:0")

def get_bert_embed(phrase_list, m, tok, normalize=True, summary_method="CLS", tqdm_bar=True):
    m = m.to(device)
    input_ids = []
    for phrase in phrase_list:
        input_ids.append(tok.encode_plus(
            phrase, max_length=32, add_special_tokens=True,
            truncation=True, pad_to_max_length=True)['input_ids'])
        # print(len(input_ids))
    m.eval()

    count = len(input_ids)
    now_count = 0
    output_list = []
    with torch.no_grad():
        if tqdm_bar:
            pbar = tqdm(total=count)
        while now_count < count:
            input_gpu_0 = torch.LongTensor(input_ids[now_count:min(
                now_count + batch_size, count)]).to(device)
            if summary_method == "CLS":
                embed = m(input_gpu_0)[1]
            if summary_method == "MEAN":
                embed = torch.mean(m(input_gpu_0)[0], dim=1)
            if normalize:
                embed_norm = torch.norm(
                    embed, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                embed = embed / embed_norm
            if now_count % 1000000 == 0:
                if now_count != 0:
                    output_list.append(output.cpu().numpy())
                    del output
                    torch.cuda.empty_cache()
                output = embed
            else:
                output = torch.cat((output, embed), dim=0)
            if tqdm_bar:
                pbar.update(min(now_count + batch_size, count) - now_count)
            now_count = min(now_count + batch_size, count)
            del input_gpu_0
            torch.cuda.empty_cache()
        if tqdm_bar:
            pbar.close()
    output_list.append(output.cpu().numpy())
    del output
    torch.cuda.empty_cache()
    return np.concatenate(output_list, axis=0)

def embed_dense(names, encoder, tokenizer, show_progress=False, batch_size=2048, agg_mode="cls", normalize=True):
    """
    Embedding data into dense representations

    Parameters
    ----------
    names : np.array
        An array of names

    Returns
    -------
    dense_embeds : list
        A list of dense embeddings
    """
    encoder.eval() # prevent dropout
    
    batch_size=batch_size
    dense_embeds = []

    #print ("converting names to list...")
    #names = names.tolist()

    with torch.no_grad():
        if show_progress:
            iterations = tqdm(range(0, len(names), batch_size))
        else:
            iterations = range(0, len(names), batch_size)
            
        for start in iterations:
            end = min(start + batch_size, len(names))
            batch = names[start:end]
            batch_tokenized_names = tokenizer.batch_encode_plus(
                    batch, add_special_tokens=True, 
                    truncation=True, max_length=32, 
                    padding="max_length", return_tensors='pt')
            batch_tokenized_names_cuda = {}
            for k,v in batch_tokenized_names.items(): 
                batch_tokenized_names_cuda[k] = v.cuda()
            
            if agg_mode == "cls":
                batch_dense_embeds = encoder(**batch_tokenized_names_cuda)[0][:,0,:] # [CLS]
            elif agg_mode == "mean_pool":
                batch_dense_embeds = encoder(**batch_tokenized_names_cuda)[0].mean(1) # pooling
            else:
                print ("no such agg_mode:", agg_mode)
            if normalize:
                embed_norm = torch.norm(
                    batch_dense_embeds, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                batch_dense_embeds = batch_dense_embeds / embed_norm

            batch_dense_embeds = batch_dense_embeds.cpu().detach().numpy()

            dense_embeds.append(batch_dense_embeds)
    dense_embeds = np.concatenate(dense_embeds, axis=0)
    
    return dense_embeds


def similarity(listA, listB, model, tokenizer, mode='CLS'):
    embedA = get_bert_embed(listA, model, tokenizer, summary_method=mode)
    embedB = get_bert_embed(listB, model, tokenizer, summary_method=mode)
    return [np.dot(a, b) for a, b in zip(embedA, embedB)]

def similarity_sapbert(listA, listB, model, tokenizer, mode='CLS'):
    embedA = embed_dense(listA, model, tokenizer)
    embedB = embed_dense(listB, model, tokenizer)
    return [np.dot(a, b) for a, b in zip(embedA, embedB)]

if __name__ == '__main__':
    # listA = ['julibroside j2', 'magnetic resonance imaging of abdomen', 'orange colored urine', 'type 2 diabetes 1', 'sb 212047', 'early onset', 'ginsenoside rh', 'protein phosphatase 1 delta', 'ly 367385', 'type ii endometrial carcinoma']
    # listB = ['julibroside c1', 'x ray of abdomen', 'pink urine', 'type 1 diabetes', 'sb 216754', 'late onset', 'ginsenoside rg', 'protein phosphatase 2c delta', 'ly 367265', 'endometrial cancer stage ii']
    # listA = ['headache', 'srgap2a', 'fhx allergies', 'herpesvirus murid 004', 'tex2', 'eppin 1 protein, human', 'cdk8 protein, s pombe', 'chmp2b gene']
    # listB = ['cephalgia', 'fnbp2', 'fh: allergy', 'murine herpesvirus 068', 'tex2 gene', 'eppin protein, human', 'srb10 protein, s pombe', 'chromatin modifying protein 2b']
    
    # CODER and CODER++
    # config = AutoConfig.from_pretrained('GanjinZero/UMLSBert_ENG')
    # tokenizer = AutoTokenizer.from_pretrained('GanjinZero/UMLSBert_ENG')
    # coder = AutoModel.from_pretrained('GanjinZero/UMLSBert_ENG', config=config).to(device)
    # tokenizer = AutoTokenizer.from_pretrained('GanjinZero/coder_eng_pp')
    # coder = AutoModel.from_pretrained('GanjinZero/coder_eng_pp').to(device)

    # coder.eval()
    # coder.load_state_dict(torch.load('old_coder.pth'))
    # # torch.save(coder.state_dict(), 'old_coder.pth')
    # print(similarity(listA, listB, coder, tokenizer))

    # coder.load_state_dict(torch.load('old_coder.pth'))
    # print(similarity(listA, listB, coder, tokenizer))
   
    tokenizer = AutoTokenizer.from_pretrained('GanjinZero/coder_eng_pp')
    coderpp_online = AutoModel.from_pretrained('GanjinZero/coder_eng_pp').to(device)
    coderpp_online.eval()
    print(similarity(listA, listB, coderpp_online, tokenizer))   

    # # config = AutoConfig.from_pretrained('GanjinZero/coder_eng')
    # tokenizer = AutoTokenizer.from_pretrained('GanjinZero/coder_eng_pp', use_fast=False)
    # coderpp_online = AutoModel.from_pretrained('GanjinZero/coder_eng_pp', return_dict=False).to(device)
    # coderpp_online.eval()
    # print(similarity(listA, listB, coderpp_online, tokenizer))

    # print('coder result:')
    # print(similarity(listA, listB, coder, tokenizer))
    # coderpp = torch.load('/media/sdb1/Zengsihang/ShenZhen/CODER_faiss_finetune/train_whole_umls/output/Final_CODER.pth').to(device)
    # coderpp_online = AutoModel.from_pretrained('GanjinZero/coder_eng_pp').to(device)
    # tokenizer = AutoTokenizer.from_pretrained('GanjinZero/coder_eng_pp')
    # coderpp_online.eval()
    # # torch.save(coderpp.state_dict(), 'coderpp_offline.pth')
    # print('coder result:')
    # print(similarity(listA, listB, coderpp_online, tokenizer))

    # coderpp_online.load_state_dict(torch.load('coderpp_offline.pth'))
    # coderpp_online.eval()
    # print('coderpp offline result:')
    # print(similarity(listA, listB, coderpp_online, tokenizer))
   

    # sapbert
    # tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    # config = AutoConfig.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    # sapbert = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext", config=config).to(device)
    # print(similarity_sapbert(listA, listB, sapbert, tokenizer))
