import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertTokenizer, BertModel
from tqdm import tqdm
import random
import string
import time
import pickle
import gc
from pathlib import Path
import pandas as pd
import csv

batch_size = 64
device = torch.device("cuda:0")

def get_bert_embed(phrase_list, m, tok, normalize=True, summary_method="CLS", tqdm_bar=False):
    '''
    This function is used to generate embedding vectors for phrases in phrase_list
    
    param:
        phrase_list: list of phrases to be embeded
        m: model
        tok: tokenizer
        normalize: normalize the embeddings or not
        summary_method: method for generating embeddings from bert output, CLS for class token or MEAN for mean pooling
        tqdm_bar: progress bar

    return:
        embeddings in numpy array with shape (phrase_list_length, embedding_dim)
    '''
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


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def get_embeds(phrase_dict, output_dir, model, tokenizer):
    embeds_list = []
    for phrase_file, phrase_list in phrase_dict.items():
        output_path = output_dir/(phrase_file.split(".")[0] + "_embeds.csv")
        print("writing to {}".format(output_path))
        if os.path.exists(output_path):
          os.remove(output_path)

        for sub_phrase_list in tqdm(chunker(phrase_list, 10000), total=len(phrase_list)//10000):    
            embeds = get_bert_embed(sub_phrase_list, model, tokenizer, summary_method="MEAN", tqdm_bar=False)
            embeds = pd.DataFrame(embeds)

            assert len(sub_phrase_list) == embeds.shape[0]
            embeds.insert(0, "source", phrase_file)
            embeds.insert(1, "input", sub_phrase_list)

            embeds.to_csv(output_path, sep="|", mode="a", index=False, header=False)

def get_sentences(phrase_dir):
    phrase_dir = Path(phrase_dir)
    phrase_file_list = [x.name for x in list(phrase_dir.glob("*.txt"))]

    phrase_dict = {}

    for phrase_file in phrase_file_list:
        with open(phrase_dir/phrase_file, encoding="ISO-8859-1") as f:
            phrase_list = f.readlines()
            phrase_list = [phrase.strip() for phrase in phrase_list]

        phrase_dict[phrase_file] = phrase_list

    return phrase_dict


if __name__ == "__main__":
    coder_filename = "GanjinZero/coder_eng"
    coder_config = AutoConfig.from_pretrained(coder_filename)
    coder_tokenizer = AutoTokenizer.from_pretrained(coder_filename)
    coder_model = AutoModel.from_pretrained(
        coder_filename,
        config=coder_config).to(device)


    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

    phrase_dict = get_sentences(Path("/n/data1/hsph/biostat/celehs/lab/sm731/CUI-search/embeds/to_embed"))
    output_dir = Path("/home/tc24/BryanWork/CODER/data/embed_output")

    get_embeds(phrase_dict, output_dir/"coder_embeds", coder_model, coder_tokenizer)
    get_embeds(phrase_dict, output_dir/"bert_embeds", bert_model, bert_tokenizer)
