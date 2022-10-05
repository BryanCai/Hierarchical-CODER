import argparse
import pandas as pd
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from scipy.stats import spearmanr


def get_bert_embed(phrase_list, model, tokenizer, device, show_progress=False, batch_size = 64, summary_method="CLS", normalize=True):
    model = model.to(device)
    input_ids = []
    for phrase in phrase_list:
        input_ids.append(tokenizer.encode_plus(
            phrase, max_length=32, add_special_tokens=True,
            truncation=True, pad_to_max_length=True)['input_ids'])
        # print(len(input_ids))
    model.eval()

    count = len(input_ids)
    now_count = 0
    output_list = []
    with torch.no_grad():
        if show_progress:
            pbar = tqdm(total=count)
        while now_count < count:
            input_gpu_0 = torch.LongTensor(input_ids[now_count:min(
                now_count + batch_size, count)]).to(device)
            if summary_method == "CLS":
                embed = model(input_gpu_0)[1]
            if summary_method == "MEAN":
                embed = torch.mean(model(input_gpu_0)[0], dim=1)
            if normalize:
                embed_norm = torch.norm(
                    embed, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                embed = embed / embed_norm
            if now_count % 1000000 == 0:
                if now_count != 0:
                    output_list.append(output.cpu().numpy())
                    del output
                output = embed
            else:
                output = torch.cat((output, embed), dim=0)
            if tqdm_bar:
                pbar.update(min(now_count + batch_size, count) - now_count)
            now_count = min(now_count + batch_size, count)
            del input_gpu_0
        if show_progress:
            pbar.close()
    output_list.append(output.cpu().numpy())
    del output
    return np.concatenate(output_list, axis=0)

def get_sapbert_embed(phrase_list, model, tokenizer, device, show_progress=False, batch_size=2048, summary_method="CLS", normalize=True):
    model = model.to(device)
    model.eval()
    
    batch_size=batch_size
    dense_embeds = []

    with torch.no_grad():
        if show_progress:
            iterations = tqdm(range(0, len(phrase_list), batch_size))
        else:
            iterations = range(0, len(phrase_list), batch_size)
            
        for start in iterations:
            end = min(start + batch_size, len(phrase_list))
            batch = phrase_list[start:end]
            batch_tokenized_names = tokenizer.batch_encode_plus(
                    batch, add_special_tokens=True, 
                    truncation=True, max_length=32, 
                    padding="max_length", return_tensors='pt')
            batch_tokenized_names_cuda = {}
            for k,v in batch_tokenized_names.items(): 
                batch_tokenized_names_cuda[k] = v.cuda()
            
            if summary_method == "CLS":
                batch_dense_embeds = model(**batch_tokenized_names_cuda)[0][:,0,:] # [CLS]
            elif summary_method == "MEAN":
                batch_dense_embeds = model(**batch_tokenized_names_cuda)[0].mean(1) # pooling
            else:
                print ("no such summary_method:", summary_method)
            if normalize:
                embed_norm = torch.norm(
                    batch_dense_embeds, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                batch_dense_embeds = batch_dense_embeds / embed_norm

            batch_dense_embeds = batch_dense_embeds.cpu().detach().numpy()

            dense_embeds.append(batch_dense_embeds)
    dense_embeds = np.concatenate(dense_embeds, axis=0)
    
    return dense_embeds



def load_model(model_name_or_path, device):
    if model_name_or_path[-8:] == 'bert.pth':
        model = torch.load(model_name_or_path, map_location=torch.device(device))
    elif model_name_or_path[-4:] == '.pth':
        model = torch.load(model_name_or_path, map_location=torch.device(device)).bert
        torch.save(model, model_name_or_path[:-4] + '_bert.pth')
    else:
        model = AutoModel.from_pretrained(model_name_or_path)
    return model



def read_rank_csv(path):
    a = pd.read_csv(path)
    d = {"score": a.Mean.tolist(),
    "string1": a.Eng_Term1.tolist(),
    "string2": a.Eng_Term2.tolist()
    }
    return d



def run(args):
    data_dir = Path(args.data_dir)
    model = load_model(args.model_name_or_path, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.model_name_or_path.find('SapBERT') > 0:
        embed_fun = get_sapbert_embed
    else:
        embed_fun = get_bert_embed


    output = {}
    for f in ["Similar.csv", "Relate.csv"]:
        data = read_rank_csv(data_dir/f)
        embed1 = embed_fun(data["string1"], model, tokenizer, args.device, normalize=True)
        embed2 = embed_fun(data["string2"], model, tokenizer, args.device, normalize=True)
        cos_sim = [np.dot(a, b) for a, b in zip(embedA, embedB)]

        output[f] = spearmanr(cos_sim, data["score"])[0]
        print(f, output[f])









if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="./data",
        type=str,
        help="eval data directory",
    )

    parser.add_argument(
        "--model_name_or_path",
        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        type=str,
        help="model path",
    )

    parser.add_argument(
        "--tokenizer",
        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        type=str,
        help="tokenizer path",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="tokenizer path",
    )


    args = parser.parse_args()
    run(args)
