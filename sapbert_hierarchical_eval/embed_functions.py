import torch
import numpy as np


def get_bert_embed(phrase_list, model, tokenizer, device, show_progress=False, batch_size = 64, summary_method="CLS", normalize=True):
    model = model.to(device)
    input_ids = []
    for phrase in phrase_list:
        input_ids.append(tokenizer.encode_plus(
            phrase, max_length=32, add_special_tokens=True,
            truncation=True, padding="max_length")['input_ids'])
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
            if show_progress:
                pbar.update(min(now_count + batch_size, count) - now_count)
            now_count = min(now_count + batch_size, count)
            del input_gpu_0
        if show_progress:
            pbar.close()
    output_list.append(output.cpu().numpy())
    del output
    return np.concatenate(output_list, axis=0)

def get_biogpt_embed(phrase_list, model, tokenizer, device, show_progress=False, batch_size = 64, summary_method="CLS", normalize=True):
    model = model.to(device)
    input_ids = []
    for phrase in phrase_list:
        input_ids.append(tokenizer.encode_plus(
            phrase, max_length=32, add_special_tokens=True,
            truncation=True, padding="max_length")['input_ids'])
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
                embed = model(input_gpu_0, output_hidden_states=True).hidden_states[-1][:,-1,:]
            if summary_method == "MEAN":
                embed = torch.mean(model(input_gpu_0, output_hidden_states=True).hidden_states[-1], dim=1)
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
            if show_progress:
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

def get_distilbert_embed(phrase_list, model, tokenizer, device, show_progress=False, batch_size = 64, summary_method="CLS", normalize=True):
    model = model.to(device)
    input_ids = []
    for phrase in phrase_list:
        input_ids.append(tokenizer.encode_plus(
            phrase, max_length=32, add_special_tokens=True,
            truncation=True, padding="max_length")['input_ids'])
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
                embed = model(input_gpu_0).last_hidden_state[:,0,:]
            if summary_method == "MEAN":
                embed = torch.mean(model(input_gpu_0).last_hidden_state, dim=1)
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
            if show_progress:
                pbar.update(min(now_count + batch_size, count) - now_count)
            now_count = min(now_count + batch_size, count)
            del input_gpu_0
        if show_progress:
            pbar.close()
    output_list.append(output.cpu().numpy())
    del output
    return np.concatenate(output_list, axis=0)

def get_wrapper_embed(phrase_list, model, tokenizer, device, show_progress=False, batch_size=64, summary_method="CLS", normalize=True):
    out =  model.embed_dense(phrase_list, batch_size=batch_size, agg_mode=summary_method.lower())
    if normalize:
        out /= (np.sqrt((out**2).sum(-1))[...,np.newaxis] + 1e-16)
    return out


def get_truncated_embed_fun(embed_fun, k):
    def f(*args, **kwargs):
        out = embed_fun(*args, **kwargs)[:, :k]
        out /= (np.sqrt((out**2).sum(-1))[...,np.newaxis] + 1e-16)
        return out
    return f

