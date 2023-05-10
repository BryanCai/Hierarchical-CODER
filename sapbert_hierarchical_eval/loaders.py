import torch
from transformers import AutoModel, AutoTokenizer, BioGptTokenizer, BioGptForCausalLM

def load_model_and_tokenizer_bert(model_name_or_path, device):
    m = torch.load(model_name_or_path, map_location=torch.device(device))
    model = m.bert
    tokenizer = m.tokenizer
    return model, tokenizer


def load_model_and_tokenizer_biogpt(model_name_or_path, device):
    model = BioGptForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = BioGptTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer


def load_model_and_tokenizer_SapBERT(model_name_or_path, tokenizer, device):
    model = AutoModel.from_pretrained(model_name_or_path, from_tf=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer


def load_model_and_tokenizer(model_name_or_path, device):
    model = AutoModel.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer
