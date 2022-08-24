#from transformers import BertConfig, BertPreTrainedModel, BertTokenizer, BertModel
from transformers import AutoConfig
from transformers import AutoModelForPreTraining
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers.modeling_utils import SequenceSummary
from torch import nn
import torch.nn.functional as F
import torch
from loss import AMSoftmax, HierarchicalMultiSimilarityLoss, HierarchicalTreeLogLoss, HierarchicalTreeLoss, ConditionalLogitLoss
import networkx as nx
from pytorch_metric_learning import losses, miners



class UMLSPretrainedModel(nn.Module):
    def __init__(self, device, model_name_or_path,
                 max_tree_dist=3,
                 clogit_alpha=2):
        super(UMLSPretrainedModel, self).__init__()

        self.device = device
        self.model_name_or_path = model_name_or_path
        if self.model_name_or_path.find("large") >= 0:
            self.feature_dim = 1024
        else:
            self.feature_dim = 768
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(0.1)

        # self.max_tree_dist = max_tree_dist
        # self.tree_dist_embedding = nn.Sequential(
        #     nn.Embedding(self.max_tree_dist + 1, 1, padding_idx=self.max_tree_dist),
        #     nn.Tanh()
        #     )
        # self.tree_dist_embedding[0].weight = nn.Parameter(torch.tensor([[3], [1.4], [1.1], [0]]))

        # self.tree_dist_embedding = nn.Embedding(self.max_tree_dist + 1, 1, padding_idx=self.max_tree_dist)
        # self.tree_dist_embedding.weight = nn.Parameter(torch.tensor([[1], [0.8], [0.4], [0]]))
        # for param in self.tree_dist_embedding.parameters():
        #     param.requires_grad = False

        # self.tree_loss = HierarchicalTreeLoss()
        # self.tree_loss = HierarchicalTreeLogLoss()
        self.tree_loss = ConditionalLogitLoss(alpha=clogit_alpha)
        self.tree_loss_fn = losses.MultiSimilarityLoss(alpha=2, beta=50)
        self.miner = miners.MultiSimilarityMiner(epsilon=0.1)

        self.standard_dataloader = None

        self.sequence_summary = SequenceSummary(AutoConfig.from_pretrained(model_name_or_path)) # Now only used for XLNet
    def ms_loss(self, pooled_output, label):
        pairs = self.miner(pooled_output, label)
        loss = self.tree_loss_fn(pooled_output, label, pairs)
        return loss

    def log_loss(self, embeddings, labels):
        pairs = self.miner(embeddings, labels)        
        emb_normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        dist_mat = torch.matmul(emb_normalized, emb_normalized.t())
        loss = self.pos_loss(dist_mat, pairs)
        return loss


    def get_sentence_feature(self, input_ids):
        # bert, albert, roberta
        if self.model_name_or_path.find("xlnet") < 0:
            outputs = self.bert(input_ids)
            pooled_output = outputs[1]
            return pooled_output

        # xlnet
        outputs = self.bert(input_ids)
        pooled_output = self.sequence_summary(outputs[0])
        return pooled_output

    def get_tree_loss(self, anchor_ids, all_samples_ids, all_samples_dists):
        anchor_output = self.get_sentence_feature(anchor_ids)
        all_samples_output = self.get_sentence_feature(all_samples_ids)
        loss = self.tree_loss(anchor_output, all_samples_output, all_samples_dists)
        
        return loss



    def init_standard_feature(self):
        if self.standard_dataloader is not None:
            for index, batch in enumerate(self.standard_dataloader):
                input_ids = batch[0].to(self.device)
                outputs = self.get_sentence_feature(input_ids)
                normalized_standard_feature = torch.norm(
                    outputs, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                normalized_standard_feature = torch.div(
                    outputs, normalized_standard_feature)
                if index == 0:
                    self.standard_feature = normalized_standard_feature
                else:
                    self.standard_feature = torch.cat(
                        (self.standard_feature, normalized_standard_feature), 0)
            assert self.standard_feature.shape == (
                self.num_label, self.feature_dim), self.standard_feature.shape
        return None

    def predict_by_cosine(self, input_ids):
        pooled_output = self.get_sentence_feature(input_ids)

        normalized_feature = torch.norm(
            pooled_output, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        normalized_feature = torch.div(pooled_output, normalized_feature)
        sim_mat = torch.matmul(normalized_feature, torch.t(
            self.standard_feature))  # batch_size * num_label
        return torch.max(sim_mat, dim=1)[1], sim_mat