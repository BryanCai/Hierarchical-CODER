#from transformers import BertConfig, BertPreTrainedModel, BertTokenizer, BertModel
from transformers import AutoConfig
from transformers import AutoModelForPreTraining
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers.modeling_utils import SequenceSummary
from torch import nn
import torch.nn.functional as F
import torch
from loss import AMSoftmax, HierarchicalMultiSimilarityLoss, HierarchicalLogLoss, HierarchicalTreeLoss
from pytorch_metric_learning import losses, miners
from trans import TransE
import networkx as nx


class UMLSPretrainedModel(nn.Module):
    def __init__(self, device, model_name_or_path,
                 rel_label_count, sty_label_count,
                 re_weight=1.0, sty_weight=0.1,
                 trans_loss_type="TransE", trans_margin=1.0,
                 id2cui=None, cuitree=None, max_tree_dist=3,
                 loss_type = "log",
                 umls_neg_loss=False):
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

        self.rel_label_count = rel_label_count
        self.re_weight = re_weight

        self.sty_label_count = sty_label_count
        self.linear_sty = nn.Linear(self.feature_dim, self.sty_label_count)
        self.sty_loss_fn = nn.CrossEntropyLoss()
        self.sty_weight = sty_weight

        if loss_type == "log":
            self.pos_loss = HierarchicalLogLoss(use_neg_loss=umls_neg_loss)
        elif loss_type == "ms":
            self.pos_loss = HierarchicalMultiSimilarityLoss()

        self.miner = miners.MultiSimilarityMiner(epsilon=0.1)

        self.max_tree_dist = max_tree_dist
        # self.tree_dist_embedding = nn.Sequential(
        #     nn.Embedding(self.max_tree_dist + 1, 1, padding_idx=self.max_tree_dist),
        #     nn.Tanh()
        #     )
        # self.tree_dist_embedding[0].weight = nn.Parameter(torch.tensor([[3], [1.4], [1.1], [0]]))

        self.tree_dist_embedding = nn.Embedding(self.max_tree_dist + 1, 1, padding_idx=self.max_tree_dist)
        self.tree_dist_embedding.weight = nn.Parameter(torch.tensor([[1], [0.9**2], [0.9**4], [0]]))
        for param in self.tree_dist_embedding.parameters():
            param.requires_grad = False

        self.tree_loss = HierarchicalTreeLoss()

        self.trans_loss_type = trans_loss_type
        if self.trans_loss_type == "TransE":
            self.re_loss_fn = TransE(trans_margin)
        self.re_embedding = nn.Embedding(
            self.rel_label_count, self.feature_dim)

        self.standard_dataloader = None

        self.sequence_summary = SequenceSummary(AutoConfig.from_pretrained(model_name_or_path)) # Now only used for XLNet


    def cui_loss(self, embeddings, labels):
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

    def get_tree_loss(self, anchor_ids, neg_samples_ids, neg_samples_dists):
        anchor_output = self.get_sentence_feature(anchor_ids)
        neg_samples_output = self.get_sentence_feature(neg_samples_ids)

        neg_samples_dists[torch.lt(neg_samples_dists, self.max_tree_dist)] = self.max_tree_dist
        neg_dists_output = self.tree_dist_embedding(neg_samples_dists)

        loss = self.tree_loss(anchor_output, neg_samples_output, neg_dists_output)
        return loss

    # @profile
    def get_rel_loss(self,
                     input_ids_0, input_ids_1, input_ids_2,
                     cui_label_0, cui_label_1, cui_label_2,
                     sty_label_0, sty_label_1, sty_label_2,
                     re_label):
        input_ids = torch.cat((input_ids_0, input_ids_1, input_ids_2), 0)
        cui_label = torch.cat((cui_label_0, cui_label_1, cui_label_2))
        sty_label = torch.cat((sty_label_0, sty_label_1, sty_label_2))

        use_len = input_ids_0.shape[0]

        pooled_output = self.get_sentence_feature(
            input_ids)  # (3 * pair) * re_label
        logits_sty = self.linear_sty(pooled_output)
        sty_loss = self.sty_loss_fn(logits_sty, sty_label)


        cui_loss = self.cui_loss(pooled_output, cui_label)

        cui_0_output = pooled_output[0:use_len]
        cui_1_output = pooled_output[use_len:2 * use_len]
        cui_2_output = pooled_output[2 * use_len:]
        re_output = self.re_embedding(re_label)
        re_loss = self.re_loss_fn(
            cui_0_output, cui_1_output, cui_2_output, re_output)

        loss = self.sty_weight * sty_loss + cui_loss + self.re_weight * re_loss

        return loss, (sty_loss, re_loss)


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
