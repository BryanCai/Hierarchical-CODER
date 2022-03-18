#from transformers import BertConfig, BertPreTrainedModel, BertTokenizer, BertModel
from transformers import AutoConfig
from transformers import AutoModelForPreTraining
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers.modeling_utils import SequenceSummary
from torch import nn
import torch.nn.functional as F
import torch
from loss import AMSoftmax, HierarchicalMultiSimilarityLoss, HierarchicalLogLoss
from pytorch_metric_learning import losses, miners
from trans import TransE
import networkx as nx


class UMLSPretrainedModel(nn.Module):
    def __init__(self, device, model_name_or_path,
                 cui_label_count, rel_label_count, sty_label_count,
                 re_weight=1.0, sty_weight=0.1,
                 cui_loss_type="ms_loss",
                 trans_loss_type="TransE", trans_margin=1.0,
                 id2cui=None, cuitree=None, max_cui_dist=3):
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

        self.cui_loss_type = cui_loss_type
        self.cui_label_count = cui_label_count

        self.id2cui = id2cui
        self.cuitree = cuitree
        self.max_cui_dist = max_cui_dist
        self.tree_embedding = nn.Sequential(
            nn.Embedding(self.max_cui_dist + 1, 1, padding_idx=self.max_cui_dist),
            nn.Tanh()
            )

        if self.cui_loss_type == "softmax":
            self.cui_loss_fn = nn.CrossEntropyLoss()
            self.linear = nn.Linear(self.feature_dim, self.cui_label_count)
        if self.cui_loss_type == "am_softmax":
            self.cui_loss_fn = AMSoftmax(
                self.feature_dim, self.cui_label_count)
        if self.cui_loss_type == "old_ms_loss":
            self.cui_loss_fn = losses.MultiSimilarityLoss(alpha=2, beta=50)
            self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
        if self.cui_loss_type == "ms_loss":
            self.cui_loss_fn = HierarchicalMultiSimilarityLoss(alpha=2, beta=50)
            self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
        if self.cui_loss_type == "log_loss":
            self.cui_loss_fn = HierarchicalLogLoss()
            self.miner = miners.MultiSimilarityMiner(epsilon=0.1)

        self.trans_loss_type = trans_loss_type
        if self.trans_loss_type == "TransE":
            self.re_loss_fn = TransE(trans_margin)
        self.re_embedding = nn.Embedding(
            self.rel_label_count, self.feature_dim)

        self.standard_dataloader = None

        self.sequence_summary = SequenceSummary(AutoConfig.from_pretrained(model_name_or_path)) # Now only used for XLNet

    def softmax(self, logits, label):
        loss = self.cui_loss_fn(logits, label)
        return loss

    def am_softmax(self, pooled_output, label):
        loss, _ = self.cui_loss_fn(pooled_output, label)
        return loss

    def old_ms_loss(self, pooled_output, label):
        pairs = self.miner(pooled_output, label)
        loss = self.cui_loss_fn(pooled_output, label, pairs)
        return loss

    def ms_loss(self, pooled_output, label):
        pairs = self.miner(pooled_output, label)
        loss = self.cui_loss_fn(pooled_output, label, pairs)
        return loss

    def log_loss(self, embeddings, labels):
        pairs = self.miner(embeddings, labels)        
        emb_normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        dist_mat = torch.matmul(emb_normalized, emb_normalized.t())
        cui_dists = torch.zeros_like(dist_mat, dtype=torch.int)
        l = labels.cpu().numpy()
        for i in range(embeddings.size()[0]):
            for j in range(embeddings.size()[0]):
                cui_dists[i][j] = self.get_tree_distance(l[i], l[j])

        tree_mask = cui_dists < self.max_cui_dist
        tree_embeds = self.tree_embedding(cui_dists).squeeze(2)

        loss = self.cui_loss_fn(dist_mat, tree_embeds, tree_mask, pairs)
        return loss

    def get_tree_distance(self, id1, id2):
        cui1 = self.id2cui[id1]
        cui2 = self.id2cui[id2]
        try:
            x = nx.shortest_path_length(self.cuitree, cui1, cui2)
        except (nx.exception.NetworkXNoPath, nx.exception.NodeNotFound) as e:
            x = self.max_cui_dist
        return min(x, self.max_cui_dist)

    def calculate_loss(self, pooled_output=None, logits=None, label=None):
        if self.cui_loss_type == "softmax":
            return self.softmax(logits, label)
        if self.cui_loss_type == "am_softmax":
            return self.am_softmax(pooled_output, label)
        if self.cui_loss_type == "old_ms_loss":
            return self.old_ms_loss(pooled_output, label)
        if self.cui_loss_type == "ms_loss":
            return self.ms_loss(pooled_output, label)
        if self.cui_loss_type == "log_loss":
            return self.log_loss(pooled_output, label)

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


    # @profile
    def forward(self,
                input_ids_0, input_ids_1, input_ids_2,
                cui_label_0, cui_label_1, cui_label_2,
                sty_label_0, sty_label_1, sty_label_2,
                re_label):
        input_ids = torch.cat((input_ids_0, input_ids_1, input_ids_2), 0)
        cui_label = torch.cat((cui_label_0, cui_label_1, cui_label_2))
        sty_label = torch.cat((sty_label_0, sty_label_1, sty_label_2))
        #print(input_ids.shape, cui_label.shape, sty_label.shape)

        use_len = input_ids_0.shape[0]

        pooled_output = self.get_sentence_feature(
            input_ids)  # (3 * pair) * re_label
        logits_sty = self.linear_sty(pooled_output)
        sty_loss = self.sty_loss_fn(logits_sty, sty_label)

        if self.cui_loss_type == "softmax":
            logits = self.linear(pooled_output)
        else:
            logits = None
        cui_loss = self.calculate_loss(pooled_output, logits, cui_label)

        cui_0_output = pooled_output[0:use_len]
        cui_1_output = pooled_output[use_len:2 * use_len]
        cui_2_output = pooled_output[2 * use_len:]
        re_output = self.re_embedding(re_label)
        re_loss = self.re_loss_fn(
            cui_0_output, cui_1_output, cui_2_output, re_output)

        loss = self.sty_weight * sty_loss + cui_loss + self.re_weight * re_loss
        #print(sty_loss.device, cui_loss.device, re_loss.device)

        return loss, (sty_loss, cui_loss, re_loss)

    """
    def predict(self, input_ids):
        if self.loss_type == "softmax":
            return self.predict_by_softmax(input_ids)
        if self.loss_type == "am_softmax":
            return self.predict_by_amsoftmax(input_ids)        

    def predict_by_softmax(self, input_ids):
        pooled_output = self.get_sentence_feature(input_ids)
        logits = self.linear(pooled_output)
        return torch.max(logits, dim=1)[1], logits

    def predict_by_amsoftmax(self, input_ids):
        pooled_output = self.get_sentence_feature(input_ids)
        logits = self.loss_fn.predict(pooled_output)
        return torch.max(logits, dim=1)[1], logits
    """

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
