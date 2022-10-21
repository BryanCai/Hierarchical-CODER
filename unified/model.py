#from transformers import BertConfig, BertPreTrainedModel, BertTokenizer, BertModel
from transformers import AutoConfig
from transformers import AutoModelForPreTraining
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers.modeling_utils import SequenceSummary
from torch import nn
import torch.nn.functional as F
import torch
from loss import ConditionalLogitLoss, HierarchicalMultiSimilarityLoss
from pytorch_metric_learning import losses, miners
from trans import TransE


class UMLSPretrainedModel(nn.Module):
    def __init__(self, base_model,
                 clogit_alpha=2,
                 sim_dim=500,
                 ):
        super(UMLSPretrainedModel, self).__init__()
        
        self.base_model = base_model
        self.sim_dim = sim_dim

        self.bert = AutoModel.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)


        self.clogit_loss_fn = ConditionalLogitLoss(alpha=clogit_alpha)
        self.ms_loss_fn = HierarchicalMultiSimilarityLoss(alpha=2, beta=2)


        self.miner = miners.MultiSimilarityMiner(epsilon=0.1)

        self.max_tree_dist = 2
        self.tree_dist_embedding = nn.Embedding(self.max_tree_dist + 1, 1, padding_idx=self.max_tree_dist)
        self.tree_dist_embedding.weight = nn.Parameter(torch.tensor([[1], [0.9**2], [0.9**4], [0]]))
        for param in self.tree_dist_embedding.parameters():
            param.requires_grad = False



    def calculate_sim_loss(self, embeddings, labels):
        sim_embeddings = embeddings[:,:self.sim_dim]
        pairs = self.miner(sim_embeddings, labels)
        loss = self.clogit_loss_fn.forward_miner(sim_embeddings, pairs)
        return loss


    def calculate_re_loss(self, re_0_embeddings, re_1_embeddings, random_embeddings):
        loss = self.clogit_loss_fn.forward_re(re_0_embeddings, re_1_embeddings, random_embeddings)
        return loss


    def calculate_ms_loss(self, pooled_output, label):
        pairs = self.miner(pooled_output, label)
        loss = self.ms_loss_fn.forward_umls(pooled_output, label, pairs)
        return loss

    def get_sentence_feature(self, input_ids):
        # bert, albert, roberta
        outputs = self.bert(input_ids)
        pooled_output = outputs[1]
        return pooled_output


    def get_umls_loss(self,
                      input_ids_0, input_ids_1, input_ids_2,
                      cui_label_0, cui_label_1, cui_label_2,
                      sty_label_0, sty_label_1, sty_label_2
                      ):
        input_ids = torch.cat((input_ids_0, input_ids_1, input_ids_2), 0)
        cui_label = torch.cat((cui_label_0, cui_label_1, cui_label_2))
        sty_label = torch.cat((sty_label_0, sty_label_1, sty_label_2))

        use_len = input_ids_0.shape[0]

        pooled_output = self.get_sentence_feature(input_ids)

        cui_loss = self.calculate_sim_loss(pooled_output, cui_label)

        # cui_0_output = pooled_output[0:use_len]
        # cui_1_output = pooled_output[use_len:2 * use_len]
        # cui_2_output = pooled_output[2 * use_len:]
        # re_loss = self.calculate_re_loss(cui_0_output, cui_1_output, cui_2_output)

        # loss = cui_loss + re_loss

        return cui_loss


    def get_ms_umls_loss(self,
                         input_ids_0, input_ids_1, input_ids_2,
                         cui_label_0, cui_label_1, cui_label_2,
                         sty_label_0, sty_label_1, sty_label_2
                         ):
        input_ids = torch.cat((input_ids_0, input_ids_1, input_ids_2), 0)
        cui_label = torch.cat((cui_label_0, cui_label_1, cui_label_2))

        use_len = input_ids_0.shape[0]

        pooled_output = self.get_sentence_feature(input_ids)

        cui_loss = self.calculate_ms_loss(pooled_output, cui_label)

        return cui_loss



    def get_tree_clogit_loss(self, anchor_ids, all_samples_ids, all_samples_dists):
        anchor_output = self.get_sentence_feature(anchor_ids)
        all_samples_output = self.get_sentence_feature(all_samples_ids)
        loss = self.clogit_loss_fn.forward_dist(anchor_output, all_samples_output, all_samples_dists)
        
        return loss


    def get_tree_ms_loss(self, anchor_ids, all_samples_ids, all_samples_dists):
        anchor_output = self.get_sentence_feature(anchor_ids)
        all_samples_output = self.get_sentence_feature(all_samples_ids)

        all_dists_output = self.tree_dist_embedding(all_samples_dists)

        loss = self.ms_loss_fn.forward_tree(anchor_output, all_samples_output, all_dists_output)

        return loss
