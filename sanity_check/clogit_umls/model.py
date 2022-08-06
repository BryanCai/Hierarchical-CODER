#from transformers import BertConfig, BertPreTrainedModel, BertTokenizer, BertModel
from transformers import AutoConfig
from transformers import AutoModelForPreTraining
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers.modeling_utils import SequenceSummary
from torch import nn
import torch.nn.functional as F
import torch
from loss import ConditionalLogitLoss
from pytorch_metric_learning import losses, miners
from trans import TransE


class UMLSPretrainedModel(nn.Module):
    def __init__(self, device, model_name_or_path,
                 rel_label_count, sty_label_count,
                 re_weight=1.0, sty_weight=0.1,
                 trans_loss_type="TransE", trans_margin=1.0,
                 ):
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



        self.cui_loss_fn = ConditionalLogitLoss()
        self.miner = miners.MultiSimilarityMiner(epsilon=0.1)


        self.trans_loss_type = trans_loss_type
        if self.trans_loss_type == "TransE":
            self.re_loss_fn = TransE(trans_margin)
        self.re_embedding = nn.Embedding(
            self.rel_label_count, self.feature_dim)


        self.standard_dataloader = None

        self.sequence_summary = SequenceSummary(AutoConfig.from_pretrained(model_name_or_path)) # Now only used for XLNet



    def calculate_loss(self, embeddings, labels):
        pairs = self.miner(embeddings, labels)

        loss = self.cui_loss_fn.forward_miner(embeddings, pairs)
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


    def forward(self,
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


        cui_loss = self.calculate_loss(pooled_output, cui_label)

        cui_0_output = pooled_output[0:use_len]
        cui_1_output = pooled_output[use_len:2 * use_len]
        cui_2_output = pooled_output[2 * use_len:]
        re_output = self.re_embedding(re_label)
        re_loss = self.re_loss_fn(
            cui_0_output, cui_1_output, cui_2_output, re_output)

        loss = self.sty_weight * sty_loss + cui_loss + self.re_weight * re_loss

        return loss
