import torch
import torch.nn as nn

def clogit_partial(alpha, control_similarities, case_similarities):
    a = torch.exp((-alpha)*control_similarities)/(torch.exp((-alpha)*control_similarities) + torch.sum(torch.exp((-alpha)*case_similarities)))
    b = torch.exp(alpha*case_similarities)/(torch.exp(alpha*case_similarities) + torch.sum(torch.exp(alpha*control_similarities)))

    return torch.mean(torch.log(torch.cat((a, b))))

class TreeMultiSimilarityLoss(nn.Module):
    def __init__(self, alpha=2, beta=50, base=0.5, **kwargs):
        super(TreeMultiSimilarityLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.cos = nn.CosineSimilarity()

    def forward(self, query_embed1, query_embed2, dists):
        cdist = self.cos(query_embed1, query_embed2)
        cdist = (cdist+1)/2

        loss = 0
        min_dist = min(dists)
        max_dist = max(dists)
        for i in range(min_dist, max_dist):
            for j in range(i + 1, max_dist + 1):
                control_similarities = cdist[dists == i]
                case_similarities = cdist[dists == j]
                pos_loss = (1.0 / self.alpha)*torch.log(torch.add(torch.exp(self.alpha*(self.base - control_similarities)), 1))
                neg_loss = (1.0 / self.beta) *torch.log(torch.add(torch.exp(self.beta*(case_similarities - self.base)), 1))
                loss += torch.mean(pos_loss) + torch.mean(neg_loss)
        return loss


class ConditionalLogitLoss(nn.Module):
    def __init__(self, alpha=2, **kwargs):
        super(ConditionalLogitLoss, self).__init__()
        self.alpha = alpha
        self.cos = nn.CosineSimilarity()

    def forward_dist(self, anchor_embed, all_samples_embed, all_dists, multi_category=True):
        cdist = self.cos(anchor_embed, all_samples_embed)
        cdist = (cdist+1)/2

        if multi_category:
            loss = 0
            min_dist = min(all_dists)
            max_dist = max(all_dists)
            for i in range(min_dist, max_dist):
                for j in range(i + 1, max_dist + 1):
                    control_similarities = cdist[all_dists == i]
                    case_similarities = cdist[all_dists == j]
                    loss += clogit_partial(self.alpha, control_similarities, case_similarities)
            return loss
        else:
            control_similarities = cdist[all_dists <= 0]
            case_similarities = cdist[all_dists > 0]
            loss = clogit_partial(self.alpha, control_similarities, case_similarities)
            return loss


    def forward_re(self, anchor_embed, pos_embed, neg_embed):        
        control_similarities = (self.cos(anchor_embed, pos_embed) + 1)/2
        case_similarities = (self.cos(anchor_embed, neg_embed) + 1)/2
        loss = clogit_partial(self.alpha, control_similarities, case_similarities)

        return loss


    def forward_miner(self, embeddings, indices_tuple):
        if len(indices_tuple) == 3:
            a1, p, n = indices_tuple
            a2 = a1
        else:
            a1, p, a2, n = indices_tuple

        control_similarities = self.cos(embeddings[a1], embeddings[p])
        case_similarities = self.cos(embeddings[a2], embeddings[n])
        loss = clogit_partial(self.alpha, control_similarities, case_similarities)

        return loss