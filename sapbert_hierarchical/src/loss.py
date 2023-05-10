import torch
import torch.nn as nn

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
                loss += torch.mean(pos_loss + neg_loss)
        return loss