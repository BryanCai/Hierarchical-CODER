import torch
import torch.nn as nn
import networkx as nx


def logsumexp(x, keep_mask=None, add_one=True, dim=1):
    if keep_mask is not None:
        x = x.masked_fill(~keep_mask, torch.finfo(x.dtype).min)
    if add_one:
        zeros = torch.zeros(x.size(dim - 1), dtype=x.dtype, device=x.device).unsqueeze(
            dim
        )
        x = torch.cat([x, zeros], dim=dim)

    output = torch.logsumexp(x, dim=dim, keepdim=True)
    if keep_mask is not None:
        output = output.masked_fill(~torch.any(keep_mask, dim=dim, keepdim=True), 0)
    return output

def sumlogexp(x, keep_mask=None, dim=1):
    if keep_mask is not None:
        x = x.masked_fill(~keep_mask, torch.finfo(x.dtype).min)

    output = torch.sum(torch.log(torch.add(torch.exp(x), 1)), dim=dim, keepdim=True)
    if keep_mask is not None:
        output = output.masked_fill(~torch.any(keep_mask, dim=dim, keepdim=True), 0)
    return output

def masked_mse(dist_mat, tree_embeds, keep_mask=None, dim=1):
    if keep_mask is not None:
        dist_mat = dist_mat.masked_fill(~keep_mask, 0)

    output = torch.sum(torch.square(dist_mat - tree_embeds), dim=dim, keepdim=True)
    if keep_mask is not None:
        output = output.masked_fill(~torch.any(keep_mask, dim=dim, keepdim=True), 0)
    return output


class AMSoftmax(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes=10,
                 m=0.35,
                 s=30):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, label):
        #print(x.shape, lb.shape, self.in_feats)
        #assert x.size()[0] == label.size()[0]
        #assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        # print(x_norm.shape, w_norm.shape, costh.shape)
        lb_view = label.view(-1, 1).to(x.device)
        delt_costh = torch.zeros(costh.size()).to(x.device).scatter_(1, lb_view, self.m)
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, label)
        return loss, costh_m_s

    def predict(self, x):
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        return costh

class HierarchicalMultiSimilarityLoss(nn.Module):
    def __init__(self, alpha=2, beta=50, base=0.5, **kwargs):
        super(HierarchicalMultiSimilarityLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base

    def forward(self, embeddings, labels, indices_tuple):
        emb_normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        mat = torch.matmul(emb_normalized, emb_normalized.t())
        a1, p, a2, n = indices_tuple
        pos_mask, neg_mask = torch.zeros_like(mat), torch.zeros_like(mat)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1
        pos_exp = self.base - mat
        neg_exp = mat - self.base
        pos_loss = (1.0 / self.alpha) * logsumexp(
            self.alpha * pos_exp, keep_mask=pos_mask.bool(), add_one=True
        )
        neg_loss = (1.0 / self.beta) * logsumexp(
            self.beta * neg_exp, keep_mask=neg_mask.bool(), add_one=True
        )

        return torch.mean(pos_loss + neg_loss)

class HierarchicalLogLoss(nn.Module):
    def __init__(self, base=0.5, **kwargs):
        super(HierarchicalLogLoss, self).__init__()
        self.base = base

    def forward(self, dist_mat, tree_embeds, tree_mask, indices_tuple):
        a1, p, a2, n = indices_tuple
        pos_mask, neg_mask = torch.zeros_like(dist_mat), torch.zeros_like(dist_mat)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1
        pos_exp = self.base - dist_mat
        neg_exp = dist_mat - self.base
        pos_loss = sumlogexp(pos_exp, keep_mask=pos_mask.bool())
        neg_loss = sumlogexp(neg_exp, keep_mask=neg_mask.bool())

        tree_loss = masked_mse(dist_mat, tree_embeds, keep_mask=tree_mask)

        return torch.mean(pos_loss + neg_loss + tree_loss)

class HierarchicalTreeLoss(nn.Module):
    def __init__(self, base=0.5, **kwargs):
        super(HierarchicalTreeLoss, self).__init__()
        self.base = base
        self.cos = nn.CosineSimilarity()

    def forward(self, anchor_embed, neg_samples_embed, neg_dists_embed):
        cdist = self.cos(anchor_embed, neg_samples_embed)
        neg_loss = torch.sum(torch.log(torch.add(torch.exp(cdist - self.base), 1)))
        tree_loss = torch.sum(torch.square(cdist - neg_dists_embed))
        return neg_loss + tree_loss

if __name__ == '__main__':
    criteria = AMSoftmax(20, 5)
    a = torch.randn(10, 20)
    lb = torch.randint(0, 5, (10, ), dtype=torch.long)
    loss = criteria(a, lb)
    loss.backward()

    print(loss.detach().numpy())
    print(list(criteria.parameters())[0].shape)
    print(type(next(criteria.parameters())))
    print(lb)
