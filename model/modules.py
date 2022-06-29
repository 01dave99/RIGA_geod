import torch
from torch import nn


class LearnableLogOptimalTransport(nn.Module):
    '''
    Optimal Transport Layer with learnable thresholds
    Reference: GeoTransformer, Zheng Qin et al.
    '''
    def __init__(self, num_iter, inf=1e6):
        super(LearnableLogOptimalTransport, self).__init__()
        self.num_iter = num_iter
        self.register_parameter('alpha', torch.nn.Parameter(torch.tensor(1.)))
        self.inf = inf

    def log_sinkhorn_normalization(self, scores, log_mu, log_nu):
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(self.num_iter):
            u = log_mu - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(scores + u.unsqueeze(2), dim=1)
        return scores + u.unsqueeze(2) + v.unsqueeze(1)

    def forward(self, scores, row_masks, col_masks):
        r"""
        Optimal transport with Sinkhorn.
        :param scores: torch.Tensor (B, M, N)
        :param row_masks: torch.Tensor (B, M)
        :param col_masks: torch.Tensor (B, N)
        :return matching_scores: torch.Tensor (B, M+1, N+1)
        """
        batch_size, num_row, num_col = scores.shape
        ninf = torch.tensor(-self.inf)

        padded_row_masks = torch.zeros(batch_size, num_row + 1, dtype=torch.bool)
        padded_row_masks[:, :num_row] = ~row_masks
        padded_col_masks = torch.zeros(batch_size, num_col + 1, dtype=torch.bool)
        padded_col_masks[:, :num_col] = ~col_masks

        padded_col = self.alpha.expand(batch_size, num_row, 1)
        padded_row = self.alpha.expand(batch_size, 1, num_col + 1)
        padded_scores = torch.cat([torch.cat([scores, padded_col], dim=-1), padded_row], dim=1)

        padded_score_masks = torch.logical_or(padded_row_masks.unsqueeze(2), padded_col_masks.unsqueeze(1))
        padded_scores[padded_score_masks] = ninf

        num_valid_row = row_masks.float().sum(1)
        num_valid_col = col_masks.float().sum(1)
        norm = -torch.log(num_valid_row + num_valid_col)  # (B,)

        log_mu = torch.empty(batch_size, num_row + 1)
        log_mu[:, :num_row] = norm.unsqueeze(1)
        log_mu[:, num_row] = torch.log(num_valid_col) + norm
        log_mu[padded_row_masks] = ninf

        log_nu = torch.empty(batch_size, num_col + 1)
        log_nu[:, :num_col] = norm.unsqueeze(1)
        log_nu[:, num_col] = torch.log(num_valid_row) + norm
        log_nu[padded_col_masks] = ninf

        outputs = self.log_sinkhorn_normalization(padded_scores, log_mu, log_nu)
        outputs = outputs - norm.unsqueeze(1).unsqueeze(2)

        return outputs

    def __repr__(self):
        format_string = self.__class__.__name__ + '(num_iter={})'.format(self.num_iter)
        return format_string


class NodeCorrespondenceSelector(nn.Module):
    def __init__(self, num_corr, thres=0.1):
        super(NodeCorrespondenceSelector, self).__init__()
        self.num_corr = num_corr
        self.thres = thres

    def forward(self, confidence_map):
        batch_size, num_src, num_tgt = confidence_map.shape[0], confidence_map.shape[1], confidence_map.shape[2]
        corr_mask = torch.gt(confidence_map, self.thres)
        confidence_map[~corr_mask] = 0.

        sel_map = confidence_map.view(batch_size, -1).contiguous()  # [B, N, M] -> [B, N * M]
        sel_corr_inds = torch.topk(sel_map, k=self.num_corr, dim=-1, largest=False)[1]  # [B, n_corr]

        sel_corr_inds = sel_corr_inds.unsqueeze(-1).repeat(1, 1, 2) #[B, n_corr, 2]
        sel_corr_inds[..., 0] = sel_corr_inds[..., 0] // num_tgt
        sel_corr_inds[..., 1] = sel_corr_inds[..., 1] % num_tgt

        return sel_corr_inds
