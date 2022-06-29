import torch
import torch.nn as nn
import numpy as np
from lib.utils import square_distance
import torch.nn.functional as F


class MetricLoss(nn.Module):
    '''
    Class including calculation of losses and metrics
    '''
    def __init__(self, config):
        super(MetricLoss, self).__init__()
        self.config = config

    def contranstiveLoss(self, src_desc, tgt_desc, pos_mask, neg_mask, margin=1., eps=1e-9):
        '''
        Constrastive loss used for metric learning
        :param src_desc: descriptors of source patches in torch.tensor of shape [B, N, F]
        :param tgt_desc: descriptors of target patches in torch.tensor of shape [B, N, F]
        :param pos_mask: binary mask in torch.tensor of shape[B, N, N] to indicate correspondences
        :param neg_mask: binary mask in torch.tensor of shape[B, N, M] to indicate non-correspondences
        '''
        #src_desc_norm = torch.sum(src_desc.pow(2), -1)
        distances = square_distance(src_desc, tgt_desc) #[B, N, F] x [B, N, F]-> [B, N, N]
        pos_loss = torch.sum(pos_mask * distances) / torch.sum(pos_mask)
        neg_loss = neg_mask * F.relu(margin - distances - eps)

        sel_neg, _ = torch.max(neg_loss, dim=-1)

        neg_loss = torch.mean(sel_neg)

        loss = 0.5 * (pos_loss + neg_loss)
        return loss

    def infoNCE(self, src_desc, tgt_desc, neg_mask, T=0.1):
        '''
        infoNCE loss used for metric learning
        :param src_desc: descriptors of source patches in torch.tensor of shape [B, N, F]
        :param tgt_desc: descriptors of target patches in torch.tensor of shape [B, N, F]
        :param neg_mask: binary mask in torch.tensor of shape[B, N, N] to indicate non-correspondences
        :param T: temperature used for re-weighting
        :return: infoNCE loss
        '''
        l_pos = torch.einsum('bnc, bnc -> bn', [src_desc, tgt_desc]) #[B, N]
        l_neg = torch.einsum('bnc, bcm->bnm', [src_desc, tgt_desc.permute(0, 2, 1).contiguous()]) #[B, N, N]

        l_pos = torch.exp(l_pos / T)

        l_src2tgt = (torch.sum(neg_mask * torch.exp(l_neg / T), dim=2)) / (1e-6 + torch.sum(neg_mask, dim=2)) #[B, N]
        l_tgt2src = (torch.sum(neg_mask * torch.exp(l_neg / T), dim=1)) / (1e-6 + torch.sum(neg_mask, dim=1)) #[B, N]

        infoNCE_src2tgt = -torch.log(l_pos / (l_pos + l_src2tgt))
        infoNCE_tgt2src = -torch.log(l_pos / (l_pos + l_tgt2src))

        infoNCE_loss = 0.5 * (torch.mean(infoNCE_src2tgt) + torch.mean(infoNCE_tgt2src))

        return infoNCE_loss

    def circleLoss(self, src_desc, tgt_desc, pos_mask, neg_mask, log_scale=16, pos_optimal=0.1, neg_optimal=1.4, pos_margin=0.1, neg_margin=1.4):
        '''
        Calculate circle loss for metric learning.
        :param src_desc: descriptors of source patches in torch.tensor of shape [B, N, F]
        :param tgt_desc: descriptors of target patches in torch.tensor of shape [B, M, F]
        :param pos_mask: binary mask in torch.tensor of shape[B, N, M] to indicate correspondences
        :param neg_mask: binary mask in torch.tensor of shape[B, N, M] to indicate non-correspondences
        :return: Calculated circle loss
        '''

        feats_dist = torch.sqrt(square_distance(src_desc, tgt_desc))

        pos_scale = pos_mask.detach().clone()

        pos_mask = torch.gt(pos_mask, 0)
        neg_mask = torch.gt(neg_mask, 0)

        # get anchors that have both positive and negative pairs
        row_sel = ((pos_mask.sum(-1) > 0) * (neg_mask.sum(-1) > 0)).detach()
        col_sel = ((pos_mask.sum(-2) > 0) * (neg_mask.sum(-2) > 0)).detach()

        # get alpha for both positive and negative pairs
        pos_weight = feats_dist - 1e5 * (~pos_mask).float() # mask the non-positive
        pos_weight = pos_weight - pos_optimal # mask the uninformative positive
        pos_weight = torch.clamp(pos_weight, min=0.).detach()

        pos_weight = pos_weight * pos_scale

        neg_weight = feats_dist + 1e5 * (~neg_mask).float() # mask the non-negative
        neg_weight = neg_optimal - neg_weight # mask the uninformative negative
        neg_weight = torch.clamp(neg_weight, min=0.).detach()

        lse_pos_row = torch.logsumexp(log_scale * (feats_dist - pos_margin) * pos_weight, dim=-1)
        lse_pos_col = torch.logsumexp(log_scale * (feats_dist - pos_margin) * pos_weight, dim=-2)

        lse_neg_row = torch.logsumexp(log_scale * (neg_margin - feats_dist) * neg_weight, dim=-1)
        lse_neg_col = torch.logsumexp(log_scale * (neg_margin - feats_dist) * neg_weight, dim=-2)

        loss_row = F.softplus(lse_pos_row + lse_neg_row) / log_scale
        loss_col = F.softplus(lse_pos_col + lse_neg_col) / log_scale

        circle_loss = (loss_row[row_sel].mean() + loss_col[col_sel].mean()) / 2.
        return circle_loss

    def calc_fine_matching_loss(self, matching_score, matching_gt, src_mask, tgt_mask):
        src_mask = src_mask.unsqueeze(-1).expand(-1, -1, -1, matching_gt.shape[-1])
        tgt_mask = tgt_mask.unsqueeze(-2).expand(-1, -1, matching_gt.shape[-2], -1)

        mask = matching_gt.clone()
        mask[:, :, :-1, :] = mask[:, :, :-1, :] * src_mask
        mask[:, :, :, :-1] = mask[:, :, :, :-1] * tgt_mask
        loss = torch.sum(-matching_score * mask) / torch.sum(mask)
        return loss

    def calc_fine_matching_recall(self, matching_score, matching_gt, src_mask, tgt_mask):
        matching_score = torch.exp(matching_score)
        matching_gt[:, :, :-1, :] = matching_gt[:, :, :-1, :] * src_mask.unsqueeze(-1)
        matching_gt[:, :, :, :-1] = matching_gt[:, :, :, :-1] * tgt_mask.unsqueeze(-2)
        _, sel_idx = torch.max(matching_score[:, :, :-1, :], dim=-1, keepdim=True)
        src_gt = torch.gather(matching_gt, index=sel_idx, dim=-1)
        _, sel_idx = torch.max(matching_score[:, :, :, :-1], dim=-2, keepdim=True)
        tgt_gt = torch.gather(matching_gt, index=sel_idx, dim=-2)
        recall = (src_gt.sum() + tgt_gt.sum()) / (src_mask.sum() + tgt_mask.sum())
        return recall

    def calc_recall(self, src_desc, tgt_desc, pos_mask, dis_type='inner'):
        '''
        Calculate matching precision and recall
        :param src_desc: descriptors of source patches in torch.tensor of shape [B, N, F]
        :param tgt_desc: descriptors of target patches in torch.tensor of shape [B, N, F]
        :return: precision and recall
        '''
        assert dis_type in ['euclid', 'inner']
        pos_mask = (pos_mask > 0).float()
        total = torch.sum((torch.sum(pos_mask, dim=-1) > 0).float())
        if dis_type == 'euclid':
            dis = square_distance(src_desc, tgt_desc)
            _, sel_idx = torch.min(dis, dim=-1) #[B, N]
        else:
            sim = torch.einsum('bnc, bcm->bnm', [src_desc, tgt_desc.permute(0, 2, 1).contiguous()]) #[B, N, N]
            _, sel_idx = torch.max(sim, dim=-1) #[B, N]

        sel_idx = sel_idx.unsqueeze(-1)
        pred_true = torch.gather(pos_mask, dim=-1, index=sel_idx)
        pred_true = torch.sum(pred_true)
        return pred_true / total


    def forward(self, src_desc, tgt_desc, pos_mask, neg_mask, matching_scores, matching_gt, src_mask, tgt_mask, ovelap_radius=0.0375):
        '''
        :param src_desc: descriptors of source patches in torch.tensor of shape [B, N, F]
        :param tgt_desc: descriptors of target patches in torch.tensor of shape [B, N, F]
        :param neg_mask: binary mask in torch.tensor of shape[B, N, M] to indicate non-correspondences
        :return:
        '''
        stats = dict()
        coarse_circle_loss = self.circleLoss(src_desc, tgt_desc, pos_mask, neg_mask)
        stats['coarse_loss'] = coarse_circle_loss
        coarse_matching_recall = self.calc_recall(src_desc, tgt_desc, pos_mask, dis_type='euclid')
        stats['coarse_matching_recall'] = coarse_matching_recall
        fine_loss = self.calc_fine_matching_loss(matching_scores, matching_gt, src_mask, tgt_mask)
        stats['fine_loss'] = fine_loss
        fine_matching_recall = self.calc_fine_matching_recall(matching_scores, matching_gt, src_mask, tgt_mask)
        stats['fine_matching_recall'] = fine_matching_recall

        return stats
