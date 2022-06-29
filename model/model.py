import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import square_distance
from model.ppftransformer import PointNetfeat, PPFTrans, MLP
from model.modules import LearnableLogOptimalTransport
from geotransformer.modules.geotransformer.geotransformer import GeometricStructureEmbedding, GeometricTransformer
from einops import rearrange


class RIGA(nn.Module):
    '''
    The RIGA pipeline
    '''
    def __init__(self, config):
        super(RIGA, self).__init__()
        self.config = config
        # descriptors from local geometry
        self.local_descriptor = PointNetfeat(input_type=config.input_type, proj_dim=config.proj_dim, out_dim=config.local_out_dim)
        # whether to take global contexts beyond local vicinity into consideration
        if config.with_transformer:
            # define the transformer part
            self.transformer = GeometricTransformer(config)
            # coarse level final descriptor projection
            self.coarse_proj = MLP([self.config.local_out_dim, self.config.local_out_dim // 2, self.config.local_out_dim // 4, self.config.descriptor_dim])
            # fine level final descriptor projection
            self.fine_proj = MLP([self.config.local_out_dim, self.config.local_out_dim // 2, self.config.local_out_dim // 4, self.config.descriptor_dim])
        else:
            self.transformer = None
        # learnable Optimal Transport Layer
        self.OT = LearnableLogOptimalTransport(num_iter=100)
        # the number of correspondences used for each point cloud pair during training
        self.max_corr = config.max_corr

        self.mode = config.mode # current phase, should be in ['train', 'val', 'test']

    def interpolate(self, pcd, nodes, node_features, knn_node_inds):
        '''
        Interpolate node descriptors to point descriptors weighted by euclidean distance in geometric space
        :param pcd: concatenated point clouds in shape [n, 3]
        :param nodes: a batch of node coordinates in shape [b, m, 3]
        :param node_features: a batch of node descriptors in shape [b, m, c]
        :param knn_node_inds: indices in shape [n, k], indicating the mapping between each point to its nearest (geometric space) node in the same frame
        :return: interpolated point features in shape [n, c]
        '''
        n, k = knn_node_inds.shape[0], knn_node_inds.shape[1]
        # flatten nodes [b, m, 3] -> [b * m, 3]
        flattened_nodes = nodes.view(-1, nodes.shape[-1]).contiguous()
        # flatten node features [b, m, c] -> [b * m, c]
        flattened_features = node_features.view(-1, node_features.shape[-1]).contiguous()
        # flatten knn node ids [n, k] -> [n * k]
        flattened_knn_node_inds = knn_node_inds.view(-1).contiguous()

        # select those knn nodes and reshape
        selected_node_xyz = flattened_nodes[flattened_knn_node_inds].view(n, k, -1) #[n, k, 3]
        selected_node_features = flattened_features[flattened_knn_node_inds].view(n, k, -1) #[n, k, 3]

        # expand point clouds [n, 3] -> [n, k, 3] (repeat each point k times)
        expanded_pcd = pcd.unsqueeze(1).expand(-1, k, -1) #[n, k, 3]
        # calculate euclidean distance
        dist = torch.sqrt(torch.sum((expanded_pcd - selected_node_xyz)**2, dim=-1)) # [n, k]
        # convert euclidean distance to similarity
        similarity = 1. / (1e-8 + dist) #[n, k]
        # normalize similarity
        similarity = similarity / torch.sum(similarity, dim=1, keepdim=True)
        # reshape [n, k] -> [n, k, 1]
        similarity = torch.unsqueeze(similarity, dim=-1)
        # interpolate node features to point features according to similarity
        interpolated_point_features = torch.sum(similarity * selected_node_features, dim=1) #[n, c]

        return interpolated_point_features

    def forward(self, src_pcd, tgt_pcd, src_nodes, tgt_nodes, src_node_geod, tgt_node_geod, src_feats, tgt_feats, src_knn_node_inds, tgt_knn_node_inds,
                src_p2n_inds, tgt_p2n_inds, src_p2n_masks, tgt_p2n_masks, rot, trans, normalize=True):
        # get current batch size
        batch_size = src_nodes.shape[0]
        # get local geometric descriptors for nodes from source point cloud
        src_feats = self.local_descriptor(src_feats.permute(0, 2, 1).contiguous())
        src_feats = src_feats.view(batch_size, -1, src_feats.shape[-1]).contiguous() #[B, N, F]
        
        # get local geometric descriptors for nodes from target point cloud
        tgt_feats = self.local_descriptor(tgt_feats.permute(0, 2, 1).contiguous())
        tgt_feats = tgt_feats.view(batch_size, -1, tgt_feats.shape[-1]).contiguous() #[B, N, F]
        
        

        if self.transformer is not None:
            # use transformer for learned global context aggregation
            src_feats, tgt_feats= self.transformer(src_feats, tgt_feats, src_node_geod, tgt_node_geod)

        # interpolate node features to point features
        src_point_feats = self.interpolate(src_pcd, src_nodes, src_feats, src_knn_node_inds)
        tgt_point_feats = self.interpolate(tgt_pcd, tgt_nodes, tgt_feats, tgt_knn_node_inds)

        # final projection of point features
        src_point_feats = self.fine_proj(src_point_feats.permute(1, 0).contiguous().unsqueeze(0))[0].permute(1, 0).contiguous()
        tgt_point_feats = self.fine_proj(tgt_point_feats.permute(1, 0).contiguous().unsqueeze(0))[0].permute(1, 0).contiguous()
        # final projection of node features
        src_feats = self.coarse_proj(src_feats.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        tgt_feats = self.coarse_proj(tgt_feats.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()

        # whether to normalize node features
        if normalize:
            src_feats = F.normalize(src_feats, p=2, dim=-1)
            tgt_feats = F.normalize(tgt_feats, p=2, dim=-1)
        #####################################################################################################################
        # coarse-to-fine correspondences
        # training and validation mode
        if self.mode != 'test':
            # get current batch size, number of nodes and number of points per node patch
            b1, n1, m1 = src_p2n_inds.shape
            # resize point to node assignment [b1, n1, m1] -> [b1 * n1 * m1, c]
            src_p2n_inds = src_p2n_inds.view(-1).contiguous().unsqueeze(-1).expand(-1, src_point_feats.shape[-1])
            # get current batch size, number of nodes and number of points per node patch
            b2, n2, m2 = tgt_p2n_inds.shape
            # resize point to node assignment [b2, n2, m2] -> [b2 * n2 * m2, c]
            tgt_p2n_inds = tgt_p2n_inds.view(-1).contiguous().unsqueeze(-1).expand(-1, tgt_point_feats.shape[-1])
            # select point features and reshape [b * n * m, c] -> [b * n, m, c]
            sel_src_point_feats = torch.gather(src_point_feats, dim=0, index=src_p2n_inds.long()).view(b1 * n1, m1, -1).contiguous()
            sel_tgt_point_feats = torch.gather(tgt_point_feats, dim=0, index=tgt_p2n_inds.long()).view(b2 * n2, m2, -1).contiguous()
            # get the dimension of features (denoted as 'c' above)
            dim = src_point_feats.shape[-1]
            # get the matching scores between corresponding node vicinity
            scores = torch.einsum('bnd, bmd->bnm', sel_src_point_feats, sel_tgt_point_feats)
            scores = scores / dim ** 0.5
            # optimal transport (dual normalization)
            matching_scores = self.OT(scores, row_masks=src_p2n_masks.view(b1 * n1, -1).contiguous().bool(), col_masks=tgt_p2n_masks.view(b2 * n2, -1).contiguous().bool())
            # resize scores [b * n, m1 + 1, m2 + 1] -> [b, n, m1 + 1, m2 + 1]
            matching_scores = matching_scores.view(b1, n1, m1 + 1, m2 + 1).contiguous() # Shape
            return src_feats, tgt_feats, src_point_feats, tgt_point_feats, matching_scores, src_p2n_masks, tgt_p2n_masks
        # test mode
        else:
            ####################################################################################
            # Dual-softmax for coarse level matching
            coarse_matching_score = 1. / (1e-8 + square_distance(src_feats, tgt_feats))
            coarse_matching_score1 = torch.softmax(coarse_matching_score, dim=-1)
            coarse_matching_score2 = torch.softmax(coarse_matching_score, dim=-2)
            coarse_matching_score = coarse_matching_score1 * coarse_matching_score2
            ####################################################################################
            # reshape [b, m1, m2] -> [b, m1 * m2]
            coarse_matching_score = coarse_matching_score.view(coarse_matching_score.shape[0], -1).contiguous()
            # select k nodes with maximum scores
            sel_conf, sel_inds = torch.topk(coarse_matching_score, k=self.max_corr, sorted=True, dim=-1)  # [b, max_corr]
            # convert 1-D indices to 2-D indices
            src_node_inds = sel_inds // coarse_matching_score1.shape[-1]
            tgt_node_inds = sel_inds % coarse_matching_score1.shape[-1]
            # collect selected nodes together with their vicinity points
            gathered_src_patch_inds = torch.gather(src_p2n_inds, dim=1, index=src_node_inds.unsqueeze(-1).expand(-1, -1, src_p2n_inds.shape[-1]))  # [b, max_corr, n_points]
            gathered_tgt_patch_inds = torch.gather(tgt_p2n_inds, dim=1, index=tgt_node_inds.unsqueeze(-1).expand(-1, -1, tgt_p2n_inds.shape[-1]))  # [b, max_corr, n_points]
            #
            gathered_src_patch_masks = torch.gather(src_p2n_masks, dim=1, index=src_node_inds.unsqueeze(-1).expand(-1, -1, src_p2n_inds.shape[-1]))  # [b, max_corr, n_points]
            gathered_tgt_patch_masks = torch.gather(tgt_p2n_masks, dim=1, index=tgt_node_inds.unsqueeze(-1).expand(-1, -1, src_p2n_inds.shape[-1]))  # [b, max_corr, n_points]

            # get new dimensions
            b1, n1, m1 = gathered_src_patch_inds.shape
            b2, n2, m2 = gathered_tgt_patch_inds.shape
            # collect selected nodes together with their vicinity point features
            gathered_src_patch_feats = torch.gather(src_point_feats, dim=0, index=gathered_src_patch_inds.long().view(-1).contiguous().unsqueeze(-1).expand(-1, src_point_feats.shape[-1])).view(b1 * n1, m1, -1).contiguous()  # [B * max_corr, n_points, c]
            gathered_tgt_patch_feats = torch.gather(tgt_point_feats, dim=0, index=gathered_tgt_patch_inds.long().view(-1).contiguous().unsqueeze(-1).expand(-1, tgt_point_feats.shape[-1])).view(b2 * n2, m2, -1).contiguous()  # [B * max_corr, n_points, c]
            # get the dimension of features (denoted as 'c' above)
            dim = src_point_feats.shape[-1]
            # get the matching scores between corresponding node vicinity
            scores = torch.einsum('bnd, bmd->bnm', gathered_src_patch_feats, gathered_tgt_patch_feats)
            scores = scores / dim ** 0.5
            # optimal transport (dual normalization)
            matching_scores = self.OT(scores, row_masks=gathered_src_patch_masks.view(b1 * n1, -1).contiguous().bool(),
                                      col_masks=gathered_tgt_patch_masks.view(b2 * n2, -1).contiguous().bool())
            # resize to [b, max_corr, n_points1, n_points2]
            matching_scores = matching_scores.view(b1, n1, m1 + 1, m2 + 1).contiguous()
            matching_scores = torch.exp(matching_scores)

            ################################################################
            # collect the coordinates of points in the vicinity area of selected nodes
            gathered_src_patch_xyz = torch.gather(src_pcd, dim=0, index=gathered_src_patch_inds.long().view(-1).contiguous().unsqueeze(-1).expand(-1, 3)).view(b1, n1, m1, -1).contiguous()  # [B, max_corr, n_points, c]
            gathered_tgt_patch_xyz = torch.gather(tgt_pcd, dim=0, index=gathered_tgt_patch_inds.long().view(-1).contiguous().unsqueeze(-1).expand(-1, 3)).view(b2, n2, m2, -1).contiguous()  # [B, max_corr, n_points, c]
            # reshape mask [b, max_corr, n_points] -> [b, max_corr, n_points, n_points]
            src_mask = gathered_src_patch_masks.unsqueeze(-1).expand(-1, -1, -1, matching_scores.shape[-1])
            # reshape mask [b, max_corr, n_points] -> [b, max_corr, n_points, n_points]
            tgt_mask = gathered_tgt_patch_masks.unsqueeze(-2).expand(-1, -1, matching_scores.shape[-2], -1)
            # filter out points that are sampled repeatedly
            matching_scores[:, :, :-1, :] = matching_scores[:, :, :-1, :] * src_mask
            matching_scores[:, :, :, :-1] = matching_scores[:, :, :, :-1] * tgt_mask
            # drop the slack rows and columns
            matching_scores = matching_scores[:, :, :-1, :-1].contiguous()

            ########################################################################################
            # Masking out the non_maximum
            # constraints
            top_k = 3
            mutual = True
            thres = 0.03

            # get dimension of matching scores
            n0, n1, n2, n3 = matching_scores.shape
            # reshape matching scores [n0, n1, n2, n3] -> [n0 * n1 * n2, n3]
            matching_scores = matching_scores.view(-1, matching_scores.shape[-1]).contiguous()
            # create mask
            non_maximum_mask1 = torch.zeros_like(matching_scores).to(matching_scores)
            # select topk scores per row
            _, src_max_ind = torch.topk(matching_scores, k=top_k, dim=-1)
            # flatten src_max_ind [n0 * n1 * n2 * top_k]
            src_max_ind = src_max_ind.view(-1).contiguous()
            # create row id
            row_id = torch.from_numpy(np.arange(matching_scores.shape[0])).to(matching_scores).long().unsqueeze(-1).repeat(1, top_k).view(-1).contiguous()
            # update the mask
            non_maximum_mask1[row_id, src_max_ind] = 1
            # reshape mask back to the original shape
            non_maximum_mask1 = non_maximum_mask1.view(n0, n1, n2, n3).contiguous()

            # change the order of row and column
            matching_scores = matching_scores.view(n0, n1, n2, n3).contiguous()
            matching_scores = matching_scores.permute(0, 1, 3, 2).contiguous().view(-1, matching_scores.shape[-1]).contiguous()
            # create mask
            non_maximum_mask2 = torch.zeros_like(matching_scores).to(matching_scores)
            # select topk scores per row
            _, tgt_max_ind = torch.topk(matching_scores, k=top_k, dim=-1)
            # select topk scores per row
            tgt_max_ind = tgt_max_ind.view(-1).contiguous()
            # create row id
            row_id = torch.from_numpy(np.arange(matching_scores.shape[0])).to(matching_scores).long().unsqueeze(-1).repeat(1, top_k).view(-1).contiguous()
            # update the mask
            non_maximum_mask2[row_id, tgt_max_ind] = 1
            # reshape mask back to the original shape
            non_maximum_mask2 = non_maximum_mask2.view(n0, n1, n3, n2).contiguous().permute(0, 1, 3, 2).contiguous()
            # change the order of row and column back to the original
            matching_scores = matching_scores.view(n0, n1, n3, n2).contiguous().permute(0, 1, 3, 2).contiguous()
            # masking out
            if not mutual:
                matching_scores = matching_scores * torch.gt(non_maximum_mask1 + non_maximum_mask2, 0.).float()
            else:
                matching_scores = matching_scores * torch.gt(non_maximum_mask1 + non_maximum_mask2, 1.).float()

            # masking out
            mask = torch.gt(matching_scores, thres).float()
            matching_scores = matching_scores * mask

            ########################################################################################
            # get correspondences
            indices = torch.nonzero(matching_scores)
            confidences = matching_scores[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]
            corr_src_pts = gathered_src_patch_xyz[indices[:, 0], indices[:, 1], indices[:, 2], :]
            corr_tgt_pts = gathered_tgt_patch_xyz[indices[:, 0], indices[:, 1], indices[:, 3], :]

            return src_feats, tgt_feats, src_point_feats, tgt_point_feats, corr_src_pts, corr_tgt_pts, confidences
        #################################################################################################