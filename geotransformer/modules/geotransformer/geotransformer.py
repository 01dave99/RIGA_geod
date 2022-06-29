import numpy as np
import torch
import torch.nn as nn


from geotransformer.modules.transformer import SinusoidalPositionalEmbedding, RPEConditionalTransformer
from einops import rearrange

class GeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, sigma_a, angle_k, reduction_a='max'):
        super(GeometricStructureEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)

        self.reduction_a = reduction_a
        if self.reduction_a not in ['max', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

    @torch.no_grad()
    def get_embedding_indices(self,node_geod):
        r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            node_geod: numpy.ndarray (B, N, 3), input geodesic distnaces

        Returns:
            d_indices: torch.FloatTensor (B, N, N), distance embedding indices
            a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
        """
        #batch_size, num_point, _ = points.shape

        dist_map = torch.tensor(node_geod)  # (B, N, N)
        d_indices=dist_map
        """
        k = self.angle_k
        knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k, 3)  # (B, N, k, 3)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
        angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
        a_indices = angles * self.factor_a
        """
        return d_indices #, a_indices

    def forward(self,node_geod):
        d_indices = self.get_embedding_indices(node_geod)

        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)

        #a_embeddings = self.embedding(a_indices)
        #a_embeddings = self.proj_a(a_embeddings)
        #if self.reduction_a == 'max':
        #    a_embeddings = a_embeddings.max(dim=3)[0]
        #else:
        #    a_embeddings = a_embeddings.mean(dim=3)

        embeddings = d_embeddings #+ a_embeddings

        return embeddings


class GeometricTransformer(nn.Module):
    def __init__(
        self,
        config,
        dropout=None,
        activation_fn='ReLU',
        reduction_a='max',
    ):
        r"""Geometric Transformer (GeoTransformer).

        Args:
            input_dim: input feature dimension
            output_dim: output feature dimension
            hidden_dim: hidden feature dimension
            num_heads: number of head in transformer
            blocks: list of 'self' or 'cross'
            sigma_d: temperature of distance
            sigma_a: temperature of angles
            angle_k: number of nearest neighbors for angular embedding
            activation_fn: activation function
            reduction_a: reduction mode of angular embedding ['max', 'mean']
        """
        


        super(GeometricTransformer, self).__init__()

        self.input_dim=config.local_out_dim
        self.output_dim=config.local_out_dim
        self.hidden_dim=config.transformer_feats_dim
        self.num_heads=config.transformer_num_head
        self.blocks=config.transformer_architecture
        self.sigma_d=config.transformer_sigma_d
        self.sigma_a=config.transformer_sigma_a
        self.angle_k=config.transformer_angle_k


        self.embedding = GeometricStructureEmbedding(self.hidden_dim, self.sigma_d, self.sigma_a, self.angle_k, reduction_a=reduction_a)

        self.in_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.transformer = RPEConditionalTransformer(
            self.blocks, self.hidden_dim, self.num_heads, dropout=dropout, activation_fn=activation_fn
        )
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(
        self,
        src_feats,
        tgt_feats,
        src_node_geod,
        tgt_node_geod,
        src_masks=None,
        tgt_masks=None,
    ):
        r"""Geometric Transformer

        Args:
            ref_points (Tensor): (B, N, 3)
            src_points (Tensor): (B, M, 3)
            ref_feats (Tensor): (B, N, C)
            src_feats (Tensor): (B, M, C)
            ref_masks (Optional[BoolTensor]): (B, N)
            src_masks (Optional[BoolTensor]): (B, M)

        Returns:
            ref_feats: torch.Tensor (B, N, C)
            src_feats: torch.Tensor (B, M, C)
        """

        src_embeddings = self.embedding(src_node_geod)
        tgt_embeddings = self.embedding(tgt_node_geod)

        src_feats = self.in_proj(src_feats)
        tgt_feats = self.in_proj(tgt_feats)

        src_feats, tgt_feats = self.transformer(
            src_feats,
            tgt_feats,
            src_embeddings,
            tgt_embeddings,
            masks0=src_masks,
            masks1=tgt_masks,
        )

        src_feats = self.out_proj(src_feats)
        tgt_feats = self.out_proj(tgt_feats)

        return src_feats, tgt_feats
