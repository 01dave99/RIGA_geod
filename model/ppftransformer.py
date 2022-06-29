import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


def MLP(channels: list, do_bn=True):
    '''
    Multi-layer perceptron
    :param channels: a list of all the applied channels of features
    :param do_bn: whether to do batch normalization
    :return: Defined MLP
    '''
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
                layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class PointNetfeat(nn.Module):
    '''
    PointNet Feature: Encode point clouds (local patches) to one-dimensional features
    '''
    def __init__(self, input_type='xyz', proj_dim=64, out_dim=64):
        super(PointNetfeat, self).__init__()
        self.input_type = input_type # can be 3-dimensional xyz, 4-dimensional ppf and 7-dimensional xyz+ppf
        self.proj_dim = proj_dim # dimension of first projection
        if input_type == 'xyz':
            init_channel = 3
        elif input_type == 'ppf':
            init_channel = 4
        elif input_type == 'mix':
            init_channel = 7
        elif input_type == 'geod':
            init_channel=1
        else:
            raise NotImplementedError

        # mlp
        self.MLPs = MLP([init_channel, proj_dim, proj_dim * 2, proj_dim * 4, proj_dim * 8])
        # final projection
        self.final_conv = nn.Conv1d(self.proj_dim * 8, out_dim, 1, bias=True)

    def forward(self, x, normalize=True):
        # mlp
        x = self.MLPs(x)
        # max-pooling, to fuse the information from each single local patch
        x = torch.max(x, 2)[0].unsqueeze(-1)
        # final projection
        x = self.final_conv(x).squeeze(-1)
        # whether to normalize the output feature
        if normalize:
            x = F.normalize(x, p=2, dim=1)
        return x


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class CrossAttention(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class PPFTrans(nn.Module):
    '''
    Transformer with PPF-like global cues for sensing global contexts beyond local vicinity
    '''
    def __init__(self, config):
        super(PPFTrans, self).__init__()
        # project the global structural cues to high dimension in a learnable manner
        self.position_proj = PointNetfeat(input_type='geod', proj_dim=64, out_dim=config.local_out_dim)
        # attention blocks
        self.blocks = config.transformer_architecture
        layers = []
        for block in self.blocks:
            # add a self-attention block
            if block == 'self':
                layers.append(CrossAttention(feature_dim=config.local_out_dim, num_heads=config.transformer_num_head))
            # add a cross-attention block
            elif block == 'cross':
                layers.append(CrossAttention(feature_dim=config.local_out_dim, num_heads=config.transformer_num_head))
            else:
                raise ValueError('Unsupported block type "{}" in `RPEConditionalTransformer`.'.format(block))
        self.layers = nn.ModuleList(layers)
        # final projection
        self.final_proj = nn.Conv1d(config.local_out_dim, config.local_out_dim, 1)

    def forward(self, src_node_geod, tgt_node_geod, src_node_feats, tgt_node_feats):

        # get current batch size
        batch_size = src_node_geod.shape[0]
        ############################################################################
        # reshape node ppf for global cues
        src_node_geod = src_node_geod.view(-1, src_node_geod.shape[-2], src_node_geod.shape[-1]).contiguous() #[B, N, N, 4] -> [B * N, N, 4]
        tgt_node_geod = tgt_node_geod.view(-1, tgt_node_geod.shape[-2], tgt_node_geod.shape[-1]).contiguous() #[B, N, N, 4] -> [B * N, N, 4]
        ############################################################################
        # project global cues encoded by ppf to global structural descriptors
        src_positional_embedding = self.position_proj(src_node_geod.permute(0, 2, 1).contiguous()) #[B, N, N, 4] -> [B * N, 4, N] -> [B * N, F]
        src_positional_embedding = src_positional_embedding.view(batch_size, -1, src_positional_embedding.shape[-1]).contiguous() #[B * N, F] -> [B, N, F]
        tgt_positional_embedding = self.position_proj(tgt_node_geod.permute(0, 2, 1).contiguous()) #[B, N, N, 4] -> [B * N, 4, N] -> [B * N, F]
        tgt_positional_embedding = tgt_positional_embedding.view(batch_size, -1, tgt_positional_embedding.shape[-1]).contiguous() #[B * N, F] -> [B, N, F]
        ############################################################################
        # reshape
        src_node_feats = src_node_feats.permute(0, 2, 1).contiguous() #[B, N, F] -> [B, F, N]
        tgt_node_feats = tgt_node_feats.permute(0, 2, 1).contiguous() #[B, N, F] -> [B, F, N]

        src_positional_embedding = src_positional_embedding.permute(0, 2, 1).contiguous()  # [B, N, F] -> [B, F, N]
        tgt_positional_embedding = tgt_positional_embedding.permute(0, 2, 1).contiguous()  # [B, N, F] -> [B, F, N]
        ############################################################################
        for i in range(len(self.blocks)):
            if self.blocks[i] == 'self':
                ###########################################################
                # inform local descriptors with global cues by adding ppf-encoded global cues to local geometric descriptors
                src_node_feats = src_node_feats + src_positional_embedding
                tgt_node_feats = tgt_node_feats + tgt_positional_embedding
                ###########################################################
                # aggregate learned intra-frame global contexts via attention mechanism
                src_node_feats = src_node_feats + self.layers[i](src_node_feats, src_node_feats)
                tgt_node_feats = tgt_node_feats + self.layers[i](tgt_node_feats, tgt_node_feats)
            elif self.blocks[i] == 'cross':
                ###########################################################
                # aggregate learned inter-frame global contexts via attention mechanism
                src_node_feats = src_node_feats + self.layers[i](src_node_feats, tgt_node_feats)
                tgt_node_feats = tgt_node_feats + self.layers[i](tgt_node_feats, src_node_feats)

        # final feature projection
        src_node_feats = self.final_proj(src_node_feats).permute(0, 2, 1).contiguous()
        tgt_node_feats = self.final_proj(tgt_node_feats).permute(0, 2, 1).contiguous()
        return src_node_feats, tgt_node_feats, src_positional_embedding.permute(0, 2, 1).contiguous(), tgt_positional_embedding.permute(0, 2, 1).contiguous()