import random, time, re
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn


def setup_seed(seed):
    '''
    fix random seed for deterministic training
    :param seed: selected seed for deterministic training
    :return: None
    '''
    torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def natural_key(string_):
    """
    Sort strings by numbers in the name
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def to_tsfm(rot, trans):
    '''
    Transfer rotation and translation to transformation
    :param rot: rotation matrix of numpy.ndarray in shape [3, 3]
    :param trans: translation vector of numpy.ndarray in shape[3, 1]
    :return: Transformation matrix of numpy.ndarray in shape[4, 4]
    '''
    tsfm = np.eye(4)
    tsfm[:3, :3] = rot
    tsfm[:3, 3] = trans.flatten()
    return tsfm


def to_o3d_pcd(pcd):
    '''
    Transfer a point cloud of numpy.ndarray to open3d point cloud
    :param pcd: point cloud of numpy.ndarray in shape[N, 3]
    :return: open3d.geometry.PointCloud()
    '''
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(pcd)
    return pcd_


def get_correspondences(src_pcd, tgt_pcd, trans, search_voxel_size, K=None):
    '''
    Get correspondences between a pair of point clouds, given the ground truth transformation
    :param src_pcd: source point cloud of open3d.geomerty.PointCloud in shape[N, 3]
    :param tgt_pcd: target point cloud of open3d.geomerty.PointCloud in shape[M, 3]
    :param trans: transformation matrix of numpy.ndarray in shape[4, 4]
    :param search_voxel_size: distrance threshold within which two points are considered as a correspondence
    :param K: if K is not None, only return K corresponding points for each point
    :return: correspondences of torch.tensor in shape[?, 2]
    '''

    src_pcd.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)

    correspondences = []
    for i, point in enumerate(src_pcd.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])

    correspondences = np.array(correspondences)
    correspondences = torch.from_numpy(correspondences)
    return correspondences


def matching_descriptors(src_desc, tgt_desc, mutual=False, major=None):
    '''
    Matching based on descriptors, return correspondences
    :param src_desc: descriptors of source point cloud
    :param tgt_desc: descriptors of target point cloud
    :param mutual: wheter to perform mutual selection
    :return: Extracted correspondences of numpy.ndarray in shape[n, 2]
    '''
    assert major in ['row', 'col'] or major is None
    distances = square_distance(torch.from_numpy(src_desc[np.newaxis, :]), torch.from_numpy(tgt_desc[np.newaxis, :]))[0].numpy()
    row_idx = np.arange(src_desc.shape[0])
    row_major_idx = np.argmin(distances, axis=1)
    col_idx = np.arange(tgt_desc.shape[0])
    col_major_idx = np.argmin(distances, axis=0)
    if not mutual:
        if major == 'row':
            correspondence = np.concatenate((row_idx[:, np.newaxis], row_major_idx[:, np.newaxis]), axis=1)
        elif major == 'col':
            correspondence = np.concatenate((col_major_idx[:, np.newaxis], col_idx[:, np.newaxis]), axis=1)
        else:
            row_major_mask = np.zeros_like(distances)
            row_major_mask[row_idx, row_major_idx] = 1
            col_major_mask = np.zeros_like(distances)
            col_major_mask[col_major_idx, col_idx] = 1
            mask = np.logical_or(row_major_mask > 0, col_major_mask > 0)
            correspondence = np.nonzero(mask)
            correspondence = np.concatenate((correspondence[0][:, np.newaxis], correspondence[1][:, np.newaxis]),
                                            axis=-1)
        return correspondence
    else:
        row_major_mask = np.zeros_like(distances)
        row_major_mask[row_idx, row_major_idx] = 1
        col_major_mask = np.zeros_like(distances)
        col_major_mask[col_major_idx, col_idx] = 1
        mask = np.logical_and(row_major_mask > 0, col_major_mask > 0)
        correspondence = np.nonzero(mask)
        correspondence = np.concatenate((correspondence[0][:, np.newaxis], correspondence[1][:, np.newaxis]), axis=-1)
        return correspondence


def square_distance(src, tgt):
    '''
    Calculate Euclidean distance between every two points, for batched point clouds in torch.tensor
    :param src: source point cloud in shape [B, N, 3]
    :param tgt: target point cloud in shape [B, M, 3]
    :return: Squared Euclidean distance matrix in torch.tensor of shape[B, N, M]
    '''
    B, N, _ = src.shape
    _, M, _ = tgt.shape
    dist = -2. * torch.matmul(src, tgt.permute(0, 2, 1).contiguous())
    dist += torch.sum(src ** 2, dim=-1).unsqueeze(-1)
    dist += torch.sum(tgt ** 2, dim=-1).unsqueeze(-2)

    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist


def weighted_procrustes(src_points, tgt_points, weights=None, weight_thresh=0., eps=1e-5, return_transform=False):
    r"""
    Compute rigid transformation from `src_points` to `tgt_points` using weighted SVD.

    Modified from [PointDSC](https://github.com/XuyangBai/PointDSC/blob/master/models/common.py).

    :param src_points: torch.Tensor (batch_size, num_corr, 3) or (num_corr, 3)
    :param tgt_points: torch.Tensor (batch_size, num_corr, 3) or (num_corr, 3)
    :param weights: torch.Tensor (batch_size, num_corr) or (num_corr,) (default: None)
    :param weight_thresh: float (default: 0.)
    :param eps: float (default: 1e-5)
    :param return_transform: bool (default: False)

    :return R: torch.Tensor (batch_size, 3, 3) or (3, 3)
    :return t: torch.Tensor (batch_size, 3) or (3,)
    :return transform: torch.Tensor (batch_size, 4, 4) or (4, 4)
    """
    if src_points.ndim == 2:
        src_points = src_points.unsqueeze(0)
        tgt_points = tgt_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_points.shape[0]
    if weights is None:
        weights = torch.ones_like(src_points[:, :, 0])
    weights = torch.where(torch.lt(weights, weight_thresh), torch.zeros_like(weights), weights)
    weights_norm = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)

    src_centroid = torch.sum(src_points * weights_norm.unsqueeze(2), dim=1, keepdim=True)
    tgt_centroid = torch.sum(tgt_points * weights_norm.unsqueeze(2), dim=1, keepdim=True)
    src_points_centered = src_points - src_centroid
    tgt_points_centered = tgt_points - tgt_centroid

    W = torch.diag_embed(weights)
    H = src_points_centered.permute(0, 2, 1) @ W @ tgt_points_centered
    U, _, V = torch.svd(H)  # H = USV^T
    Ut, V = U.transpose(1, 2), V
    eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
    eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    R = V @ eye @ Ut

    t = tgt_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
    t = t.squeeze(2)

    if return_transform:
        transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t
        if squeeze_first:
            transform = transform.squeeze(0)
        return transform
    else:
        if squeeze_first:
            R = R.squeeze(0)
            t = t.squeeze(0)
        return R, t


def sinkhorn(log_alpha, n_iters: int = 5, slack: bool = True, eps: float = -1) -> torch.Tensor:
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1
    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.
    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)
    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    # Sinkhorn iterations
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

    return log_alpha



def interpolate(weights, points):
    '''
    Do interpolation based on provided weights
    :param weights: interpolation weights in torch.tensor of shape [b, n, m]
    :param points: points to be interpolated, in torch.tensor of shape [b, m, 3]
    :return: Interpolated coordinates in torch.tensor of shape[b, n, 3]
    '''
    weights = torch.unsqueeze(weights, dim=-1).expand(-1, -1, -1, 3) # [b, n, m] -> [b, n, m, 3]
    points = torch.unsqueeze(points, dim=1).expand(-1, weights.shape[1], -1, -1) #[b, m, 3] -> [b, n, m, 3]
    interpolation = torch.sum(weights * points, dim=-2)
    return interpolation


def soft_assignment(src_xyz, src_feats, tgt_xyz, tgt_feats):
    '''
    Differentiablely compute correspondences between points, return with weights.
    :param src_xyz: Torch tensor in shape[b, n, 3]
    :param src_feats: Torch tensor in shape[b, n, c]
    :param tgt_xyz: Torch tensor in shape[b, n, 3]
    :param tgt_feats: Torch tensor in shape[b, n, c]
    :return: src2tgt_assignment_confidence: confidence of each corresponding point, torch.tensor in shape[b, n]
             src2tgt_interpolated_xyz: interpolated xyz coordinates in tgt space, torch.tensor in shape[b, n, 3]
             tgt2src_assignment_confidence: confidence of each corresponding point, torch.tensor in shape[b, n]
             tgt2src_interpolated_xyz: interpolated xyz coordinates in src space, torch.tensor in shape[b, n]
    '''
    feat_distance = torch.sqrt(square_distance(src_feats, tgt_feats))
    feat_similarity = 1. / (1e-8 + feat_distance) #similarity matrix in shape [b, n, n]
    # calculate src's corresponding weights and confidence in tgt
    src2tgt_assignment_weights = feat_similarity / torch.sum(feat_similarity, dim=-1, keepdim=True) #row-normalized similarity matrix in shape [b, n, n]
    src2tgt_assignment_max_sim = torch.max(feat_similarity, dim=-1)[0] #row-major max similarity in shape [b, n]
    src2tgt_assignment_confidence = src2tgt_assignment_max_sim / torch.sum(src2tgt_assignment_max_sim, dim=-1, keepdim=True) #normalized confidence of softassignment in shape [b, n]
    src2tgt_interpolated_xyz = interpolate(src2tgt_assignment_weights, tgt_xyz)
    # calculate tgt's corresponding weights and confidence in src
    tgt2src_assignment_weights = feat_similarity / torch.sum(feat_similarity, dim=1, keepdim=True)  # column-normalized similarity matrix in shape [b, n, n]
    tgt2src_assignment_max_sim = torch.max(feat_similarity, dim=1)[0]  # row-major max similarity in shape [b, n]
    tgt2src_assignment_confidence = tgt2src_assignment_max_sim / torch.sum(tgt2src_assignment_max_sim, dim=-1, keepdim=True)  # normalized confidence of softassignment in shape [b, n]
    tgt2src_interpolated_xyz = interpolate(tgt2src_assignment_weights, src_xyz)
    return src2tgt_assignment_confidence, src2tgt_interpolated_xyz, tgt2src_assignment_confidence, tgt2src_interpolated_xyz


def get_geometric_structure_embeddings(points, angle_k=3):

    batch_size, num_point, _ = points.shape

    dist_map = torch.sqrt(square_distance(points, points))  # (B, N, N)

    knn_indices = dist_map.topk(k=angle_k + 1, dim=2, largest=False)[1]  # (B, N, k)
    knn_indices = knn_indices[:, :, 1:]
    knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, angle_k, 3)  # (B, N, k, 3)
    expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
    knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
    ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
    anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
    ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, angle_k, 3)  # (B, N, N, k, 3)
    anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, angle_k, 3)  # (B, N, N, k, 3)
    sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
    cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
    angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)

    return dist_map, angles



########################
# utils classes
########################

class AverageMeter(object):
    '''
    A class computes and stores the average and current values
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.sq_sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sq_sum += val ** 2 * n
        self.var = self.sq_sum / self.count - self.avg ** 2


class Timer(object):
    '''
    A simple timer
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.avg = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.avg = self.total_time / self.calls
        if average:
            return self.avg
        else:
            return self.diff


class Logger(object):
    '''
    A simple logger
    '''

    def __init__(self, path):
        self.path = path
        self.fw = open(self.path + '/log', 'a')

    def write(self, text):
        self.fw.write(text)
        self.fw.flush()

    def close(self):
        self.fw.close()
