import torch.utils.data as data
import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from dataset.common import collect_local_neighbors, build_ppf_patch, farthest_point_subsampling,\
    point2node_sampling, calc_patch_overlap_ratio, get_square_distance_matrix, calc_ppf_cpu, sample_gt_node_correspondence, calc_gt_patch_correspondence, calc_geod
from lib.utils import to_o3d_pcd
import open3d as o3d
from geodist.geod_dist_calculation import voxels2graph, calcPointPairDistances, point_to_vertex

class TDMatchDataset(data.Dataset):
    '''
    Load subsampled coordinates, relative rotation and translation
    Output (torch.Tensor):
    src_pcd: (N, 3) source point cloud
    tgt_pcd: (M, 3) target point cloud
    src_node_xyz: (n, 3) nodes sparsely sampled from source point cloud
    tgt_node_xyz: (m, 3) nodes sparsely sampled from target point cloud
    rot: (3, 3)
    trans: (3, 1)
    correspondences: (?, 3)
    '''

    def __init__(self, infos, config, data_augmentation=True):
        super(TDMatchDataset, self).__init__()
        # information of data
        self.infos = infos
        # root dir
        self.base_dir = config.root
        # whether to do data augmentation
        self.data_augmentation = data_augmentation
        # whether to resample the point cloud
        self.resample = config.resample
        # configurations
        self.config = config
        # the number of nodes sampled for each frame of point cloud
        self.patch_per_frame = config.patch_per_frame
        # the number of point per patch (for local encoding)
        self.point_per_patch = config.point_per_patch
        # the number of point per patch (for coarse-to-fine correspondences)
        self.point_per_p2n = config.point_per_p2n
        # the way to represent local patches, should be in ['xyz', 'ppf', 'mix]
        self.input_type = config.input_type
        # whether to de-center each patch
        self.decentralization = config.decentralization
        # the radius of each local patch
        self.patch_vicinity = config.patch_vicinity
        # factor used to control the maximum rotation during data augmentation
        self.rot_factor = 1.
        # maximum noise used in data augmentation
        self.augment_noise = config.augment_noise
        # the maximum number allowed in each single frame of point cloud
        self.points_lim = 30000
        # the number of neighboring nodes used in feature interpolation
        self.interpolate_neighbors = config.interpolate_neighbors
        # the threshold used to determine correspondences between nodes
        self.overlap_thres = config.overlap_thres
        # the threshold of distance used to determine correspondences between points
        self.overlap_radius = config.overlap_radius
        # the number of selected node correspondences
        self.max_corr = config.max_corr
        # can be in ['train', 'val', 'test']
        self.mode = config.mode
        # original benchmark or rotated benchmark
        self.rotated = config.rotated
        #voxel size
        self.voxel_size = config.voxel_size

    def __getitem__(self, index):
        # get gt transformation
        rot = self.infos['rot'][index]
        trans = self.infos['trans'][index]
        # get original input point clouds
        src_path = os.path.join(self.base_dir, self.infos['src'][index])
        tgt_path = os.path.join(self.base_dir, self.infos['tgt'][index])
        src_pcd = torch.load(src_path)
        tgt_pcd = torch.load(tgt_path)

        ##################################################################################################
        # if we get too many points, we do random down-sampling
        if src_pcd.shape[0] > self.points_lim:
            idx = np.random.permutation(src_pcd.shape[0])[:self.points_lim]
            src_pcd = src_pcd[idx]

        if tgt_pcd.shape[0] > self.points_lim:
            idx = np.random.permutation(tgt_pcd.shape[0])[:self.points_lim]
            tgt_pcd = tgt_pcd[idx]

        ##################################################################################################
        # whether to augment data for training / to rotate data for testing
        if self.data_augmentation:
            # rotate the point cloud
            euler_ab = np.random.rand(3) * np.pi * 2. / self.rot_factor  # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if (np.random.rand(1)[0] > 0.5):
                src_pcd = np.matmul(rot_ab, src_pcd.T).T

                rot = np.matmul(rot, rot_ab.T)
            else:
                tgt_pcd = np.matmul(rot_ab, tgt_pcd.T).T

                rot = np.matmul(rot_ab, rot)
                trans = np.matmul(rot_ab, trans)
            # add noise
            src_pcd += (np.random.rand(src_pcd.shape[0], 3) - 0.5) * self.augment_noise
            tgt_pcd += (np.random.rand(tgt_pcd.shape[0], 3) - 0.5) * self.augment_noise
        # wheter test on rotated benchmark
        elif self.rotated:
            # rotate the point cloud
            euler_ab = np.random.rand(3) * np.pi * 2. / self.rot_factor  # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if (np.random.rand(1)[0] > 0.5):
                src_pcd = np.matmul(rot_ab, src_pcd.T).T

                rot = np.matmul(rot, rot_ab.T)
            else:
                tgt_pcd = np.matmul(rot_ab, tgt_pcd.T).T

                rot = np.matmul(rot_ab, rot)
                trans = np.matmul(rot_ab, trans)
        else:
            pass

        if (trans.ndim == 1):
            trans = trans[:, None]
        ##################################################################################################
        # Normal estimation
        o3d_src_pcd = to_o3d_pcd(src_pcd)
        o3d_tgt_pcd = to_o3d_pcd(tgt_pcd)
        o3d_src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
        src_normals = np.asarray(o3d_src_pcd.normals)
        o3d_tgt_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
        tgt_normals = np.asarray(o3d_tgt_pcd.normals)

        #################################
        # generate patches
        # use farthest point sampling (FPS) to sub-sample nodes out of the input point clouds
        src_node_inds, src_node_xyz = farthest_point_subsampling(src_pcd, self.patch_per_frame)
        tgt_node_inds, tgt_node_xyz = farthest_point_subsampling(tgt_pcd, self.patch_per_frame)
        # collect local vicinity of nodes for encoding the local geometric descriptors (radius-based)
        src_patch_inds, src_patch_xyz, src_patch_masks = collect_local_neighbors(src_pcd, src_node_xyz, self.patch_vicinity, self.point_per_patch)
        tgt_patch_inds, tgt_patch_xyz, tgt_patch_masks = collect_local_neighbors(tgt_pcd, tgt_node_xyz, self.patch_vicinity, self.point_per_patch)
        # collect local vicinity of nodes for coarse-to-fine correspondences (point to node assignment)
        src_p2n_inds, src_p2n_xyz, src_p2n_masks = point2node_sampling(src_pcd, src_node_xyz, src_node_inds, self.point_per_p2n)
        tgt_p2n_inds, tgt_p2n_xyz, tgt_p2n_masks = point2node_sampling(tgt_pcd, tgt_node_xyz, tgt_node_inds, self.point_per_p2n)
        # calculate ground truth node correspondence for training, accoording to the overlap ratios between the vicinity areas of node pairs
        row_major_overlap_ratio, col_major_overlap_ratio = calc_patch_overlap_ratio(src_node_xyz, src_p2n_xyz, src_p2n_masks, tgt_node_xyz, tgt_p2n_xyz, tgt_p2n_masks, rot, trans)
        pos_mask = (row_major_overlap_ratio + col_major_overlap_ratio) / 2.

        if self.mode != 'test':
            # sample ground truth corresponding nodes for training
            src_corr_node, tgt_corr_node = sample_gt_node_correspondence(pos_mask, corr_num=self.max_corr, thres=self.overlap_thres)
            # safety check, whether the threshold is too high
            if src_corr_node is None:
                src_corr_node, tgt_corr_node = sample_gt_node_correspondence(pos_mask, corr_num=self.max_corr, thres=0.)
            # collect selected patches indices,  corresponding masks and coordinates
            sel_src_p2n_inds, sel_src_p2n_masks = src_p2n_inds[src_corr_node, :], src_p2n_masks[src_corr_node, :]
            sel_tgt_p2n_inds, sel_tgt_p2n_masks = tgt_p2n_inds[tgt_corr_node, :], tgt_p2n_masks[tgt_corr_node, :]
            sel_src_p2n_xyz, sel_tgt_p2n_xyz = src_p2n_xyz[src_corr_node, ...], tgt_p2n_xyz[tgt_corr_node, ...]

            gt_patch_corr = calc_gt_patch_correspondence(sel_src_p2n_xyz, sel_tgt_p2n_xyz, rot, trans, distance_thres=self.overlap_radius)

        else:
            # we don't use the following information for testing, so they can be arbitrary
            sel_src_p2n_inds, sel_src_p2n_masks = src_p2n_inds, src_p2n_masks
            sel_tgt_p2n_inds, sel_tgt_p2n_masks = tgt_p2n_inds, tgt_p2n_masks
            #sel_src_p2n_xyz, sel_tgt_p2n_xyz = src_p2n_xyz, tgt_p2n_xyz
            gt_patch_corr = np.zeros(1)

        # decide the type of input
        if self.input_type in ['ppf', 'mix']:
            # calculate the ppf inside each node vicinity

            src_patch_ppf, src_node_normal = build_ppf_patch(src_pcd, src_node_inds, src_patch_inds, point_normals=src_normals, view_point=np.array([[0, 0, 0]]))
            tgt_patch_ppf, tgt_node_normal = build_ppf_patch(tgt_pcd, tgt_node_inds, tgt_patch_inds, point_normals=tgt_normals, view_point=np.array([[0, 0, 0]]))

            #calculate geodesic ditances for source and target
            geodesic_src=calcPointPairDistances(src_node_xyz,src_pcd,self.voxel_size)
            geodesic_tgt=calcPointPairDistances(tgt_node_xyz,tgt_pcd,self.voxel_size)
            #######################################################
            # Create list of global neighbours
            indices = np.arange(src_node_xyz.shape[0])[np.newaxis, :]
            neighbors = indices.repeat(src_node_xyz.shape[0], axis=0)
            delete_index = (np.arange(src_node_xyz.shape[0]) * (src_node_xyz.shape[0] + 1)).astype(np.int32)
            neighbors = np.reshape(neighbors, -1)
            neighbors = np.reshape(neighbors, (src_node_xyz.shape[0], src_node_xyz.shape[0]))
            # use geodesic distance to encode the global contexts for each node in a global perspective
            src_node_geod = calc_geod(src_node_xyz, src_node_normal, src_node_xyz, src_node_normal, geodesic_src, neighbors=neighbors) #(N,N)
            tgt_node_geod = calc_geod(tgt_node_xyz, tgt_node_normal, tgt_node_xyz, tgt_node_normal, geodesic_tgt, neighbors=neighbors) #(N,N)


        if self.input_type in ['xyz', 'mix']:
            # de-center each patch
            src_patch_xyz_decent = src_patch_xyz - np.mean(src_patch_xyz, axis=-2, keepdims=True)
            tgt_patch_xyz_decent = tgt_patch_xyz - np.mean(tgt_patch_xyz, axis=-2, keepdims=True)

        if self.input_type == 'xyz':
            src_patch, tgt_patch = src_patch_xyz_decent, tgt_patch_xyz_decent
        elif self.input_type == 'ppf':
            src_patch, tgt_patch = src_patch_ppf, tgt_patch_ppf
        elif self.input_type == 'mix':
            src_patch = np.concatenate([src_patch_xyz_decent, src_patch_ppf], axis=-1)
            tgt_patch = np.concatenate([tgt_patch_xyz_decent, tgt_patch_ppf], axis=-1)
        else:
            raise NotImplementedError
        ################################################################################
        # For consecutive feature interpolation
        # for each point, choose its k nearest nodes in geometry space
        src_point_node_dist = get_square_distance_matrix(src_pcd, src_node_xyz)
        src_k_node_neighbor_inds = np.argpartition(src_point_node_dist, axis=1, kth=self.interpolate_neighbors)[:, :self.interpolate_neighbors]
        tgt_point_node_dist = get_square_distance_matrix(tgt_pcd, tgt_node_xyz)
        tgt_k_node_neighbor_inds = np.argpartition(tgt_point_node_dist, axis=1, kth=self.interpolate_neighbors)[:, :self.interpolate_neighbors]
        ################################################################################
        return src_pcd.astype(np.float32), tgt_pcd.astype(np.float32), \
               src_node_xyz.astype(np.float32), tgt_node_xyz.astype(np.float32), \
               src_node_geod.astype(np.float32), tgt_node_geod.astype(np.float32),\
               src_patch_xyz.astype(np.float32), tgt_patch_xyz.astype(np.float32), \
               src_patch.astype(np.float32), tgt_patch.astype(np.float32),\
               sel_src_p2n_inds.astype(np.int32), sel_tgt_p2n_inds.astype(np.int32), \
               sel_src_p2n_masks.astype(np.float32), sel_tgt_p2n_masks.astype(np.float32),\
               gt_patch_corr.astype(np.float32),\
               rot.astype(np.float32), trans.astype(np.float32), \
               pos_mask.astype(np.float32),\
               src_k_node_neighbor_inds, tgt_k_node_neighbor_inds

    def __len__(self):
        return len(self.infos['rot'])


