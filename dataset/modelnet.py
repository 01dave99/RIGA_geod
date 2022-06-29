from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
import os, h5py
import glob
from torch.utils.data import Dataset
import numpy as np
from scipy.spatial.distance import minkowski
from dataset.common import collect_local_neighbors, build_ppf_patch, farthest_point_subsampling, calc_patch_overlap_ratio, x_axis_crop, random_point_subsampling,\
    get_square_distance_matrix, point2node_sampling, calc_ppf_cpu, calc_gt_patch_correspondence, sample_gt_node_correspondence, get_batched_square_distance_matrix
from lib.utils import to_o3d_pcd, get_correspondences, to_tsfm


def download():
    BASE_DIR = './'
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = './'
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label



def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def farthest_subsample_points(pointcloud1, pointcloud2, num_subsampled_points=768):
    pointcloud1 = pointcloud1.T
    pointcloud2 = pointcloud2.T
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1)
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud2)
    random_p2 = random_p1 #np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))
    return random_p1, pointcloud1[idx1, :].T, pointcloud2[idx2, :].T


class ModelNet40(Dataset):
    def __init__(self, configs, num_points, num_subsampled_points=768, partition='train', gaussian_noise=False, unseen=False, rot_factor=4, category=None):
        super(ModelNet40, self).__init__()
        self.config = configs
        self.data, self.label = load_data(partition)
        if category is not None:
            self.data = self.data[self.label == category]
            self.lable = self.label[self.label == category]
        self.num_points = num_points
        self.num_subsampled_points = num_subsampled_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.rot_factor = rot_factor
        if num_points != num_subsampled_points:
            self.subsampled = True
        else:
            self.subsampled = False
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label >= 20]
                self.label = self.label[self.label >= 20]
            elif self.partition == 'train':
                self.data = self.data[self.label < 20]
                self.label = self.label[self.label < 20]

        self.sample_patch_per_frame = configs.sample_patch_per_frame
        self.patch_vicinity = configs.patch_vicinity
        self.point_per_patch = configs.point_per_patch
        self.point_per_p2n = configs.point_per_p2n

        self.input_type = configs.input_type
        self.interpolate_neighbors = configs.interpolate_neighbors

        self.overlap_thres = configs.overlap_thres
        self.overlap_radius = configs.overlap_radius
        self.max_corr = configs.max_corr
        self.mode = configs.mode

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        if self.partition != 'train':
            np.random.seed(item)

        anglex = np.random.uniform() * np.pi / self.rot_factor
        angley = np.random.uniform() * np.pi / self.rot_factor
        anglez = np.random.uniform() * np.pi / self.rot_factor
        #cosx = np.cos(anglex)
        #cosy = np.cos(angley)
        #cosz = np.cos(anglez)
        #sinx = np.sin(anglex)
        #siny = np.sin(angley)
        #sinz = np.sin(anglez)
        #Rx = np.array([[1, 0, 0],
        #               [0, cosx, -sinx],
        #               [0, sinx, cosx]])
        #Ry = np.array([[cosy, 0, siny],
        #               [0, 1, 0],
        #               [-siny, 0, cosy]])
        #Rz = np.array([[cosz, -sinz, 0],
        #               [sinz, cosz, 0],
        #               [0, 0, 1]])
        #R_ab = Rx.dot(Ry).dot(Rz)
        #R_ba = R_ab.T

        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        #translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T
        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)
        rot = rotation_ab.as_matrix()
        trans = translation_ab
        #euler_ab = np.asarray([anglez, angley, anglex])
        #euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        if self.gaussian_noise:
            pointcloud1 = jitter_pointcloud(pointcloud1)
            pointcloud2 = jitter_pointcloud(pointcloud2)

        if self.subsampled:
            view_point, pointcloud1, pointcloud2 = farthest_subsample_points(pointcloud1, pointcloud2,
                                                                 num_subsampled_points=self.num_subsampled_points)
        if (trans.ndim == 1):
            trans = trans[:, None]
        pointcloud1 = pointcloud1.T
        pointcloud2 = pointcloud2.T

        tsfm = to_tsfm(rot, trans)
        #correspondences = get_correspondences(to_o3d_pcd(pointcloud1), to_o3d_pcd(pointcloud2), tsfm, 0.0375, K=None)
        src_node_inds, src_node_xyz = farthest_point_subsampling(pointcloud1, self.sample_patch_per_frame)
        tgt_node_inds, tgt_node_xyz = farthest_point_subsampling(pointcloud2, self.sample_patch_per_frame)

        src_patch_inds, src_patch_xyz, src_patch_masks = collect_local_neighbors(pointcloud1, src_node_xyz, self.patch_vicinity, self.point_per_patch)
        tgt_patch_inds, tgt_patch_xyz, tgt_patch_masks = collect_local_neighbors(pointcloud2, tgt_node_xyz, self.patch_vicinity, self.point_per_patch)

        #src_patch_inds, src_patch_xyz, src_patch_masks = point2node_sampling(src_pcd, src_node_xyz, self.point_per_patch)
        #tgt_patch_inds, tgt_patch_xyz, tgt_patch_masks = point2node_sampling(tgt_pcd, tgt_node_xyz, self.point_per_patch)

        #########################################################
        # Drop and shuffle patches
        #########################################################
        #src_sel_inds = np.random.choice(self.sample_patch_per_frame, self.patch_per_frame, replace=False)
        #src_node_inds, src_node_xyz = src_node_inds[src_sel_inds], src_node_xyz[src_sel_inds]
        #src_patch_inds, src_patch_xyz, src_patch_masks = src_patch_inds[src_sel_inds], src_patch_xyz[src_sel_inds], src_patch_masks[src_sel_inds]
        #tgt_sel_inds = np.random.choice(self.sample_patch_per_frame, self.patch_per_frame, replace=False)
        #tgt_node_inds, tgt_node_xyz = tgt_node_inds[tgt_sel_inds], tgt_node_xyz[tgt_sel_inds]
        #tgt_patch_inds, tgt_patch_xyz, tgt_patch_masks = tgt_patch_inds[tgt_sel_inds], tgt_patch_xyz[tgt_sel_inds], tgt_patch_masks[tgt_sel_inds]
        ###########################################################################################################################################################################################3

        src_p2n_inds, src_p2n_xyz, src_p2n_masks = point2node_sampling(pointcloud1, src_node_xyz, src_node_inds,
                                                                       self.point_per_p2n)
        tgt_p2n_inds, tgt_p2n_xyz, tgt_p2n_masks = point2node_sampling(pointcloud2, tgt_node_xyz, tgt_node_inds,
                                                                       self.point_per_p2n)
        row_major_overlap_ratio, col_major_overlap_ratio = calc_patch_overlap_ratio(src_node_xyz, src_p2n_xyz,
                                                                                    src_p2n_masks, tgt_node_xyz,
                                                                                    tgt_p2n_xyz,
                                                                                    tgt_p2n_masks, rot, trans)
        rot_src_patch = np.matmul(src_p2n_xyz, rot.T) + trans.T

        #dis = np.sqrt(get_square_distance_matrix(rot_src_patch[1], tgt_p2n_xyz[3]))

        pos_mask = (row_major_overlap_ratio + col_major_overlap_ratio) / 2.
        #if self.mode != 'test':
        src_corr_node, tgt_corr_node = sample_gt_node_correspondence(pos_mask, corr_num=self.max_corr,
                                                                     thres=self.overlap_thres)
        if self.mode != 'test':
            sel_src_p2n_inds, sel_src_p2n_masks = src_p2n_inds[src_corr_node, :], src_p2n_masks[src_corr_node, :]
            sel_tgt_p2n_inds, sel_tgt_p2n_masks = tgt_p2n_inds[tgt_corr_node, :], tgt_p2n_masks[tgt_corr_node, :]
            sel_src_p2n_xyz, sel_tgt_p2n_xyz = src_p2n_xyz[src_corr_node, ...], tgt_p2n_xyz[tgt_corr_node, ...]

            gt_patch_corr = calc_gt_patch_correspondence(sel_src_p2n_xyz, sel_tgt_p2n_xyz, rot, trans,
                                                         distance_thres=self.overlap_radius)
        else:
            sel_src_p2n_inds, sel_src_p2n_masks = src_p2n_inds, src_p2n_masks
            sel_tgt_p2n_inds, sel_tgt_p2n_masks = tgt_p2n_inds, tgt_p2n_masks
            sel_src_p2n_xyz, sel_tgt_p2n_xyz = src_p2n_xyz, tgt_p2n_xyz
            gt_patch_corr = np.zeros(1)

            #sel_src_p2n_inds, sel_src_p2n_masks = src_p2n_inds[src_corr_node, :], src_p2n_masks[src_corr_node, :]
            #sel_tgt_p2n_inds, sel_tgt_p2n_masks = tgt_p2n_inds[tgt_corr_node, :], tgt_p2n_masks[tgt_corr_node, :]
            #sel_src_p2n_xyz, sel_tgt_p2n_xyz = src_p2n_xyz[src_corr_node, ...], tgt_p2n_xyz[tgt_corr_node, ...]

            #gt_patch_corr = calc_gt_patch_correspondence(sel_src_p2n_xyz, sel_tgt_p2n_xyz, rot, trans,
                                                         #distance_thres=self.overlap_radius)
        #    sel_src_p2n_inds, sel_src_p2n_masks = src_p2n_inds, src_p2n_masks
        #    sel_tgt_p2n_inds, sel_tgt_p2n_masks = tgt_p2n_inds, tgt_p2n_masks
        #    gt_patch_corr = None

        if self.input_type in ['ppf', 'mix']:
            src_patch_ppf, src_node_normal = build_ppf_patch(pointcloud1, src_node_inds, src_patch_inds, view_point=view_point)
            tgt_patch_ppf, tgt_node_normal = build_ppf_patch(pointcloud2, tgt_node_inds, tgt_patch_inds, view_point=view_point)

            indices = np.arange(src_node_xyz.shape[0])[np.newaxis, :]
            neighbors = indices.repeat(src_node_xyz.shape[0], axis=0)
            ###########################
            # Delete it self
            ###########################
            delete_index = (np.arange(src_node_xyz.shape[0]) * (src_node_xyz.shape[0] + 1)).astype(np.int32)
            # delete_index = [(i * src_node_xyz.shape[0] + i) for i in range(src_node_xyz.shape[0])]
            neighbors = np.reshape(neighbors, -1)
            neighbors = np.delete(neighbors, delete_index, axis=0)
            neighbors = np.reshape(neighbors, (src_node_xyz.shape[0], src_node_xyz.shape[0] - 1))
            ############################################################################################################
            src_node_ppf = calc_ppf_cpu(src_node_xyz, src_node_normal, src_node_xyz, src_node_normal,
                                        neighbors=neighbors)
            tgt_node_ppf = calc_ppf_cpu(tgt_node_xyz, tgt_node_normal, tgt_node_xyz, tgt_node_normal,
                                        neighbors=neighbors)

        if self.input_type in ['xyz', 'mix']:
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

        src_point_node_dist = get_square_distance_matrix(pointcloud1, src_node_xyz)
        src_k_node_neighbor_inds = np.argpartition(src_point_node_dist, axis=1, kth=self.interpolate_neighbors)[:, :self.interpolate_neighbors]
        tgt_point_node_dist = get_square_distance_matrix(pointcloud2, tgt_node_xyz)
        tgt_k_node_neighbor_inds = np.argpartition(tgt_point_node_dist, axis=1, kth=self.interpolate_neighbors)[:, :self.interpolate_neighbors]
        return pointcloud1.astype(np.float32), pointcloud2.astype(np.float32), \
               src_node_xyz.astype(np.float32), tgt_node_xyz.astype(np.float32), \
               src_node_ppf.astype(np.float32), tgt_node_ppf.astype(np.float32), \
               src_patch_xyz.astype(np.float32), tgt_patch_xyz.astype(np.float32), \
               src_patch.astype(np.float32), tgt_patch.astype(np.float32), \
               sel_src_p2n_inds.astype(np.int32), sel_tgt_p2n_inds.astype(np.int32), \
               sel_src_p2n_masks.astype(np.float32), sel_tgt_p2n_masks.astype(np.float32), \
               gt_patch_corr.astype(np.float32), \
               rot.astype(np.float32), trans.astype(np.float32), \
               pos_mask.astype(np.float32), \
               src_k_node_neighbor_inds, tgt_k_node_neighbor_inds

        #return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
        #       translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
        #       euler_ab.astype('float32'), euler_ba.astype('float32')

    def __len__(self):
        return self.data.shape[0]
