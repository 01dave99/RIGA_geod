# Reference: Choy et.al. FCGF, ICCV 2019
import torch.utils.data as data
import logging
import numpy as np
import glob, os, copy
import open3d as o3d
from scipy.linalg import expm, norm
from dataset.common import collect_local_neighbors, build_ppf_patch, farthest_point_subsampling, calc_patch_overlap_ratio, x_axis_crop, random_point_subsampling, get_square_distance_matrix, point2node_sampling
from lib.utils import to_o3d_pcd, get_correspondences, to_tsfm
from scipy.spatial.transform import Rotation

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def get_blue():
    '''
    Get color blue
    :return:
    '''
    return np.array([0, 0.651, 0.929])


def get_yellow():
    '''
    Get color yellow
    :return:
    '''
    return np.array([1, 0.706, 0])


class CameraPose:

  def __init__(self, meta, mat):
    self.metadata = meta
    self.pose = mat

  def __str__(self):
    return 'metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
        "pose : " + "\n" + np.array_str(self.pose)


def read_trajectory(filename, dim=4):
  traj = []
  assert os.path.exists(filename)
  with open(filename, 'r') as f:
    metastr = f.readline()
    while metastr:
      metadata = list(map(int, metastr.split()))
      mat = np.zeros(shape=(dim, dim))
      for i in range(dim):
        matstr = f.readline()
        mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
      traj.append(CameraPose(metadata, mat))
      metastr = f.readline()
    return traj


def write_trajectory(traj, filename, dim=4):
  with open(filename, 'w') as f:
    for x in traj:
      p = x.pose.tolist()
      f.write(' '.join(map(str, x.metadata)) + '\n')
      f.write('\n'.join(' '.join(map('{0:.12f}'.format, p[i])) for i in range(dim)))
      f.write('\n')


def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def sample_random_trans(pcd, randg, rotation_range=360.):
    T = np.eye(4)
    R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
    T[:3, :3] = R
    T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
    return T


def pcd_resample(pcd, min_keep_ratio):
    keep_ratio = 1. - np.random.rand(1)[0] * (1 - min_keep_ratio)
    pick_id = np.random.choice(pcd.shape[0], int(pcd.shape[0] * keep_ratio), replace=False)
    pcd = copy.deepcopy(pcd[pick_id, :])
    np.random.shuffle(pcd)
    return pcd


class PairDataset(data.Dataset):
    '''
    Template of pairwise dataset
    '''
    def __init__(self,
                 phase,
                 config=None):
        self.phase = phase
        self.files = []
        self.data_objects = []
        self.voxel_size = config.voxel_size
        self.randg = np.random.RandomState()
        self.config = config

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        t = trans[:3, 3]
        pts = pts @ R.T + t
        return pts

    def __len__(self):
        return len(self.files)

class OriginalTDMatchDataset(PairDataset):
    '''
    The original 3dmatch dataset that contains only the frame pairs with more than 30% overlap. This dataset is mainly used in works like FCGF.
    '''
    OVERLAP_RATIO = 0.3
    DATA_FILES = {
        'train': './configs/original_tdmatch/train_3dmatch.txt',
        'val': './configs/original_tdmatch/val_3dmatch.txt',
        'test': './configs/original_tdmatch/test_3dmatch.txt'
    }

    def __init__(self,
                 phase,
                 self_training=False,
                 data_augmentation=True,
                 config=None):
        PairDataset.__init__(self, phase, config)
        if self.phase == 'test':
            self.root = root = config.benchmark
        else:
            self.root = root = config.root
        self.self_training = self_training
        self.data_augmentation = data_augmentation
        self.resample = config.resample
        self.sample_patch_per_frame = self.patch_per_frame = config.patch_per_frame
        self.point_per_patch = config.point_per_patch
        self.input_type = config.input_type
        self.decentralization = config.decentralization
        self.patch_vicinity = config.patch_vicinity
        self.rot_factor = 360.
        self.augment_noise = config.augment_noise
        self.interpolate_neighbors = config.interpolate_neighbors

        logging.info(f"Loading the subset {phase} from {root}")

        subset_names = open(self.DATA_FILES[phase]).read().split()
        if self.phase == 'test':
            for sname in subset_names:
                traj_file = os.path.join(self.root, sname + '-evaluation/gt.log')

                assert os.path.exists(traj_file)
                traj = read_trajectory(traj_file)
                for ctraj in traj:
                    i = ctraj.metadata[0]
                    j = ctraj.metadata[1]
                    T_gt = ctraj.pose
                    self.files.append((sname, i, j, T_gt))
            #self.files = self.files[1430:]
        else:
            for name in subset_names:
                fname = name + "*%.2f.txt" % self.OVERLAP_RATIO
                fnames_txt = glob.glob(root + "/" + fname)
                assert len(fnames_txt) > 0, f"Make sure that the path {root} has data {fname}"
                for fname_txt in fnames_txt:
                    with open(fname_txt) as f:
                        content = f.readlines()
                    fnames = [x.strip().split() for x in content]
                    for fname in fnames:
                        self.files.append([fname[0], fname[1]])


        #if self.phase == 'train':
        #    for idx in range(len(self.files)):
        #        file0 = os.path.join(self.root, self.files[idx][0])
        #        file1 = os.path.join(self.root, self.files[idx][1])
        #        data0 = np.load(file0)
        #        data1 = np.load(file1)
        #        src_pcd = data0["pcd"]
        #        tgt_pcd = data1["pcd"]
        #        print('{}/{}: {}'.format(idx, len(self.files), file0))
        #        print('{}/{}: {}'.format(idx, len(self.files), file1))


    def __getitem__(self, idx):
        idx = 0

        if self.phase == 'test':
            sname, i, j, T_gt = self.files[idx]
            #ply_name0 = os.path.join(self.root, sname, f'cloud_bin_{j}.ply')
            #ply_name1 = os.path.join(self.root, sname, f'cloud_bin_{i}.ply')

            #pcd0 = o3d.io.read_point_cloud(ply_name0)
            #pcd1 = o3d.io.read_point_cloud(ply_name1)

            #down_ply_name0 = os.path.join(self.root, sname, f'cloud_bin_{j}_down.ply')
            #down_ply_name1 = os.path.join(self.root, sname, f'cloud_bin_{i}_down.ply')
            #o3d.io.write_point_cloud(down_ply_name0, pcd0.voxel_down_sample(self.voxel_size))
            #o3d.io.write_point_cloud(down_ply_name1, pcd1.voxel_down_sample(self.voxel_size))

            ply_name0 = os.path.join(self.root, sname, f'cloud_bin_{j}_down.ply')
            ply_name1 = os.path.join(self.root, sname, f'cloud_bin_{i}_down.ply')

            pcd0 = o3d.io.read_point_cloud(ply_name0)
            pcd1 = o3d.io.read_point_cloud(ply_name1)
            pcd0, _ = pcd0.remove_radius_outlier(nb_points=17, radius=0.1)
            pcd1, _ = pcd1.remove_radius_outlier(nb_points=17, radius=0.1)
            src_pcd = np.asarray(pcd0.points)
            tgt_pcd = np.asarray(pcd1.points)
            ####################################################################
            # voxel down-sampling
            #o3d_src_pcd, o3d_tgt_pcd = to_o3d_pcd(src_pcd), to_o3d_pcd(tgt_pcd)
            #src_pcd = np.asarray(o3d_src_pcd.voxel_down_sample(self.voxel_size).points)
            #tgt_pcd = np.asarray(o3d_tgt_pcd.voxel_down_sample(self.voxel_size).points)

            ###############################
            # get rotation & translation
            rot, trans = T_gt[:3, :3], T_gt[:3, 3]

        else:

            file0 = os.path.join(self.root, self.files[idx][0])
            file1 = os.path.join(self.root, self.files[idx][1])
            data0 = np.load(file0)
            data1 = np.load(file1)
            src_pcd = data0["pcd"]
            tgt_pcd = data1["pcd"]

            o3d_src_pcd, o3d_tgt_pcd = to_o3d_pcd(src_pcd), to_o3d_pcd(tgt_pcd)

            o3d_src_pcd = o3d_src_pcd.voxel_down_sample(self.voxel_size)
            src_cl, src_ind = o3d_src_pcd.remove_radius_outlier(nb_points=17, radius=0.1)
            src_pcd = np.asarray(src_cl.points)
            #display_inlier_outlier(o3d_src_pcd, src_ind)
            o3d_tgt_pcd = o3d_tgt_pcd.voxel_down_sample(self.voxel_size)
            tgt_cl, tgt_ind = o3d_tgt_pcd.remove_radius_outlier(nb_points=17, radius=0.1)
            tgt_pcd = np.asarray(tgt_cl.points)
            #display_inlier_outlier(o3d_tgt_pcd, tgt_ind)


            if self.self_training or self.data_augmentation:
                if self.self_training:
                    if np.random.rand(1)[0] > 0.5:
                        pcd = copy.deepcopy(src_pcd)
                    else:
                        pcd = copy.deepcopy(tgt_pcd)

                    euler_ab = np.random.rand(3) * np.pi * 2. / self.rot_factor  # anglez, angley, anglex
                    rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
                    pcd = np.matmul(pcd, rot_ab.T)

                    src_pcd, tgt_pcd = x_axis_crop(pcd)

                    src_pcd = np.matmul(src_pcd, rot_ab)
                    tgt_pcd = np.matmul(tgt_pcd, rot_ab)

                    np.random.shuffle(src_pcd)
                    np.random.shuffle(tgt_pcd)
                T0 = sample_random_trans(src_pcd, self.randg, self.rot_factor)
                T1 = sample_random_trans(tgt_pcd, self.randg, self.rot_factor)
                trans = T1 @ np.linalg.inv(T0)

                rot, trans = trans[:3, :3], trans[:3, 3]

                #############################################################
                # point cloud resample
                src_pcd = pcd_resample(src_pcd, min_keep_ratio=self.resample)
                tgt_pcd = pcd_resample(tgt_pcd, min_keep_ratio=self.resample)
                ##############################################################
                # generate guassian noise
                src_noise = np.clip(np.random.normal(0.0, scale=0.005, size=(src_pcd.shape[0], 3)),
                            a_min=-0.02, a_max=0.02)
                tgt_noise = np.clip(np.random.normal(0.0, scale=0.005, size=(tgt_pcd.shape[0], 3)),
                                    a_min=-0.02, a_max=0.02)

                src_pcd += src_noise
                tgt_pcd += tgt_noise
                ##############################################################
                # generate random transformation for both src and tgt point clouds
                src_pcd = self.apply_transform(src_pcd, T0)
                tgt_pcd = self.apply_transform(tgt_pcd, T1)
            else:
                T0 = sample_random_trans(src_pcd, self.randg, self.rot_factor)
                T1 = sample_random_trans(tgt_pcd, self.randg, self.rot_factor)
                trans = T1 @ np.linalg.inv(T0)
                src_pcd = self.apply_transform(src_pcd, T0)
                tgt_pcd = self.apply_transform(tgt_pcd, T1)



        if (trans.ndim == 1):
            trans = trans[:, None]
        #################################
        # generate patches
        #src_node_inds, src_node_xyz = random_point_subsampling(overlapped_src_pcd, self.patch_per_frame)
        #src_node_inds, src_node_xyz = farthest_point_subsampling(overlapped_src_pcd, self.patch_per_frame)
        tsfm = to_tsfm(rot, trans)
        correspondences = get_correspondences(to_o3d_pcd(src_pcd), to_o3d_pcd(tgt_pcd), tsfm, 0.0375, K=None)
        src_node_inds, src_node_xyz = farthest_point_subsampling(src_pcd, self.sample_patch_per_frame)
        tgt_node_inds, tgt_node_xyz = farthest_point_subsampling(tgt_pcd, self.sample_patch_per_frame)

        src_patch_inds, src_patch_xyz, src_patch_masks = collect_local_neighbors(src_pcd, src_node_xyz, self.patch_vicinity, self.point_per_patch)
        tgt_patch_inds, tgt_patch_xyz, tgt_patch_masks = collect_local_neighbors(tgt_pcd, tgt_node_xyz, self.patch_vicinity, self.point_per_patch)

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

        _, src_patch_xyz_, src_patch_masks_ = point2node_sampling(src_pcd, src_node_xyz, src_node_inds, self.point_per_patch)
        _, tgt_patch_xyz_, tgt_patch_masks_ = point2node_sampling(tgt_pcd, tgt_node_xyz, tgt_node_inds, self.point_per_patch)
        row_major_overlap_ratio, col_major_overlap_ratio = calc_patch_overlap_ratio(src_node_xyz, src_patch_xyz_, src_patch_masks_, tgt_node_xyz, tgt_patch_xyz_, tgt_patch_masks_, rot, trans)

        if self.input_type in ['ppf', 'mix']:
            src_patch_ppf = build_ppf_patch(src_pcd, src_node_inds, src_patch_inds)

            tgt_patch_ppf = build_ppf_patch(tgt_pcd, tgt_node_inds, tgt_patch_inds)

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

        src_point_node_dist = get_square_distance_matrix(src_pcd, src_node_xyz)
        src_k_node_neighbor_inds = np.argpartition(src_point_node_dist, axis=1, kth=self.interpolate_neighbors)[:, :self.interpolate_neighbors]
        tgt_point_node_dist = get_square_distance_matrix(tgt_pcd, tgt_node_xyz)
        tgt_k_node_neighbor_inds = np.argpartition(tgt_point_node_dist, axis=1, kth=self.interpolate_neighbors)[:, :self.interpolate_neighbors]
        return src_pcd.astype(np.float32), tgt_pcd.astype(np.float32), \
               src_node_xyz.astype(np.float32), tgt_node_xyz.astype(np.float32),\
               src_patch_xyz.astype(np.float32), tgt_patch_xyz.astype(np.float32),\
               src_patch.astype(np.float32), tgt_patch.astype(np.float32), \
               rot.astype(np.float32), trans.astype(np.float32), \
               row_major_overlap_ratio.astype(np.float32), col_major_overlap_ratio.astype(np.float32), \
               correspondences, src_k_node_neighbor_inds, tgt_k_node_neighbor_inds

