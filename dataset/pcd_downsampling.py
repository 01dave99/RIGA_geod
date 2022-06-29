import open3d as o3d
import os
from tqdm import tqdm
import numpy as np


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


DATA_FILES = {
    'train': './configs/original_tdmatch/train_3dmatch.txt',
    'val': './configs/original_tdmatch/val_3dmatch.txt',
    'test': './configs/original_tdmatch/test_3dmatch.txt'
}

phase = 'test'
root = 'data/original_tdmatch_test'
voxel_size = 0.025

subset_names = open(DATA_FILES[phase]).read().split()
files = []
for sname in subset_names:
    traj_file = os.path.join(root, sname + '-evaluation/gt.log')
    assert os.path.exists(traj_file)
    traj = read_trajectory(traj_file)
    for ctraj in traj:
        i = ctraj.metadata[0]
        j = ctraj.metadata[1]
        T_gt = ctraj.pose
        files.append((sname, i, j, T_gt))

for idx in tqdm(range(len(files))):
    sname, i, j, T_gt = files[idx]
    ply_name0 = os.path.join(root, sname, f'cloud_bin_{j}.ply')
    ply_name1 = os.path.join(root, sname, f'cloud_bin_{i}.ply')

    pcd0 = o3d.io.read_point_cloud(ply_name0)
    pcd1 = o3d.io.read_point_cloud(ply_name1)

    down_ply_name0 = os.path.join(root, sname, f'cloud_bin_{j}_down.ply')
    down_ply_name1 = os.path.join(root, sname, f'cloud_bin_{i}_down.ply')
    o3d.io.write_point_cloud(down_ply_name0, pcd0.voxel_down_sample(voxel_size))
    o3d.io.write_point_cloud(down_ply_name1, pcd1.voxel_down_sample(voxel_size))
