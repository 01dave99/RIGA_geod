import os, sys, glob, torch, argparse
cwd = os.getcwd()
sys.path.append(cwd)
import numpy as np
from lib.utils import setup_seed, natural_key
from dataset.common import get_square_distance_matrix
from tqdm import tqdm
from registration.benchmark_utils import ransac_pose_estimation, get_inlier_ratio, get_scene_split, write_est_trajectory
from lib.utils import AverageMeter, square_distance, weighted_procrustes, to_o3d_pcd
import math
import open3d as o3d
setup_seed(0)


def benchmark_evaluation(desc):
    tsfm_est = []
    tsfm_gt = []
    LR = AverageMeter()
    Lt = AverageMeter()
    RMSE = AverageMeter()
    i = 0
    sel_pts_num = 10
    #desc = desc[275:276]
    error_idx1 = []
    error_idx2 = []

    for eachfile in tqdm(desc):

        ######################################################
        # 1. take the nodes and descriptors

        print(eachfile)
        data = torch.load(eachfile)
        src_pcd, tgt_pcd = data['src_pcd'], data['tgt_pcd']

        src_nodes, tgt_nodes = data['src_nodes'], data['tgt_nodes']
        src_feats, tgt_feats = data['src_node_desc'], data['tgt_node_desc']
        src_point_feats, tgt_point_feats = data['src_point_desc'], data['tgt_point_desc']
        rot, trans = data['rot'], data['trans']
        transformed_src_pcd = (torch.matmul(rot, src_pcd.T) + trans).T
        feat_dist = square_distance(src_point_feats[None, ...], tgt_point_feats[None, ...])[0].numpy()
        geo_dist = square_distance(transformed_src_pcd[None, ...], tgt_pcd[None, ...])[0].numpy()
        o3d_src_pcd = to_o3d_pcd(src_pcd.numpy())
        o3d_tgt_pcd = to_o3d_pcd(tgt_pcd.numpy())
        o3d_tsfm_src_pcd = to_o3d_pcd(transformed_src_pcd.numpy())
        #o3d.visualization.draw_geometries([o3d_src_pcd, o3d_tgt_pcd])
        gt_tsfm = np.eye(4)
        gt_tsfm[:3, :3] = rot.numpy()
        gt_tsfm[:3, -1] = trans[:, 0].numpy()
        np.savez('example.npz', src_points=src_pcd.numpy(), ref_points=tgt_pcd.numpy(), transform=gt_tsfm)
        ######################################################
        # 2. run ransac
        '''
        min_dist = np.min(feat_dist, axis=-1)
        indices = np.argsort(min_dist)
        indices = indices[:sel_pts_num]
        src_pcd = src_pcd[indices, :]
        src_point_feats = src_point_feats[indices, :]
        #min_dist = np.min(feat_dist, axis=0)
        #indices = np.argsort(min_dist)
        #tgt_pcd = tgt_pcd[indices[:sel_pts_num], :]
        #tgt_point_feats = tgt_point_feats[indices[:sel_pts_num], :]
        feat_dist = square_distance(src_point_feats[None, ...], tgt_point_feats[None, ...])[0].numpy()
        transformed_src_pcd = (torch.matmul(rot, src_pcd.T) + trans).T

        geo_dist = square_distance(transformed_src_pcd[None, ...], tgt_pcd[None, ...])[0].numpy()
        min_id = np.argmin(feat_dist, axis=-1)
        corr_tgt_pts = tgt_pcd[min_id]
        geo_dist_ = torch.sum((transformed_src_pcd - corr_tgt_pts) **2, dim=-1).numpy()
        '''
        tsfm_est.append(ransac_pose_estimation(src_pcd, tgt_pcd, src_point_feats, tgt_point_feats, distance_threshold=0.025))

        rot_ = tsfm_est[i][:3, :3]
        trans_ = tsfm_est[i][:3, -1]
        #print(rot, ' ', rot_)
        transformed_src_pcd = (torch.matmul(torch.from_numpy(rot_).float(), src_pcd.T) + trans).T

        o3d_est_tsfm_src_pcd = to_o3d_pcd(transformed_src_pcd.numpy())
        o3d.io.write_point_cloud('./visualization/src.ply', o3d_src_pcd)
        o3d.io.write_point_cloud('./visualization/gt_trans_src.ply', o3d_tsfm_src_pcd)
        o3d.io.write_point_cloud('./visualization/tgt.ply', o3d_tgt_pcd)
        o3d.io.write_point_cloud('./visualization/est_trans_src.ply', o3d_est_tsfm_src_pcd)






        '''
        feat_dist = square_distance(src_point_feats[None, ...], tgt_point_feats[None, ...])[0]
        sel_src_id = torch.from_numpy(np.arange(src_point_feats.shape[0])).long()
        sel_tgt_id = torch.min(feat_dist, dim=-1)[1]
        conf = feat_dist[sel_src_id, sel_tgt_id]
        conf = 1 / (1e-8 + conf)
        sel_src_pcd = src_pcd[sel_src_id]
        sel_tgt_pcd = tgt_pcd[sel_tgt_id]
        tsfm = weighted_procrustes(sel_src_pcd.cuda(), sel_tgt_pcd.cuda(), conf.cuda(), return_transform=True).cpu().numpy()
        tsfm_est.append(tsfm)
        '''
        transformation = np.eye(4)
        transformation[:3, :3] = rot.numpy()
        transformation[:3, -1] = trans[:, 0].numpy()
        tsfm_gt.append(transformation)

        cur_tsfm_est, cur_tsfm_gt = tsfm_est[i], tsfm_gt[i]

        cur_rot_est, cur_rot_gt = cur_tsfm_est[:3, :3], cur_tsfm_gt[:3, :3]

        cur_trans_est, cur_trans_gt = cur_tsfm_est[:3, -1], cur_tsfm_gt[:3, -1]
        v = (np.trace(np.matmul(cur_rot_est.T, cur_rot_gt)) - 1) / 2.
        if v < -1:
            v = -1
        if v > 1.:
            v = 1

        cur_LR = math.acos(v) * 180. / np.pi

        if cur_LR > 100.:
            error_idx1.append(i)
        elif cur_LR > 10:
            error_idx2.append(i)

        cur_Lt = np.linalg.norm(cur_trans_est - cur_trans_gt)

        gt_transformed_src_pcd = np.matmul(src_pcd.numpy(), cur_rot_gt.T) + cur_trans_gt[np.newaxis, :]
        pred_transformed_src_pcd = np.matmul(src_pcd.numpy(), cur_rot_est.T) + cur_trans_est[np.newaxis, :]
        #cur_RMSE = math.sqrt(np.sum(np.sum((gt_transformed_src_pcd - pred_transformed_src_pcd) ** 2, axis=-1))) / src_pcd.shape[0]
        cur_RMSE = np.mean(np.linalg.norm(gt_transformed_src_pcd - pred_transformed_src_pcd, axis=-1))
        print('{}: Rot Error {}, trans error {}, RMSE {}'.format(i, cur_LR, cur_Lt, cur_RMSE))

        i += 1

        LR.update(cur_LR)
        Lt.update(cur_Lt)
        RMSE.update(cur_RMSE)


    #LRMSE = AverageMeter()
    print(len(error_idx1))
    print(error_idx1)
    print(len(error_idx2))
    print(error_idx2)
    print('Rotation Error: {}, Translation Error: {}， RMSE: {}'.format(LR.avg, Lt.avg, RMSE.avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', default=None, type=str, help='path to precomputed features and scores')
    args = parser.parse_args()
    desc = sorted(glob.glob(f'{args.source_path}/*.pth'), key=natural_key)[:100]
    benchmark_evaluation(desc)

