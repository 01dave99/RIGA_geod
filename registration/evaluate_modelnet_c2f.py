import os, sys, glob, torch, argparse
cwd = os.getcwd()
sys.path.append(cwd)
import numpy as np
from lib.utils import setup_seed, natural_key
from tqdm import tqdm
from registration.benchmark_utils import ransac_pose_estimation_correspondences, get_inlier_ratio_correspondence, get_scene_split, write_est_trajectory
from registration.benchmark import benchmark
from lib.utils import square_distance
from visualizer.visualizer import Visualizer
from visualizer.plot import draw_distance_geo_feat
from dataset.common import collect_local_neighbors, get_square_distance_matrix, point2node_sampling
from lib.utils import weighted_procrustes, AverageMeter
import math


setup_seed(0)


def benchmark_evaluation(desc):
    tsfm_est = []
    tsfm_gt = []
    LR = AverageMeter()
    Lt = AverageMeter()
    RMSE = AverageMeter()
    ii = 0
    coarse_sample = 32
    n_points = 1000
    mutual = True

    for eachfile in tqdm(desc):
        ######################################################
        # 1. take the nodes and descriptors
        data = torch.load(eachfile)
        src_pcd, tgt_pcd = data['src_pcd'], data['tgt_pcd']
        src_nodes, tgt_nodes = data['src_nodes'], data['tgt_nodes']
        src_feats, tgt_feats = data['src_node_desc'], data['tgt_node_desc']
        src_point_feats, tgt_point_feats = data['src_point_desc'], data['tgt_point_desc']
        rot, trans = data['rot'], data['trans']
        src_corr_pts, tgt_corr_pts = data['src_corr_pts'], data['tgt_corr_pts']
        confidence = data['confidence']
        ######################################################
        # 2. run ransac
        prob = confidence / torch.sum(confidence)
        #print(confidence.topk(10)[0])
        print(confidence.shape[0])
        if prob.shape[0] > n_points:
            sel_idx = np.random.choice(prob.shape[0], n_points, replace=False, p=prob.numpy())
            #sel_idx = torch.topk(confidence, k=n_points)[1]
            src_corr_pts, tgt_corr_pts = src_corr_pts[sel_idx], tgt_corr_pts[sel_idx]

        correspondences = torch.from_numpy(np.arange(src_corr_pts.shape[0])[:, np.newaxis]).expand(-1, 2)
        tsfm_est.append(ransac_pose_estimation_correspondences(src_corr_pts, tgt_corr_pts, correspondences))

        #tsfm_est.append(ransac_pose_estimation(src_pcd, tgt_pcd, src_point_feats, tgt_point_feats, distance_threshold=0.025))
        transformation = np.eye(4)
        transformation[:3, :3] = rot.numpy()
        transformation[:3, -1] = trans[:, 0].numpy()
        tsfm_gt.append(transformation)

        cur_tsfm_est, cur_tsfm_gt = tsfm_est[ii], tsfm_gt[ii]

        cur_rot_est, cur_rot_gt = cur_tsfm_est[:3, :3], cur_tsfm_gt[:3, :3]

        cur_trans_est, cur_trans_gt = cur_tsfm_est[:3, -1], cur_tsfm_gt[:3, -1]
        v = (np.trace(np.matmul(cur_rot_gt.T, cur_rot_est)) - 1) / 2.
        if v < -1:
            v = -1
        if v > 1.:
            v = 1

        cur_LR = math.acos(v) * 180. / np.pi
        cur_Lt = np.linalg.norm(cur_trans_est - cur_trans_gt)

        gt_transformed_src_pcd = np.matmul(src_pcd, cur_rot_gt.T) + cur_trans_gt[np.newaxis, :]
        pred_transformed_src_pcd = np.matmul(src_pcd, cur_rot_est.T) + cur_trans_est[np.newaxis, :]
        #cur_RMSE = math.sqrt(np.sum(np.sum((gt_transformed_src_pcd - pred_transformed_src_pcd) ** 2, axis=-1))) / src_pcd.shape[0]
        cur_RMSE = np.mean(np.linalg.norm(gt_transformed_src_pcd - pred_transformed_src_pcd, axis=-1))
        print('{}: Rot Error {}, trans error {}, RMSE {}'.format(ii, cur_LR, cur_Lt, cur_RMSE))

        ii += 1

        LR.update(cur_LR)
        Lt.update(cur_Lt)
        RMSE.update(cur_RMSE)


    #LRMSE = AverageMeter()


    print('Rotation Error: {}, Translation Error: {}， RMSE: {}'.format(LR.avg, Lt.avg, RMSE.avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', default=None, type=str, help='path to precomputed features and scores')
    args = parser.parse_args()
    desc = sorted(glob.glob(f'{args.source_path}/*.pth'), key=natural_key)[:100]
    benchmark_evaluation(desc)

