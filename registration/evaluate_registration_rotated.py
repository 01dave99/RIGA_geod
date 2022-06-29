import os, sys, glob, torch, argparse
cwd = os.getcwd()
sys.path.append(cwd)
import numpy as np
from lib.utils import setup_seed, natural_key
from tqdm import tqdm
from registration.benchmark_utils import ransac_pose_estimation, get_inlier_ratio, get_scene_split, write_est_trajectory
from registration.benchmark import benchmark
from lib.utils import square_distance
from visualizer.visualizer import Visualizer
from dataset.common import collect_local_neighbors

setup_seed(0)


def benchmark_registration(desc, exp_dir, whichbenchmark, n_points, ransac_with_mutual=False, inlier_ratio_threshold=0.05):
    gt_folder = f'configs/benchmarks/{whichbenchmark}'
    exp_dir = f'{exp_dir}/{whichbenchmark}/{n_points}'
    if (not os.path.exists(exp_dir)):
        os.makedirs(exp_dir)

    results = dict()
    results['w_mutual'] = {'inlier_ratios': [], 'distances': []}
    results['wo_mutual'] = {'inlier_ratios': [], 'distances': []}
    tsfm_est = []
    for eachfile in tqdm(desc):
        ######################################################
        # 1. take the nodes and descriptors
        print(eachfile)
        data = torch.load(eachfile)
        src_pcd, tgt_pcd = data['src_pcd'], data['tgt_pcd']
        src_nodes, tgt_nodes = data['src_nodes'], data['tgt_nodes']
        src_feats, tgt_feats = data['src_node_desc'], data['tgt_node_desc']
        src_point_feats, tgt_point_feats = data['src_point_desc'], data['tgt_point_desc']
        #feat_distance = torch.sqrt(square_distance(src_point_feats[None, :], tgt_point_feats[None, :])).squeeze(0).cuda()
        rot, trans = data['rot'], data['trans']
        #################################################################
        # visualization
        #visualizer = Visualizer(src_pcd, tgt_pcd, src_nodes, tgt_nodes, None, None, rot, trans)
        #visualizer.show_alignment()
        #visualizer.show_correspondences()
        #visualizer.show_pcd_with_nodes_and_one_patch(with_node=True, patch_id=None)
        #################################################################
        transformed_src_nodes = (torch.matmul(rot, src_nodes.T) + trans).T
        #geo_distance = torch.sqrt(square_distance(transformed_src_nodes[None, :], tgt_nodes[None, :]))

        #feat_dis_nn_src_tgt = 1. / (1e-8 + torch.min(feat_distance, dim=-1)[0].cpu().numpy())
        #feat_dis_nn_tgt_src = 1. / (1e-8 + torch.min(feat_distance, dim=-2)[0].cpu().numpy())
        #geo_dis_nn = torch.min(geo_distance, dim=-1)[0].numpy()
        #dis_nn = np.concatenate((geo_dis_nn, feat_dis_nn), axis=0)
        ######################################################
        # 2. run ransac
        #src_prob = feat_dis_nn_src_tgt / feat_dis_nn_src_tgt.sum()
        if src_pcd.shape[0] > n_points:
            sel_src_idx = np.random.choice(src_pcd.shape[0], n_points, replace=False)
        else:
            sel_src_idx = np.random.choice(src_pcd.shape[0], n_points, replace=True)

        sel_src_pcd, sel_src_desc = src_pcd[sel_src_idx, :], src_point_feats[sel_src_idx]

        #tgt_prob = feat_dis_nn_tgt_src / feat_dis_nn_tgt_src.sum()

        if tgt_pcd.shape[0] > n_points:
            sel_tgt_idx = np.random.choice(tgt_pcd.shape[0], n_points, replace=False)
        else:
            sel_tgt_idx = np.random.choice(tgt_pcd.shape[0], n_points, replace=True)

        sel_tgt_pcd, sel_tgt_desc = tgt_pcd[sel_tgt_idx, :], tgt_point_feats[sel_tgt_idx]

        #tsfm_est.append(ransac_pose_estimation(src_nodes, tgt_nodes, src_feats, tgt_feats, mutual=ransac_with_mutual))
        tsfm_est.append(ransac_pose_estimation(sel_src_pcd, sel_tgt_pcd, sel_src_desc, sel_tgt_desc, mutual=ransac_with_mutual))
        ######################################################
        # 3. calculate inlier ratios
        inlier_ratio_results = get_inlier_ratio(sel_src_pcd, sel_tgt_pcd, sel_src_desc, sel_tgt_desc, rot, trans)

        results['w_mutual']['inlier_ratios'].append(inlier_ratio_results['w']['inlier_ratio'])
        results['w_mutual']['distances'].append(inlier_ratio_results['w']['distance'])
        results['wo_mutual']['inlier_ratios'].append(inlier_ratio_results['wo']['inlier_ratio'])
        results['wo_mutual']['distances'].append(inlier_ratio_results['wo']['distance'])
        print(inlier_ratio_results['w']['inlier_ratio'], ' ', inlier_ratio_results['wo']['inlier_ratio'])

    tsfm_est = np.array(tsfm_est)

    ########################################
    # wirte the estimated trajectories
    write_est_trajectory(gt_folder, exp_dir, tsfm_est)

    ########################################
    # evaluate the results, here FMR and Inlier ratios are all average twice
    benchmark(exp_dir, gt_folder)
    split = get_scene_split(whichbenchmark)
    for key in ['w_mutual', 'wo_mutual']:
        inliers = []
        fmrs = []

        for ele in split:
            c_inliers = results[key]['inlier_ratios'][ele[0]:ele[1]]
            inliers.append(np.mean(c_inliers))
            fmrs.append((np.array(c_inliers) > inlier_ratio_threshold).mean())

        with open(os.path.join(exp_dir, 'result'), 'a') as f:
            f.write(f'Inlier ratio {key}: {np.mean(inliers):.3f} : +- {np.std(inliers):.3f}\n')
            f.write(f'Feature match recall {key}: {np.mean(fmrs):.3f} : +- {np.std(fmrs):.3f}\n')
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', default=None, type=str, help='path to precomputed features and scores')
    parser.add_argument('--benchmark', default='3DLoMatch', type=str, help='[3DMatch, 3DLoMatch]')
    parser.add_argument('--n_points', default=1000, type=int, help='number of points used by RANSAC')
    parser.add_argument('--exp_dir', default='est_traj', type=str, help='export final results')
    args = parser.parse_args()
    desc = sorted(glob.glob(f'{args.source_path}/*.pth'), key=natural_key)
    benchmark_registration(desc, args.exp_dir, args.benchmark, args.n_points, ransac_with_mutual=False)

