import os
import torch
from tqdm import tqdm
from lib.trainer import Trainer
from lib.utils import to_o3d_pcd
from visualizer.visualizer import Visualizer, create_visualizer
from visualizer.feature_space import visualize_feature_space
import open3d as o3d
import numpy as np

class Tester(Trainer):
    '''
    Tester
    '''

    def __init__(self, config):
        Trainer.__init__(self, config)

    def test(self):
        print('Starting to evaluate on test datasets...')
        os.makedirs(f'{self.snapshot_dir}/{self.config.benchmark}', exist_ok=True)

        #num_iter = int(len(self.loader['test'].dataset) // self.loader['test'].batch_size)

        num_iter = len(self.loader['test'])

        c_loader_iter = self.loader['test'].__iter__()
        self.model.eval()
        cnt = 0
        with torch.no_grad():
            for idx in tqdm(range(num_iter)):
                inputs = c_loader_iter.next()

                #######################################
                # Load inputs to device
                for k, v in inputs.items():
                    if v is None:
                        pass
                    elif type(v) == list:
                        inputs[k] = [items.to(self.device) for items in v]
                    else:
                        inputs[k] = v.to(self.device)

                #######################################
                # Forward pass
                rot, trans = inputs['rot'], inputs['trans']
                pos_mask = inputs['pos_mask']
                src_patch_feats, tgt_patch_feats = inputs['src_patch_feat'], inputs['tgt_patch_feat']
                src_node_geod, tgt_node_geod = inputs['src_node_geod'], inputs['tgt_node_geod']
                batch_size = src_patch_feats.shape[0]
                patch_num = src_patch_feats.shape[1]
                point_num = src_patch_feats.shape[2]
                src_patch_feats = src_patch_feats.view(batch_size * self.patch_per_frame, self.point_per_patch,
                                                       -1)  # [B, N, P, F] -> [B*N, P, F]
                tgt_patch_feats = tgt_patch_feats.view(batch_size * self.patch_per_frame, self.point_per_patch,
                                                       -1)  # [B, N, P, F] -> [B*N, P, F]

                src_patch_xyz, tgt_patch_xyz = inputs['src_nodes'], inputs['tgt_nodes']  # [B, N, 3] and [B, N, 3]
                src_pcd, tgt_pcd = inputs['src_points'], inputs['tgt_points']

                src_p2n_inds, src_p2n_masks = inputs['src_p2n_inds'], inputs['src_p2n_masks']
                tgt_p2n_inds, tgt_p2n_masks = inputs['tgt_p2n_inds'], inputs['tgt_p2n_masks']
                gt_patch_corr = inputs['gt_patch_corr']
                src_knn_node_inds, tgt_knn_node_inds = inputs['src_knn_node_inds'], inputs['tgt_knn_node_inds']
                src_descriptors, tgt_descriptors, src_pcd_desc, tgt_pcd_desc, src_corr_pts, tgt_corr_pts, confidence = self.model.forward(
                    src_pcd, tgt_pcd,
                    src_patch_xyz,
                    tgt_patch_xyz,
                    src_node_geod,
                    tgt_node_geod,
                    src_patch_feats,
                    tgt_patch_feats,
                    src_knn_node_inds,
                    tgt_knn_node_inds,
                    src_p2n_inds, tgt_p2n_inds,
                    src_p2n_masks, tgt_p2n_masks, rot, trans)

                ##################################################
                # Visulization
                #visualizer = create_visualizer(inputs, src_descriptors, tgt_descriptors, src_pcd_desc, tgt_pcd_desc, offset=2., to_mesh=False)
                #visualizer.show_pcd_with_nodes_and_one_patch(with_node=True, with_patch=0)
                # visualizer.save_gt_correspondences(save_dir=self.config.visual_dir, idx=idx)
                # visualizer.save_est_correspondences(save_dir=self.config.visual_dir, idx=idx)
                # visualizer.save_pcd_and_patch(save_dir=self.config.visual_dir, idx=idx)
                # src_pcd, tgt_pcd = inputs['src_points'], inputs['tgt_points']
                visual_check = True
                if visual_check:

                    visualize_feature_space(src_pcd.cpu().numpy(), src_pcd_desc.detach().cpu().numpy(),
                                            tgt_pcd.cpu().numpy() + 2., tgt_pcd_desc.detach().cpu().numpy(), to_sphere=False)
                    visualize_feature_space(src_patch_xyz[0].cpu().numpy(), src_descriptors[0].detach().cpu().numpy(), tgt_patch_xyz[0].cpu().numpy() + 2., tgt_descriptors[0].detach().cpu().numpy())
                ##########################################################
                # Store results to hard disk
                src_pcd_length, tgt_pcd_length = inputs['src_lengths'], inputs['tgt_lengths']
                src_pcd, tgt_pcd = inputs['src_points'], inputs['tgt_points']
                for i in range(rot.shape[0]):
                    data = dict()
                    if i == 0:
                        src_start = 0
                        tgt_start = 0
                    else:
                        src_start = src_pcd_length[i - 1]
                        tgt_start = tgt_pcd_length[i - 1]

                    src_end = src_pcd_length[i]
                    tgt_end = tgt_pcd_length[i]

                    data['src_pcd'], data['tgt_pcd'] = src_pcd[src_start:src_end, :].cpu(), tgt_pcd[tgt_start:tgt_end, :].cpu()
                    data['src_nodes'], data['tgt_nodes'] = inputs['src_nodes'][i].cpu(), inputs['tgt_nodes'][i].cpu()
                    data['src_node_desc'], data['tgt_node_desc'] = src_descriptors[i].cpu().detach(), tgt_descriptors[i].cpu().detach()
                    data['src_point_desc'], data['tgt_point_desc'] = src_pcd_desc[src_start:src_end, :].cpu().detach(), tgt_pcd_desc[tgt_start:tgt_end, :].cpu().detach()
                    data['src_corr_pts'], data['tgt_corr_pts'] = src_corr_pts.view(-1, 3).cpu(), tgt_corr_pts.view(-1, 3).cpu()
                    data['confidence'] = confidence.view(-1).cpu().detach()
                    print(data['confidence'].shape[0])
                    data['rot'], data['trans'] = rot[i].cpu(), trans[i].cpu()
                    print(f'{self.snapshot_dir}/{self.config.benchmark}/{cnt}.pth', ' ', data.keys())
                    torch.save(data, f'{self.snapshot_dir}/{self.config.benchmark}/{cnt}.pth')
                    cnt += 1

                ###########################################################


def get_trainer(config):
    '''
    Get corresponding trainer according to the config file
    :param config:
    :return:
    '''

    if config.dataset == 'tdmatch' or config.dataset == 'original_tdmatch' or config.dataset=='modelnet40':
        return Tester(config)
    else:
        raise NotImplementedError