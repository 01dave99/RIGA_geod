import os, gc
import torch
from tensorboardX import SummaryWriter
from lib.utils import AverageMeter, Logger
from tqdm import tqdm


class Trainer(object):
    '''
    Class Trainer
    '''

    def __init__(self, config):
        self.config = config
        #############################
        # hyper-parameters
        #############################
        self.verbose = config.verbose
        self.verbose_freq = config.verbose_freq
        self.start_epoch = 1
        self.max_epoch = config.max_epoch
        self.training_max_iter = config.training_max_iter
        self.val_max_iter = config.val_max_iter
        self.device = 'cpu'

        self.best_total_loss = self.best_coarse_loss = self.best_fine_loss = 1e5
        self.best_coarse_matching_recall = self.best_fine_matching_recall = -1.

        self.save_dir = config.save_dir
        self.snapshot_dir = config.snapshot_dir

        self.model = config.model.to(self.device)
        self.local_rank = config.local_rank
        self.optimizer = config.optimizer
        self.scheduler = config.scheduler
        self.scheduler_interval = config.scheduler_interval
        self.snapshot_interval = config.snapshot_interval
        self.iter_size = config.iter_size

        self.patch_per_frame = config.patch_per_frame
        self.point_per_patch = config.point_per_patch
        self.patch_vicinity = config.patch_vicinity
        self.ratio_drop = config.ratio_drop

        if self.local_rank <= 0:
            self.writer = SummaryWriter(logdir=config.tboard_dir)
            self.logger = Logger(self.snapshot_dir)
            self.logger.write(f'#parameters {sum([x.nelement() for x in self.model.parameters()]) / 1000000.} M\n')
            with open(f'{config.snapshot_dir}/model.log', 'w') as f:
                f.write(str(self.model))

            f.close()
        else:
            self.writer = None
            self.logger = None

        if config.pretrain != '':
            self._load_pretrain(config.pretrain)

        self.loader = dict()

        self.loader['train'] = config.train_loader
        self.loader['val'] = config.val_loader
        self.loader['test'] = config.test_loader

        self.metric_loss = config.metric_loss

        self.w_coarse = config.w_coarse
        self.w_fine = config.w_fine

    def _snapshot(self, epoch, name=None):
        '''
        Save a trained model
        :param epoch:  epoch of current model
        :param name: path to save current model
        :return: None
        '''
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_total_loss': self.best_total_loss,
            'best_coarse_loss': self.best_coarse_loss,
            'best_fine_loss': self.best_fine_loss,
            'best_coarse_matching_recall': self.best_coarse_matching_recall,
            'best_fine_matching_recall': self.best_fine_matching_recall
        }

        if name is None:
            filename = os.path.join(self.save_dir, f'model_{epoch}.pth')
        else:
            filename = os.path.join(self.save_dir, f'model_{name}.pth')

        print(f'Save model to {filename}')
        self.logger.write(f'Save model to {filename}\n')
        torch.save(state, filename)

    def _load_pretrain(self, resume):
        '''
        Load a pretrained model
        :param resume: the path to the pretrained model
        :return: None
        '''
        if os.path.isfile(resume):
            print(f'=> loading checkpoint {resume}')
            state = torch.load(resume)
            self.start_epoch = state['epoch']
            #print(state['state_dict'])
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in state['state_dict'].items()})
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
            self.best_total_loss = state['best_total_loss']
            self.best_coarse_loss = state['best_coarse_loss']
            self.best_fine_loss = state['best_fine_loss']
            self.best_coarse_matching_recall = state['best_coarse_matching_recall']
            self.best_fine_matching_recall = state['best_fine_matching_recall']

            self.logger.write(f'Successfully load pretrained model from {resume}!\n')
            self.logger.write(f'Current best total loss {self.best_total_loss}\n')
            self.logger.write(f'Current best coarse loss {self.best_coarse_loss}\n')
            self.logger.write(f'Current best fine loss {self.best_fine_loss}\n')

            self.logger.write(f'Current best coarse matching recall {self.best_coarse_matching_recall}\n')
            self.logger.write(f'Current best fine matching recall {self.best_fine_matching_recall}\n')

        else:
            raise ValueError(f'=> no checkpoint found at {resume}')

    def _get_lr(self, group=0):
        '''
        Get current learning rate
        :param group:
        :return:
        '''
        return self.optimizer.param_groups[group]['lr']

    def stats_dict(self):
        '''
        Create the dictionary consisting of all the metrics
        :return: as described
        '''
        stats = dict()
        stats['total_loss'] = 0.
        stats['coarse_loss'] = 0.
        stats['fine_loss'] = 0.
        stats['coarse_matching_recall'] = 0.
        stats['fine_matching_recall'] = 0.

        '''
        to be added
        '''
        return stats

    def stats_meter(self):
        '''
        For each metric in stats dict, create an AverageMeter class for updating
        :return: as described
        '''
        meters = dict()
        stats = self.stats_dict()
        for key, _ in stats.items():
            meters[key] = AverageMeter()
        return meters

    def inference_one_batch(self, inputs, phase, idx=None):
        '''
        Inference for a single batch data
        :param inputs: the dictionary consisting of all the input data
        :param phase: train, validation or test
        :return: dictionary consisting of losses and metrics
        '''
        assert phase in ['train', 'val']
        #########################################
        # training
        #########################################
        if phase == 'train':
            self.model.train()

            ##################
            # forward pass
            ##################
            rot, trans = inputs['rot'], inputs['trans']
            pos_mask = inputs['pos_mask']
            src_patch_feats, tgt_patch_feats = inputs['src_patch_feat'], inputs['tgt_patch_feat']
            src_node_geod, tgt_node_geod = inputs['src_node_geod'], inputs['tgt_node_geod']
            batch_size = src_patch_feats.shape[0]

            src_patch_feats = src_patch_feats.view(batch_size * self.patch_per_frame, self.point_per_patch, -1) #[B, N, P, F] -> [B*N, P, F]
            tgt_patch_feats = tgt_patch_feats.view(batch_size * self.patch_per_frame, self.point_per_patch, -1) #[B, N, P, F] -> [B*N, P, F]

            src_patch_xyz, tgt_patch_xyz = inputs['src_nodes'], inputs['tgt_nodes'] #[B, N, 3] and [B, N, 3]
            src_pcd, tgt_pcd = inputs['src_points'], inputs['tgt_points']
            src_p2n_inds, src_p2n_masks = inputs['src_p2n_inds'], inputs['src_p2n_masks']
            tgt_p2n_inds, tgt_p2n_masks = inputs['tgt_p2n_inds'], inputs['tgt_p2n_masks']
            gt_patch_corr = inputs['gt_patch_corr']
            src_knn_node_inds, tgt_knn_node_inds = inputs['src_knn_node_inds'], inputs['tgt_knn_node_inds']
            src_descriptors, tgt_descriptors, src_pcd_desc, tgt_pcd_desc, matching_scores, src_p2n_masks, tgt_p2n_masks = self.model.forward(
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

            neg_mask = (pos_mask == 0.)

            stats = self.metric_loss(src_descriptors, tgt_descriptors, pos_mask, neg_mask, matching_scores,
                                     gt_patch_corr, src_p2n_masks, tgt_p2n_masks)

            stats['total_loss'] = loss = self.w_coarse * stats['coarse_loss'] + self.w_fine * stats['fine_loss']
            loss.backward()

        else:
            self.model.eval()
            with torch.no_grad():
                rot, trans = inputs['rot'], inputs['trans']
                pos_mask = inputs['pos_mask']
                src_patch_feats, tgt_patch_feats = inputs['src_patch_feat'], inputs['tgt_patch_feat']
                src_node_geod, tgt_node_geod = inputs['src_node_geod'], inputs['tgt_node_geod']
                batch_size = src_patch_feats.shape[0]
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
                src_descriptors, tgt_descriptors, src_pcd_desc, tgt_pcd_desc, matching_scores, src_p2n_masks, tgt_p2n_masks = self.model.forward(src_pcd, tgt_pcd,
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

                neg_mask = (pos_mask == 0.)

                stats = self.metric_loss(src_descriptors, tgt_descriptors, pos_mask, neg_mask, matching_scores, gt_patch_corr, src_p2n_masks, tgt_p2n_masks)

                stats['total_loss'] = self.w_coarse * stats['coarse_loss'] + self.w_fine * stats['fine_loss']

        ########################################
        # re-organize dictionary stats
        ########################################
        stats['total_loss'] = float(stats['total_loss'].detach())
        stats['coarse_loss'] = float(stats['coarse_loss'].detach())
        stats['fine_loss'] = float(stats['fine_loss'].detach())
        return stats

    def inference_one_epoch(self, epoch, phase):
        '''
        Inference for an epoch
        :param epoch: current epoch
        :param phase: current phase of training
        :return:
        '''
        gc.collect()
        assert phase in ['train', 'val']

        #init stats meter
        stats_meter = self.stats_meter()

        num_iter = int(len(self.loader[phase]))
        c_loader_iter = self.loader[phase].__iter__()

        self.optimizer.zero_grad()
        idx = 0
        for c_iter in tqdm(range(num_iter)):
            inputs = c_loader_iter.next()
            for k, v in inputs.items():
                if type(v) == list:
                    inputs[k] = [item.to(self.device) for item in v]
                else:
                    inputs[k] = v.to(self.device)

            ######################
            # forward pass
            ######################
            stats = self.inference_one_batch(inputs, phase, idx=idx)
            idx += 1

            ######################
            # run optimization
            ######################
            if (c_iter + 1) % self.iter_size == 0 and phase == 'train':
                self.optimizer.step()
                self.optimizer.zero_grad()

            ########################
            # update to stats_meter
            ########################
            for key, value in stats.items():
                stats_meter[key].update(value)

            #torch.cuda.empty_cache()

            if self.local_rank <= 0 and self.verbose and (c_iter + 1) % self.verbose_freq == 0:
                cur_iter = num_iter * (epoch - 1) + c_iter
                for key, value in stats_meter.items():
                    self.writer.add_scalar(f'{phase}/{key}', value.avg, cur_iter)

                message = f'{phase} Epoch: {epoch} [{c_iter + 1:4d}/{num_iter}] '
                for key, value in stats_meter.items():
                    message += f'{key}:{value.avg:.2f}\t'

                self.logger.write(message + '\n')
        if self.local_rank <= 0:
            message = f'{phase} Epoch: {epoch} '
            for key, value in stats_meter.items():
                message += f'{key}: {value.avg:.4f}\t'

            self.logger.write(message + '\n')
        return stats_meter

    def train(self):
        '''
        Train
        :return:
        '''
        print('start training...')
        for epoch in range(self.start_epoch, self.max_epoch):
            if self.local_rank > -1:
                self.loader['train'].sampler.set_epoch(epoch)

            self.inference_one_epoch(epoch, 'train')
            self.scheduler.step()
            stats_meter = self.inference_one_epoch(epoch, 'val')
            if self.local_rank <= 0:
                if stats_meter['total_loss'].avg < self.best_total_loss:
                    self.best_total_loss = stats_meter['total_loss'].avg
                    self._snapshot(epoch, 'best_total_loss')

                if stats_meter['coarse_loss'].avg < self.best_coarse_loss:
                    self.best_coarse_loss = stats_meter['coarse_loss'].avg
                    self._snapshot(epoch, 'best_coarse_loss')

                if stats_meter['fine_loss'].avg < self.best_fine_loss:
                    self.best_fine_loss = stats_meter['fine_loss'].avg
                    self._snapshot(epoch, 'best_fine_loss')

                if stats_meter['coarse_matching_recall'].avg > self.best_coarse_matching_recall:
                    self.best_coarse_matching_recall = stats_meter['coarse_matching_recall'].avg
                    self._snapshot(epoch, 'best_coarse_matching_recall')

                if stats_meter['fine_matching_recall'].avg > self.best_fine_matching_recall:
                    self.best_fine_matching_recall = stats_meter['fine_matching_recall'].avg
                    self._snapshot(epoch, 'best_fine_matching_recall')


        print('training finish!')

    def eval(self):
        '''
        Evaluation
        :return:
        '''
        print('start to evaluate on validation sets...')
        stats_meter = self.inference_one_epoch(0, 'val')

        for key, value in stats_meter.items():
            print(f'{key}: {value.avg}')
