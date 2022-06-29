import torch
from functools import partial
from dataset.common import load_info, collate_fn
from dataset.tdmatch import TDMatchDataset
from dataset.modelnet import ModelNet40
#from dataset.original_tdmatch import OriginalTDMatchDataset


def get_dataset(config):
    '''
    Make pytorch dataset for training, validation and testing
    :param config: configuration
    :return: train_set: training dataset
             val_set: validation dataset
             benchmark_set: testing dataset
    '''
    if config.dataset == 'tdmatch':
        info_train = load_info(config.train_info)
        info_val = load_info(config.val_info)
        info_benchmark = load_info(f'configs/tdmatch/{config.benchmark}.pkl')
        #test on same data as train:
        info_benchmark=info_train
        
        training_set = TDMatchDataset(info_train, config, data_augmentation=True)
        val_set = TDMatchDataset(info_val, config, data_augmentation=False)
        testing_set = TDMatchDataset(info_benchmark, config, data_augmentation=False)
    #elif config.dataset == 'original_tdmatch':
    #    training_set = OriginalTDMatchDataset(phase='train', self_training=config.self_training, data_augmentation=True, config=config)
    #    val_set = OriginalTDMatchDataset(phase='val', self_training=False, data_augmentation=True, config=config)
    #    testing_set = OriginalTDMatchDataset(phase='test', self_training=False, data_augmentation=False, config=config)
    elif config.dataset == 'modelnet40':
        training_set = ModelNet40(configs=config, num_points=config.n_points, num_subsampled_points=config.n_subsampled_points, partition='train', gaussian_noise=config.gaussian_noise,
                                 unseen=config.unseen, rot_factor=config.rot_factor)
        val_set = ModelNet40(configs=config, num_points=config.n_points, num_subsampled_points=config.n_subsampled_points, partition='test', gaussian_noise=config.gaussian_noise,
                             unseen=config.unseen, rot_factor=config.rot_factor)
        testing_set = ModelNet40(configs=config, num_points=config.n_points, num_subsampled_points=config.n_subsampled_points,
                             partition='test', gaussian_noise=config.gaussian_noise,
                             unseen=config.unseen, rot_factor=config.rot_factor)
    else:
        raise NotImplementedError

    return training_set, val_set, testing_set


def get_dataloader(dataset, sampler=None, batch_size=1, num_workers=8, shuffle=True, drop_last=True):
    '''
    Get the pytorch dataloader for specific pytorch dataset
    :param dataset: pytorch dataset
    :param batch_size: size of a batch of data
    :param num_workers: the number of threads used in dataloader
    :param shuffle: whether to shuffle dataset for each epoch
    :return: pytorch dataloader
    '''
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        collate_fn=partial(collate_fn, config=dataset.config),
        drop_last=drop_last
    )
    return data_loader

