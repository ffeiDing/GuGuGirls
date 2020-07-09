
from __future__ import print_function

import sys
import warnings
import pickle
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')

import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic=True
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parallel import DataParallel

import time
import os.path as osp
import numpy as np
import argparse
import json
from sklearn.cluster import KMeans, SpectralClustering

from tri_loss.dataset import create_dataset
from tri_loss.dataset.TestSet import TestSet
from tri_loss.model.model import Model
from tri_loss.model.loss import *
from tri_loss.utils.utils import *
from tri_loss.utils.utils import tight_float_str as tfs


class Config(object):
    def __init__(self):
        
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,))
        parser.add_argument('-r', '--run', type=int, default=1)
        parser.add_argument('--set_seed', type=str2bool, default=False)
        parser.add_argument('--partition_path', type=str, default='/content/drive/My Drive/mask/partition.pkl')
        parser.add_argument('--trainset_part', type=str, default='trainval')
        parser.add_argument('--resize_h_w', type=eval, default=(128,128))#(128, 128))
        # These several only for training set
        parser.add_argument('--crop_prob', type=float, default=0.0)
        parser.add_argument('--crop_ratio', type=float, default=0.9)
        parser.add_argument('--rotate_prob', type=float, default=0.5)
        parser.add_argument('--rotate_degree', type=float, default=0)
        parser.add_argument('--mirror', type=str2bool, default=True)
        parser.add_argument('--is_pool', type=str2bool, default=True)


        parser.add_argument('--ids_per_batch', type=int, default=16)
        parser.add_argument('--ims_per_id', type=int, default=8)
        parser.add_argument('--batch_size', type=int, default=64)  # Testing Batch
        
        parser.add_argument('--log_to_file', type=str2bool, default=True)
        parser.add_argument('--steps_per_log', type=int, default=5)
        parser.add_argument('--epochs_per_val', type=int, default=1)
        parser.add_argument('--epochs_per_cluster', type=int, default=5)
        
        parser.add_argument('--last_conv_stride', type=int, default=1,
                            choices=[1, 2, 11])
        parser.add_argument('--normalize_feature', type=str2bool, default=False)
        parser.add_argument('--margin', type=float, default=0.5)
        parser.add_argument('--loss', type=str, default="Softmax")
        
        parser.add_argument('--only_test', type=str2bool, default=False)
        parser.add_argument('--resume', type=str2bool, default=False)
        parser.add_argument('--exp_dir', type=str, default='/content/drive/My Drive/GuGuGirls/FaceRecognition/logs')
        parser.add_argument('--model_weight_file', type=str, default='/content/drive/My Drive/GuGuGirls/FaceRecognition/logs/ckpt_best.pth')
        parser.add_argument('--model', type=str, default='Resnet50')
        
        parser.add_argument('--base_lr', type=float, default=0.01)#
        parser.add_argument('--lr_decay_type', type=str, default='warmup',
                            choices=['exp', 'staircase', 'warmup'])
        parser.add_argument('--exp_decay_at_epoch', type=int, default=61)  # 41
        parser.add_argument('--staircase_decay_at_epochs',
                            type=eval, default=(20, 60, 100))
        parser.add_argument('--staircase_decay_multiply_factor',
                            type=float, default=0.1) 
        parser.add_argument('--total_epochs', type=int, default=150)
        
        args = parser.parse_args()
        
        # gpu ids
        self.sys_device_ids = args.sys_device_ids
        
        # If you want to make your results exactly reproducible, you have
        # to fix a random seed.
        if args.set_seed:
            self.seed = 1
        else:
            self.seed = None
        
        if self.seed is not None:
            self.prefetch_threads = 1
        else:
            self.prefetch_threads = 1

        self.is_pool = args.is_pool
        
        # The experiments can be run for several times and performances be averaged.
        # `run` starts from `1`, not `0`.
        self.run = args.run
        
        ###########
        # Dataset #
        ###########
        self.partition_path = args.partition_path
        self.trainset_part = args.trainset_part
        
        # Image Processing
        self.crop_prob = args.crop_prob
        self.crop_ratio = args.crop_ratio
        self.resize_h_w = args.resize_h_w
        self.rotate_prob = args.rotate_prob
        self.rotate_degree=args.rotate_degree
        self.loss = args.loss
        # Whether to scale by 1/255
        self.scale_im = True
        self.im_mean = [0.486, 0.459, 0.408]
        self.im_std = [0.229, 0.224, 0.225]
        
        self.ids_per_batch = args.ids_per_batch
        self.ims_per_id = args.ims_per_id
        
        # training
        self.train_mirror_type = 'random' if args.mirror else None
        self.train_final_batch = False
        self.train_shuffle = True  # True
        
        self.test_batch_size = args.batch_size
        self.test_final_batch = True
        self.test_mirror_type = None
        self.test_shuffle = False
        
        dataset_kwargs = dict(
            resize_h_w=self.resize_h_w,
            scale=self.scale_im,
            im_mean=self.im_mean,
            im_std=self.im_std,
            batch_dims='NCHW',
            num_prefetch_threads=self.prefetch_threads)

        # train set
        prng = np.random
        if self.seed is not None:
            prng = np.random.RandomState(self.seed)
        self.train_set_kwargs = dict(
            path=self.partition_path,
            part=self.trainset_part,
            ids_per_batch=self.ids_per_batch,
            ims_per_id=self.ims_per_id,
            final_batch=self.train_final_batch,
            shuffle=self.train_shuffle,
            crop_prob=self.crop_prob,
            crop_ratio=self.crop_ratio,
            rotate_prob=self.rotate_prob,
            rotate_degree=self.rotate_degree,
            mirror_type=self.train_mirror_type,
            prng=prng)
        self.train_set_kwargs.update(dataset_kwargs)
        
        # test set
        prng = np.random
        if self.seed is not None:
            prng = np.random.RandomState(self.seed)
        self.test_set_kwargs = dict(
            path=self.partition_path,
            part='test',
            batch_size=self.test_batch_size,
            final_batch=self.test_final_batch,
            shuffle=self.test_shuffle,
            mirror_type=self.test_mirror_type,
            crop_prob=self.crop_prob,
            crop_ratio=self.crop_ratio,
            rotate_prob=0.0,
            prng=prng)
        self.test_set_kwargs.update(dataset_kwargs)

        
        ###############
        # ReID Model  #
        ###############
        # The last block of ResNet has stride 2. We can set the stride to 1 so that
        # the spatial resolution before global pooling is doubled.
        self.last_conv_stride = args.last_conv_stride
        
        # Whether to normalize feature to unit length along the Channel dimension,
        # before computing distance
        self.normalize_feature = args.normalize_feature
        
        # Margin of triplet loss(inter triplet, intra triplet)
        self.margin = args.margin
        self.model = args.model
        
        #############
        # Training  #
        #############
        self.weight_decay = 0.0005
        # Initial learning rate
        self.base_lr = args.base_lr
        self.lr_decay_type = args.lr_decay_type
        self.exp_decay_at_epoch = args.exp_decay_at_epoch
        self.staircase_decay_at_epochs = args.staircase_decay_at_epochs
        self.staircase_decay_multiply_factor = args.staircase_decay_multiply_factor
        # Number of epochs to train
        self.total_epochs = args.total_epochs
        
        # How often (in epochs) to test on val set.
        self.epochs_per_val = args.epochs_per_val
        self.epochs_per_cluster = args.epochs_per_cluster
        
        # How often (in batches) to log. If only need to log the average
        # information for each epoch, set this to a large value, e.g. 1e10.
        self.steps_per_log = args.steps_per_log
        self.only_test = args.only_test
        self.resume = args.resume
        
        #######
        # Log #
        #######
        
        # If True,
        # 1) stdout and stderr will be redirected to file,
        # 2) training loss etc will be written to tensorboard,
        # 3) checkpoint will be saved
        self.log_to_file = args.log_to_file
        
        # The root dir of logs.
        if args.exp_dir == '':
            self.exp_dir = osp.join(
                '{}'.format(self.model),
                '{}'.format(self.dataset),
                #
                'BR_lcs_{}_'.format(self.last_conv_stride) +
                'margin_{}_'.format(tfs(self.margin)) +
                'erasing_{}_'.format(self.rotate_prob) +
                '_{}_'.format(self.crop_prob) +
                'ids_{}_'.format(tfs(self.ids_per_batch)) +
                'ims_{}_'.format(tfs(self.ims_per_id)) +
                'lr_{}_'.format(tfs(self.base_lr,fmt='{:.7f}')) +
                '{}_'.format(self.lr_decay_type) +
                ('decay_at_{}_'.format(self.exp_decay_at_epoch)
                 if self.lr_decay_type == 'exp'
                 else 'decay_at_{}_factor_{}_'.format(
                    '_'.join([str(e) for e in args.staircase_decay_at_epochs]),
                    tfs(self.staircase_decay_multiply_factor))) +
                'total_{}'.format(self.total_epochs),
            )
        else:
            self.exp_dir = args.exp_dir
        
        self.stdout_file = osp.join(
            self.exp_dir, 'stdout_{}.txt'.format(time_str()))
        self.stderr_file = osp.join(
            self.exp_dir, 'stderr_{}.txt'.format(time_str()))
        
        # Saving model weights and optimizer states, for resuming.
        self.model_file=osp.join(
                '{}'.format(self.model))
        self.ckpt_file = osp.join(self.exp_dir, 'ckpt.pth')
        # Just for loading a pretrained model; no optimizer states is needed.
        self.model_weight_file = args.model_weight_file


class ExtractFeature(object):
    """
    A function to be called in the val/test set, to extract features.
    Args:
        TVT: A callable to transfer images to specific device.
    """
    
    def __init__(self, model, TVT=None, test=False):
        # for param in model.parameters():  # model.module.parameters
        #     param.requires_grad = False
        self.model = model
        self.TVT = TVT
        self.test = test
    
    def __call__(self, ims):
        old_train_eval_model = self.model.training
        # Set eval mode: Force all BN layers to use global mean and variance,
        # also disable dropout.
        self.model.eval()

        ims = Variable(self.TVT(torch.from_numpy(ims).float()))
        feat,_, _= self.model(ims, [])
        feat=feat.data.cpu()
        #feat = torch.cat((feat, id_feat), 1)
        # feat = feat.cpu()
        self.model.train(old_train_eval_model)
        return feat


class ExtractFeatureMap(object):
    """
    A function to be called in the val/test set, to extract features.
    Args:
        TVT: A callable to transfer images to specific device.
    """

    def __init__(self, model, TVT=None, test=False):
        # for param in model.parameters():  # model.module.parameters
        #     param.requires_grad = False
        self.model = model
        self.TVT = TVT
        self.test = test

    def __call__(self, ims):
        old_train_eval_model = self.model.training
        # Set eval mode: Force all BN layers to use global mean and variance,
        # also disable dropout.
        self.model.eval()

        ims = Variable(self.TVT(torch.from_numpy(ims).float()))
        feat, _, _ = self.model(ims)
        feat = feat.data.cpu()
        #featMap=featMap.data.cpu()
        # feat = torch.cat((feat, id_feat), 1)
        # feat = feat.cpu()
        self.model.train(old_train_eval_model)
        return feat#,featMap



def deep_clone_model(cfg, model, train_set_nums, TMO):

    state_dict = model.state_dict()
    model_copy = Model(last_conv_stride=cfg.last_conv_stride, num_classes=train_set_nums)
    model_copy.load_state_dict(state_dict)
    for param in model_copy.parameters():  # model.module.parameters
        param.requires_grad = False
    TMO([model_copy])
    return model_copy





def main():
    cfg = Config()
    
    # Redirect logs to both console and file.
    if cfg.log_to_file:
        ReDirectSTD(cfg.stdout_file, 'stdout', True)
        # ReDirectSTD(cfg.stderr_file, 'stderr', False)
    
    # Lazily create SummaryWriter
    writer = None
    
    TVT, TMO = set_devices(cfg.sys_device_ids)
    
    if cfg.seed is not None:
        set_seed(cfg.seed)
    
    # Dump the configurations to log.
    import pprint
    print('-' * 60)
    print('cfg.__dict__')
    pprint.pprint(cfg.__dict__)
    print('-' * 60)
    
    ###########
    # Dataset #
    ###########
    train_set = create_dataset(**cfg.train_set_kwargs)
    test_set = create_dataset(**cfg.test_set_kwargs)

    
    ###########
    # Models  #
    ###########
    model = Model(last_conv_stride=cfg.last_conv_stride, num_classes = 10572,basenet=cfg.model, 
                  loss=cfg.loss, is_pool=cfg.is_pool)
    # model.load_state_dict(torch.load(osp.join(cfg.exp_dir, 'model.pth')))
    # Model wrapper
    model_w = DataParallel(model)
    
    #############################
    # Criteria and Optimizers   #
    #############################
    softmax_criterion = torch.nn.CrossEntropyLoss()
    
    base_params = list(model.base.parameters())  # finetuned
    new_params = [p for n, p in model.named_parameters()
                  if not n.startswith('base.')]
    param_groups = [{'params': base_params, 'lr': cfg.base_lr},
                    {'params': new_params, 'lr': cfg.base_lr}]

    optimizer = optim.SGD(param_groups,momentum = 0.9)    
    # Bind them together just to save some codes in the following usage.
    modules_optims = [model, optimizer]
    
    ################################
    # May Resume Models and Optims #
    ################################
    min_loss = 9999
    if cfg.resume:
        if cfg.model_weight_file != '':
            resume_ep, min_loss = load_ckpt(modules_optims, cfg.model_weight_file)
        else:
            resume_ep, min_loss = load_ckpt(modules_optims, cfg.ckpt_file)
    
    # May Transfer Models and Optims to Specified Device. Transferring optimizer
    # is to cope with the case when you load the checkpoint to a new device.
    TMO(modules_optims)
    
    ###########
    # Testing #
    ###########
    # def test(model, test_set):
    #     print('\n=========> Test on dataset: {} <=========\n'.format(cfg.dataset))
    #     test_set.set_feat_func(ExtractFeature(model, TVT, test=True))
    #
    #     cmc1, cmc5, cmc10, mAP = test_set.eval(
    #         normalize_feat=cfg.normalize_feature,
    #         verbose=True)
    #     print('\n=========> cmc1: {}, cmc5: {}, cmc10: {}, mAP: {} <=========\n'.
    #           format(cmc1, cmc5, cmc10, mAP))
    #     return cmc1, cmc5, cmc10, mAP

    def test(test_set,load_model_weight=False):
        time_start = time.time()
        if load_model_weight:
            if cfg.model_weight_file != '':
                print(1)
                map_location = (lambda storage, loc: storage)
                sd = torch.load(cfg.model_weight_file, map_location=map_location)
                load_state_dict(model, sd)
                print('Loaded model weights from {}'.format(cfg.model_weight_file))
            else:
                print(2)
                # print(modules_optims)
                load_ckpt(modules_optims, cfg.ckpt_file)
        model.eval()
        test_set.set_feat_func(ExtractFeature(model_w, TVT,test=True))
        mAP, cmc_scores= test_set.eval(
            normalize_feat=cfg.normalize_feature,
            verbose=True)
        # mAP, cmc_scores, _, _ = test_set.eval_BR(
        #     normalize_feat=cfg.normalize_feature,
        #     verbose=True)
        return mAP, cmc_scores[0]

    def buildGraph(test_set,load_model_weight=False):
        if load_model_weight:
            if cfg.model_weight_file != '':
                print(1)
                map_location = (lambda storage, loc: storage)
                sd = torch.load(cfg.model_weight_file, map_location=map_location)
                load_state_dict(model, sd)
                print('Loaded model weights from {}'.format(cfg.model_weight_file))
            else:
                print(2)
                load_ckpt(modules_optims, cfg.ckpt_file)
        test_set.set_feat_func(ExtractFeatureMap(model_w, TVT,test=True))
        test_set.buildGraph()


    ###########
    # Testing #
    ###########
    if cfg.only_test:
        # model_copy = deep_clone_model(cfg, model, len(train_set.ids), TMO)
        print('Testing with model {}'.format(cfg.ckpt_file))
        mAP, rank1 = test(test_set,load_model_weight=True)
        # buildGraph(test_set, load_model_weight=True)
        return
    
    ############
    # Training #
    ############
    #test(test_set,load_model_weight=False)
    start_ep = resume_ep if cfg.resume else 0
    best_rank=0
    best_epoch=0
    for ep in range(start_ep, cfg.total_epochs):
        # Adjust Learning Rate
        if cfg.lr_decay_type == 'exp':
            adjust_lr_exp(
                optimizer,
                cfg.base_lr,
                ep + 1,
                cfg.total_epochs,
                cfg.exp_decay_at_epoch)
        elif cfg.lr_decay_type == 'staircase':
            adjust_lr_staircase(
                optimizer,
                cfg.base_lr,
                ep + 1,
                cfg.staircase_decay_at_epochs,
                cfg.staircase_decay_multiply_factor)
        elif cfg.lr_decay_type == 'warmup':
            adjust_lr_warmup(
                optimizer,
                cfg.base_lr,
                ep + 1,
                cfg.staircase_decay_at_epochs,
                cfg.staircase_decay_multiply_factor)
        
        may_set_mode(modules_optims, 'train')
        
        # For recording precision, satisfying margin, etc
        loss_meter = AverageMeter()
        loss_softmax_meter1 = AverageMeter()
        loss_softmax_meter2 = AverageMeter()
        loss_softmax_meter3 = AverageMeter()
        ep_st = time.time()
        step_st = time.time()
        step = 0
        epoch_done = False
        train_set.epoch = ep + 1
        
        
        while not epoch_done:
            step += 1
            # step_st = time.time()
            ims, im_names, labels, mirrored,  epoch_done, mask_labels = train_set.next_batch()
            # print(type(idxs))

            ims_var = Variable(TVT(torch.from_numpy(ims).float()))
            labels_t = TVT(torch.from_numpy(labels).long())
            labels_var = Variable(TVT(torch.from_numpy(labels).long()))
            
            feat, logit1, logit3 = model_w(ims_var, labels_var)

            loss1 = softmax_criterion(logit1, labels_var)
            loss2 = softmax_criterion(logit3, labels_var)
            loss3 = loss1+loss2
            
            loss = loss1+loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            loss_meter.update(to_scalar(loss))

            loss_softmax_meter1.update(to_scalar(loss1))
            loss_softmax_meter2.update(to_scalar(loss2))
            loss_softmax_meter3.update(to_scalar(loss3))
            if step % cfg.steps_per_log == 0:
                time_log = '\tStep {}/Ep {}, {:.2f}s'.format(
                    step, ep + 1, time.time() - step_st, )
                
                tri_log = (', loss softmax1 {:.4f}, loss softmax2 {:.4f}, loss softmax {:.4f}'.format(
                    loss_softmax_meter1.val, loss_softmax_meter2.val, loss_softmax_meter3.val))
                
                log = time_log + tri_log
                print(log)
                step_st = time.time()


        #############
        # Epoch Log #
        #############
        time_log = 'Ep {}, {:.2f}s'.format(ep + 1, time.time() - ep_st)
        tri_log = (', loss softmax1 {:.4f}, loss softmax2 {:.4f}, loss {:.4f}'.format(
             loss_softmax_meter1.avg,  loss_softmax_meter2.avg, loss_meter.avg))
        log = time_log + tri_log
        print(log)
        
        # save ckpt
        # if cfg.log_to_file and min_loss > loss_meter.avg:
        #     min_loss = loss_meter.avg
        #     save_ckpt(modules_optims, ep + 1, min_loss, cfg.ckpt_file)
        
        ##########################
        # Test on Test Set #
        ##########################
        mAP, Rank1 = 0, 0
        if (ep + 1) % cfg.epochs_per_val == 0:
            # if 'model_copy' not in locals().keys():
            #     model_copy = deep_clone_model(cfg, model, len(train_set.ids), TMO)
            mAP,Rank1 = test(test_set,load_model_weight=False)
            if Rank1 > best_rank:
                best_rank = Rank1
                best_epoch = ep+1
                save_ckpt(modules_optims, ep + 1, min_loss, osp.join(cfg.exp_dir, 'ckpt_best.pth'))
            print(best_epoch)

        if cfg.log_to_file:
            save_ckpt(modules_optims, ep + 1, min_loss, cfg.ckpt_file)
            #best_rank=Rank1
       

    test(test_set, load_model_weight=False)



if __name__ == '__main__':
    main()

