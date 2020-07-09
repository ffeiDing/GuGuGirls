"""Refactor file directories, save/rename images and partition the 
train/val/test set, in order to support the unified dataset interface.
"""

from __future__ import print_function

import sys
sys.path.insert(0, '.')

from zipfile import ZipFile
import os
import os.path as osp
import numpy as np
import argparse

from tri_loss.utils.utils import may_make_dir
from tri_loss.utils.utils import save_pickle
from tri_loss.utils.utils import load_pickle

from tri_loss.utils.dataset_utils import get_im_names
from tri_loss.utils.dataset_utils import partition_train_val_set
from tri_loss.utils.dataset_utils import new_im_name_tmpl
from tri_loss.utils.dataset_utils import parse_im_name as parse_new_im_name
from tri_loss.utils.dataset_utils import move_ims
import pickle
import cv2
import random
ospj = osp.join
ospeu = osp.expanduser



def partitions(save_path, test_dataset_path):
    new_trainval_im_names = []
    new_trainval_im_ids = []

    train_ids=list(set(new_trainval_im_ids))
    train_ids.sort()

    train_ids2labels=dict(zip(train_ids,range(len(train_ids))))

    #Test_set
    all_im_names = []
    all_im_ids=[]
    name_list = os.listdir(test_dataset_path)
    name_list.sort()
    id_num=0
    odd = 0
    for id_name in name_list:
        id_name_path = test_dataset_path+'/'+id_name
        if not os.path.isdir(id_name_path):
            continue
        id_name_list = os.listdir(id_name_path)
        id_name_list.sort()
        for im_name in id_name_list:
            if im_name[-4:]=='.jpg' and 'mask' not in im_name:
                all_im_names.append(osp.join(id_name_path,im_name))

                all_im_ids.append(str(id_num).zfill(4))
                print(im_name)
                print(id_num)
        
        if odd%2==1:
          id_num = id_num+1
        odd = odd + 1
    all_ids=list(set(all_im_ids))
    all_ids.sort()
    print('all_im_names',len(all_im_names))
    print('all_ids',len(all_ids))



    test_im_names=[]
    test_im_ids=[]
    test_marks=[]
    test_im_cams=[]    
    ##val is test
    q = 0
    g = 0
 
    for i in range(len(all_im_names)):
      # test_im_name = test_im_names[i]
      mask_id = int(all_im_names[i].split('_')[-2]) 


      if mask_id==1:
        test_im_names.append(all_im_names[i])
        # print(test_im_names)
        test_im_ids.append(all_im_ids[i])
        test_marks.append(0)
        test_im_cams.append(0)
        q = q+1
      else:
        test_im_names.append(all_im_names[i])
        # print(test_im_names)
        test_im_ids.append(all_im_ids[i])
        test_marks.append(1)
        test_im_cams.append(1)
        g = g+1
      

    new_partitions = {
        'trainval_im_names': new_trainval_im_names,
        'trainval_im_ids': new_trainval_im_ids,
        'trainval_ids2labels': train_ids2labels,
        'test_marks': test_marks,
        'test_im_names':test_im_names,
        'test_ids': test_im_ids,
        'test_cams':test_im_cams
    }
    save_pickle(new_partitions, save_path)
    print('Partition file saved to {}'.format(save_path))
    print('trainval_im_names',len(new_trainval_im_ids))
    print('trainval_im_ids',len(np.unique(new_trainval_im_ids)))
    print('test_im_names',len(test_im_names))
    print('query',q)
    print('gallery',g)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset_path', type=str, default="/content/drive/My Drive/MTCNN_align_600id")
    args = parser.parse_args()
    test_dataset_path = args.test_dataset_path
    test_partition_path = osp.join(test_dataset_path, "test_partition.pkl")
    partitions(test_partition_path, test_dataset_path)
