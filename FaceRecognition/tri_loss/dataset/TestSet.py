from __future__ import print_function
import sys
import time,math
import os.path as osp
import torch
from PIL import Image
import numpy as np
from collections import defaultdict
ospeu = osp.expanduser
from .Dataset import Dataset

from ..utils.utils import measure_time,may_make_dir
from ..utils.re_ranking import re_ranking
from ..utils.metric import cmc, mean_ap
from ..utils.dataset_utils import parse_im_name
from ..utils.distance import normalize
from ..utils.distance import compute_dist
import pickle
from ..utils.utils import load_pickle
import torch.nn.functional as F


# from re_ranking import re_ranking


def save_pickle(obj, path):
    """Create dir and save file."""
    may_make_dir(osp.dirname(osp.abspath(path)))
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=2)

def fliplr(img):
    img = torch.from_numpy(img)
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    img_flip = img.index_select(3, inv_idx)
    img_flip = img_flip.numpy()
    return img_flip

def cosine(t1,t2):
    d=torch.mul(t1,t2).sum()
    d1=torch.mul(t1,t1).sum()
    d2=torch.mul(t2,t2).sum()
    if d1*d2==0:
        return 0
    else:
        return d/math.sqrt(d1*d2)

def calculateGraph(featMap):
    m=torch.zeros(featMap.size(1)*featMap.size(2),featMap.size(1)*featMap.size(2))
    for i in range(featMap.size(1)*featMap.size(2)):
      for j in range(featMap.size(1)*featMap.size(2)):
        h=i%featMap.size(1)
        w=i//featMap.size(1)
        t1=featMap[:,h,w]

        h = j % featMap.size(1)
        w = j // featMap.size(1)

        t2 = featMap[:, h, w]


        # print(t1.size(),t2.size())
        # print(t2)
        m[i,j]=cosine(t1,t2)
    print(torch.max(m),torch.min(m),torch.sum(m)/(192*192))

    return m








class TestSet(Dataset):
  """
  Args:
    extract_feat_func: a function to extract features. It takes a batch of
      images and returns a batch of features.
    marks: a list, each element e denoting whether the image is from 
      query (e == 0), or
      gallery (e == 1), or 
      multi query (e == 2) set
  """

  def __init__(
      self,
      im_dir="",
      im_names=None,
      im_ids = None,
      im_cams = None,
      marks=None,
      extract_feat_func=None,
      separate_camera_set=None,
      single_gallery_shot=None,
      first_match_break=None,
      im_type='p',
      **kwargs):

    super(TestSet, self).__init__(dataset_size=len(im_names), **kwargs)

    # The im dir of all images
    self.im_dir = im_dir
    self.im_names = im_names
    self.marks = marks
    self.extract_feat_func = extract_feat_func
    self.separate_camera_set = separate_camera_set
    self.single_gallery_shot = single_gallery_shot
    self.first_match_break = first_match_break
    self.im_type=im_type
    self.im_ids=im_ids
    self.im_cams=im_cams
  def set_feat_func(self, extract_feat_func):
    self.extract_feat_func = extract_feat_func

  def get_sample(self, ptr):
    im_name = self.im_names[ptr]
    im_path = osp.join(self.im_dir, im_name)
    im = np.asarray(Image.open(im_path).convert("RGB"))
    im, _ = self.pre_process_im(im)
    id=self.im_ids[ptr]
    cam=self.im_cams[ptr]
    # denoting whether the im is from query, gallery, or multi query set
    if self.im_type=='p':
      mark = self.marks[ptr]
    else:
      mark=None
    return im, id, cam, im_name, mark

  def next_batch(self):
    if self.epoch_done and self.shuffle:
      self.prng.shuffle(self.im_names)
    samples, self.epoch_done = self.prefetcher.next_batch()
    im_list, ids, cams, im_names, marks = zip(*samples)
    # Transform the list into a numpy array with shape [N, ...]
    ims = np.stack(im_list, axis=0)
    ids = np.array(ids)
    cams = np.array(cams)
    im_names = np.array(im_names)
    if self.im_type=='p':
      marks = np.array(marks)
      return ims, ids, cams, im_names, marks, self.epoch_done
    else:
      marks=None
      return ims, ids, cams, im_names, marks, self.epoch_done

  def extract_feat(self, normalize_feat, verbose=True):
    """Extract the features of the whole image set.
    Args:
      normalize_feat: True or False, whether to normalize feature to unit length
      verbose: whether to print the progress of extracting feature
    Returns:
      feat: numpy array with shape [N, C]
      ids: numpy array with shape [N]
      cams: numpy array with shape [N]
      im_names: numpy array with shape [N]
      marks: numpy array with shape [N]
    """
    feat1, ids, cams, im_names, marks = [], [], [], [], []
    done = False
    step = 0
    printed = False
    st = time.time()
    last_time = time.time()
    while not done:
      ims_, ids_, cams_, im_names_, marks_, done = self.next_batch()
      feat1_= self.extract_feat_func(ims_)
      feat_flip= self.extract_feat_func(fliplr(ims_))
      feat1_ = feat1_ + feat_flip
      feat1.append(feat1_)
      ids.append(ids_)
      cams.append(cams_)
      im_names.append(im_names_)
      marks.append(marks_)

      if verbose:
        # Print the progress of extracting feature
        total_batches = (self.prefetcher.dataset_size
                         // self.prefetcher.batch_size + 1)
        step += 1
        if step % 20 == 0:
          if not printed:
            printed = True
          else:
            # Clean the current line
            sys.stdout.write("\033[F\033[K")
          print('{}/{} batches done, +{:.2f}s, total {:.2f}s'
                .format(step, total_batches,
                        time.time() - last_time, time.time() - st))
          last_time = time.time()

    feat1 = np.vstack(feat1)
    ids = np.hstack(ids)
    cams = np.hstack(cams)
    im_names = np.hstack(im_names)
    marks = np.hstack(marks)

    return feat1, ids, cams, im_names, marks



  def eval(
      self,
      normalize_feat=False,
      to_re_rank=False,
      pool_type='average',
      verbose=True):

    """Evaluate using metric CMC and mAP.
    Args:
      normalize_feat: whether to normalize features before computing distance
      to_re_rank: whether to also report re-ranking scores
      pool_type: 'average' or 'max', only for multi-query case
      verbose: whether to print the intermediate information
    """

    with measure_time('Extracting feature...', verbose=verbose):
       feat1, ids, cams, im_names, marks = self.extract_feat(
         normalize_feat, verbose)


    # query, gallery, multi-query indices
    q_inds = marks == 0
    g_inds = marks == 1

    # A helper function just for avoiding code duplication.
    def compute_score(
        dist_mat,
        query_ids=ids[q_inds],
        gallery_ids=ids[g_inds],
        query_cams=cams[q_inds],
        gallery_cams=cams[g_inds]):
      # Compute mean AP
      mAP = mean_ap(
        distmat=dist_mat,
        query_ids=query_ids, gallery_ids=gallery_ids,
        query_cams=query_cams, gallery_cams=gallery_cams)
      # Compute CMC scores
      cmc_scores, idx = cmc(
        distmat=dist_mat,
        query_ids=query_ids, gallery_ids=gallery_ids,
        query_cams=query_cams, gallery_cams=gallery_cams,
        separate_camera_set=self.separate_camera_set,
        single_gallery_shot=self.single_gallery_shot,
        first_match_break=self.first_match_break,
        topk=10)

      result = [[]for _ in range(len(query_ids))]
      query_name = [[]for _ in range(len(query_ids))]
      q_names = im_names[q_inds]
      g_names = im_names[g_inds]
      for i in range(len(query_ids)):
        query_name[i].append(q_names[i])
        for j in range(10):
          result[i].append(g_names[idx[i][j]])
      
      pickle_file = open('result.pkl','wb')
      pickle.dump(result,pickle_file)
      pickle_file.close()
      pickle_file = open('query_name.pkl','wb')
      pickle.dump(query_name,pickle_file)
      pickle_file.close()
      
      return mAP, cmc_scores

    def print_scores(mAP, cmc_scores):
      print('[mAP: {:5.2%}], [cmc1: {:5.2%}], [cmc5: {:5.2%}], [cmc10: {:5.2%}]'
            .format(mAP, *cmc_scores[[0, 4, 9]]))

    ################
    # Single Query #
    ################

    with measure_time('Computing distance...', verbose=verbose):
      # query-gallery distance
      q_g_dist_list=[]
      q_g_dist1 = compute_dist(feat1[q_inds], feat1[g_inds], type='cosine')
    with measure_time('Computing scores...', verbose=verbose):
      mAP, cmc_scores = compute_score(q_g_dist1)

    print('{:<30}'.format('Single Query:'), end='')
    print_scores(mAP, cmc_scores)
    return mAP,cmc_scores
