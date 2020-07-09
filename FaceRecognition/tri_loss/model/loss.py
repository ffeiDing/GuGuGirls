from __future__ import print_function
import torch
import os
import torch.nn as nn
import numpy as np

def normalize(x, axis=-1):
    """
    Normalizing to unit length along the specified dimension.
      Args:
        x: pytorch Variable
      Returns:
        x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
      Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
      Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels):  #group_inds
    """
    For each anchor, find the hardest positive and negative sample.
      Args:
        dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
        labels: pytorch LongTensor, with shape [N]
        return_inds: whether to return the indices. Save time if `False`(?)
      Returns:
        dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
        dist_an: pytorch Variable, distance(anchor, negative); shape [N]
        p_inds: pytorch LongTensor, with shape [N];
          indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
        n_inds: pytorch LongTensor, with shape [N];
          indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
      NOTE: Only consider the case in which all labels have same num of samples,
        thus we can cope with all anchors in parallel.
  """
    
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    # if N < 64:
    #     print(N)
    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    # choose the most dissimilarity positive sample

    # get hard sample
    # dist_ap = torch.FloatTensor().cuda()
    # dist_an = torch.FloatTensor().cuda()
    # dist_pos = dist_mat[is_pos].contiguous().view(N, -1)
    # dist_neg = dist_mat[is_neg].contiguous().view(N, -1)
    # # tmp = dist_pos.cpu().data.numpy()
    # # tmp2 = dist_neg.cpu().data.numpy()
    # for i in range(0, N, 4):
    #     dist_ap = torch.cat((dist_ap, dist_pos[i, 1:4]), 0)
    #     an, _ = torch.min(dist_neg[i], 0, keepdim=True)
    #     dist_an = torch.cat((dist_an, an), 0)
    #     dist_an = torch.cat((dist_an, an), 0)
    #     dist_an = torch.cat((dist_an, an), 0)

    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    dist_ap = dist_ap.squeeze(1)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    dist_an = dist_an.squeeze(1)
    # if return_inds:
    #     # shape [N, N]
    #     ind = (labels.new().resize_as_(labels)
    #            .copy_(torch.arange(0, N).long())
    #            .unsqueeze(0).expand(N, N))
    #     # shape [N, 1]
    #     p_inds = torch.gather(
    #         ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
    #     n_inds = torch.gather(
    #         ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
    #     # shape [N]
    #     p_inds = p_inds.squeeze(1)
    #     n_inds = n_inds.squeeze(1)
    #     return dist_ap, dist_an, p_inds, n_inds
    
    return dist_ap, dist_an


def global_loss(tri_loss, global_feat, labels,normalize_feature):
    """
      Args:
        tri_loss: a `TripletLoss` object
        global_feat: pytorch Variable, shape [N, C]
        labels: pytorch LongTensor, with shape [N]
        normalize_feature: whether to normalize feature to unit length along the
          Channel dimension
      Returns:
        loss: pytorch Variable, with shape [1]
        p_inds: pytorch LongTensor, with shape [N];
          indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
        n_inds: pytorch LongTensor, with shape [N];
          indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
        ==================
        For Debugging, etc
        ==================
        dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
        dist_an: pytorch Variable, distance(anchor, negative); shape [N]
        dist_mat: pytorch Variable, pairwise euclidean distance; shape [N, N]
  """
    if normalize_feature:
        global_feat = normalize(global_feat, axis=-1)
    # shape [N, N]
    dist_mat = euclidean_dist(global_feat, global_feat)
    dist_ap, dist_an = hard_example_mining(dist_mat, labels)
    loss = tri_loss(dist_ap, dist_an)
    return loss, dist_ap, dist_an#, dist_mat


def strict_sample_mining(dist_mat, labels):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    dist_ap = torch.diag(dist_mat)

    dist_an, _ = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    dist_an = dist_an.squeeze(1)
    return dist_ap, dist_an


def cluster_loss(loose_loss, strict_loss, feat, cluster_centers, labels, ims_per_id):
    N, feat_dim = feat.shape
    center0 = torch.FloatTensor().cuda()
    center1 = torch.FloatTensor().cuda()
    center2 = torch.FloatTensor().cuda()
    # 0, 3, 6, 9, ...
    for i in range(0, cluster_centers.shape[0], 3):
        # center = cluster_centers[i].expand(8, feat_dim)
        # if i % 2 == 0:
        #     center0 = torch.cat((center0, center), 0)
        # else:
        #     center1 = torch.cat((center1, center), 0)
        center0 = torch.cat((center0, cluster_centers[i].expand(ims_per_id, feat_dim)), 0)
        center1 = torch.cat((center1, cluster_centers[i+1].expand(ims_per_id, feat_dim)), 0)
        center2 = torch.cat((center2, cluster_centers[i+2].expand(ims_per_id, feat_dim)), 0)
        
    dist_mat0 = euclidean_dist(center0, feat)
    dist_mat1 = euclidean_dist(center1, feat)
    dist_mat2 = euclidean_dist(center2, feat)
    
    dist_ap_l = []
    dist_an_l = []
    dist_ap_s = []
    dist_an_s = []
    dist_ap0, dist_an0 = strict_sample_mining(dist_mat0, labels)
    dist_ap1, dist_an1 = strict_sample_mining(dist_mat1, labels)
    dist_ap2, dist_an2 = strict_sample_mining(dist_mat2, labels)
    for i in range(dist_ap0.shape[0]):
        min_idx = np.argmin([dist_ap0[i], dist_ap1[i], dist_ap2[i]], axis=0)
        max_idx = np.argmax([dist_ap0[i], dist_ap1[i], dist_ap2[i]], axis=0)
        # 近
        if min_idx == 0:
            dist_ap_s.append(dist_ap0[i])
            dist_an_s.append(dist_an0[i])
        elif min_idx == 1:
            dist_ap_s.append(dist_ap1[i])
            dist_an_s.append(dist_an1[i])
        else:
            dist_ap_s.append(dist_ap2[i])
            dist_an_s.append(dist_an2[i])
        # 远
        if max_idx == 0:
            dist_ap_l.append(dist_ap0[i])
            dist_an_l.append(dist_an0[i])
        elif max_idx == 1:
            dist_ap_l.append(dist_ap1[i])
            dist_an_l.append(dist_an1[i])
        else:
            dist_ap_l.append(dist_ap2[i])
            dist_an_l.append(dist_an2[i])
            
    # dist_ap_l = torch.stack(dist_ap_l)
    # dist_an_l = torch.stack(dist_an_l)
    dist_ap_s = torch.stack(dist_ap_s)
    dist_an_s = torch.stack(dist_an_s)
    
    # loss_l = loose_loss(dist_ap_l, dist_an_l)
    loss_s = strict_loss(dist_ap_s, dist_an_s)

    return loss_s #loss_l + loss_s


def multi_pose_sample_mining(dist_mat, labels, pose_labels, ims_per_id):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    dist_an = dist_an.squeeze(1)
    
    dist_mat_ap = dist_mat[is_pos].contiguous().view(N, -1)
    dist_ap = torch.FloatTensor().type_as(dist_an)
    for i in range(0, N, ims_per_id):
        id_sample_index = list(range(i, i+ims_per_id))
        id_sample_index = np.array(id_sample_index)
        # id_mat = dist_mat_ap[id_sample_index]
        id_pose_labels = pose_labels[id_sample_index].cpu().numpy()
        if id_pose_labels.min() == id_pose_labels.max():
            dist_ap_id, _ = torch.max(dist_mat_ap[id_sample_index], 1, keepdim=True)
            for ap in dist_ap_id:
                dist_ap = torch.cat((dist_ap, ap), 0)
        else:
            for j in id_sample_index:
                cur_pose_label = pose_labels[j].cpu().numpy()
                # print(type(id_pose_labels))
                # print(type(cur_pose_label))
                other_pose_idx = np.where(id_pose_labels != cur_pose_label)[0]
                # print(id_pose_labels)
                # print(cur_pose_label)
                # print(type(other_pose_idx))
                # print(np.where(id_pose_labels != cur_pose_label))
                # print(other_pose_idx)
                tmp = dist_mat_ap[j, other_pose_idx]
                dist_ap_id, _ = torch.max(dist_mat_ap[j, other_pose_idx], 0, keepdim=True)
                dist_ap = torch.cat((dist_ap, dist_ap_id), 0)
    
    return dist_ap, dist_an


def multi_pose_loss(tri_loss, global_feat, labels, pose_labels, ims_per_id):
    # shape [N, N]
    dist_mat = euclidean_dist(global_feat, global_feat)
    dist_ap, dist_an = multi_pose_sample_mining(dist_mat, labels, pose_labels, ims_per_id)
    loss = tri_loss(dist_ap, dist_an)
    return loss, dist_ap, dist_an  # , dist_mat


def single_pose_sample_mining(dist_mat, labels):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    
    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    
    dist_ap = torch.diag(dist_mat)
    
    dist_an, _ = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    dist_an = dist_an.squeeze(1)
    return dist_ap, dist_an


def single_pose_loss(tri_loss, feat, cluster_centers, labels, ims_per_id):
    N, feat_dim = feat.shape
    center0 = torch.FloatTensor()
    center1 = torch.FloatTensor()
    center2 = torch.FloatTensor()
    center3 = torch.FloatTensor()
    # center4 = torch.FloatTensor()
    ids_per_batch = len(cluster_centers)
    clusters_per_id = cluster_centers[0].shape[0]
    
    for i in range(ids_per_batch):
        for j in range(clusters_per_id):
            center = torch.from_numpy(cluster_centers[i][j])
            center = center.expand(ims_per_id, feat_dim)
            
            # center = cluster_centers[i][j].expand(ims_per_id, feat_dim)
            if j == 0:
                center0 = torch.cat((center0, center), 0)
            elif j == 1:
                center1 = torch.cat((center1, center), 0)
            elif j == 2:
                center2 = torch.cat((center2, center), 0)
            elif j == 3:
                center3 = torch.cat((center3, center), 0)
            # elif j == 4:
            #     center4 = torch.cat((center4, center), 0)
    center0 = center0.type_as(feat)
    center1 = center1.type_as(feat)
    center2 = center2.type_as(feat)
    center3 = center3.type_as(feat)
    # center4 = center4.type_as(feat)
    dist_mat0 = euclidean_dist(center0, feat)
    dist_mat1 = euclidean_dist(center1, feat)
    dist_mat2 = euclidean_dist(center2, feat)
    dist_mat3 = euclidean_dist(center3, feat)
    # dist_mat4 = euclidean_dist(center4, feat)
 
    dist_ap = []  # strict triplet 与 近的聚类中心 计算margin=0.5
    dist_an = []
    dist_ap0, dist_an0 = single_pose_sample_mining(dist_mat0, labels)
    dist_ap1, dist_an1 = single_pose_sample_mining(dist_mat1, labels)
    dist_ap2, dist_an2 = single_pose_sample_mining(dist_mat2, labels)
    dist_ap3, dist_an3 = single_pose_sample_mining(dist_mat3, labels)
    # dist_ap4, dist_an4 = single_pose_sample_mining(dist_mat4, labels)
    for i in range(N):
        min_idx = np.argmin([dist_ap0[i], dist_ap1[i], dist_ap2[i], dist_ap3[i]], axis=0)
        # max_idx = np.argmax([dist_ap0[i], dist_ap1[i], dist_ap2[i]], axis=0)
        # 近
        if min_idx == 0:
            dist_ap.append(dist_ap0[i])
            dist_an.append(dist_an0[i])
        elif min_idx == 1:
            dist_ap.append(dist_ap1[i])
            dist_an.append(dist_an1[i])
        elif min_idx == 2:
            dist_ap.append(dist_ap2[i])
            dist_an.append(dist_an2[i])
        elif min_idx == 3:
            dist_ap.append(dist_ap3[i])
            dist_an.append(dist_an3[i])
        # elif min_idx == 4:
        #     dist_ap.append(dist_ap4[i])
        #     dist_an.append(dist_an4[i])

    dist_ap = torch.stack(dist_ap)
    dist_an = torch.stack(dist_an)
    
    # loss_l = loose_loss(dist_ap_l, dist_an_l)
    loss_s = tri_loss(dist_ap, dist_an)
    return loss_s, dist_ap, dist_an