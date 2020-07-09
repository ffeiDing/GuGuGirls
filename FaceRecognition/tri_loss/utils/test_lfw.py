import numpy as np
from distance import compute_dist
from metric import cmc, mean_ap
gallery_f = np.loadtxt('/home/dff/f/Interpret_FR/test/features_gallery_lfw.txt', dtype=np.float32)
query_f = np.loadtxt('/home/dff/f/Interpret_FR/test/features_query_lfw.txt', dtype=np.float32)
print(gallery_f.shape)
print(query_f.shape)
dist = compute_dist(query_f, gallery_f, 'cosine')
#dist = compute_dist(query_f, gallery_f)
print(dist.shape)
#print(dist)
gallery_id_f = open('/home/dff/f/Interpret_FR/test/labels_gallery_lfw.txt')
lines = gallery_id_f.readlines()
gallery_id = []
for line in lines:
    gallery_id.append(int(line.split()[0]))
print(len(gallery_id))
query_id_f = open('/home/dff/f/Interpret_FR/test/labels_query_lfw.txt')
lines = query_id_f.readlines()
query_id = []
for line in lines:
    query_id.append(int(line.split()[0]))
print(len(query_id))
gallery_cams = [ 0 for i in range(len(gallery_id))]
query_cams = [ 1 for i in range(len(query_id))]
mAP = mean_ap(dist, np.asarray(query_id), np.asarray(gallery_id), 
                    np.asarray(query_cams), np.asarray(gallery_cams))
print(mAP)
cmc_scores = cmc(dist, np.asarray(query_id), np.asarray(gallery_id), 
                    np.asarray(query_cams), np.asarray(gallery_cams), topk=10)
print(cmc_scores)
