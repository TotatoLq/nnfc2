from codes.save_evaluate import save_evaluating_indicator
from utils import *
import os
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import random
import numpy
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import time
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import math
dataname=['balancescale','segmentation','ecoli','zoo','thyroid','wine','banknote','sonar','liver','ionosphere'
          ,'leaf','liver','spambase','vehicle','Yeast']
dataname=['movement_libras','compound','Aggregation','jain','Pathbased','R15']
dataname=['wine','breast','ecoli','zoo','thyroid','seeds','abalone'
          ,'gesture','liver','ionosphere','heart','balancescale','vehicle','banknote','sonar' ,'leaf'
          ,'Yeast']
dataname=['iris']
datasets=["citeseer_Louvain","citeseer_Metis","cora_Louvain","cora_Metis","pubmed_Louvain","pubmed_Metis",
          "chameleon_Louvain","chameleon_Metis","computers_Louvain","computers_Metis","photo_Louvain","photo_Metis",
          "squirrel_Louvain","squirrel_Metis","webkb_Louvain","webkb_Metis"]
# datasets=["abalonefed"]

def get_mean(data):
    return numpy.mean(data, axis=0)

for dataset_name in datasets:
    data_path = os.path.join('..', 'datasets','pkl_datasets', f'{dataset_name}.pkl')
    print(data_path)
    ari = []
    nmi = []
    acc = []
    use_gpu = os.getenv("NNFC_USE_GPU", "0") == "1"

    for _ in range(100):
        a, n, c, p = nnfc(data_path, use_gpu=use_gpu)
        print('ari', a, 'nmi', n, 'acc', c)
        ari.append(a)
        nmi.append(n)
        acc.append(c)
