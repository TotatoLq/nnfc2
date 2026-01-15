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

    for _ in range(100):
        a, n, c, p = nnfc(data_path)
        print('ari', a, 'nmi', n, 'acc', c)
        ari.append(a)
        nmi.append(n)
        acc.append(c)
    print("------------------------------------------------------------------------------")
    print('ari', max(ari), 'nmi', max(nmi), 'acc', max(acc), p)
    print('ari', get_mean(ari), 'nmi', get_mean(nmi), 'acc', get_mean(acc), p)
    save_evaluating_indicator('max', dataset_name, max(ari), max(nmi), max(acc), save_dir="./acc_records/nnfc")
    save_evaluating_indicator('mean', dataset_name, get_mean(ari), get_mean(nmi), get_mean(acc), save_dir="./acc_records/nnfc")



    # 保存结果。
    with open('result.txt', 'a') as f:
        f.write(data_path)
        f.write(
            'ari' + str(max(ari)) + 'nmi' + str(max(nmi)) + 'acc' + str(max(acc)) + str(p) + '\n'
        )