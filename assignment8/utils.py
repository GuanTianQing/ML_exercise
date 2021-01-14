# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_loss(trainloss, testloss):
    plt.figure()
    plt.plot(trainloss, 'b-', testloss, 'r-')
    #plt.xlim(0, n_iter)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('learning curve')
    plt.legend(['training loss', 'testing loss'])
    plt.show()


def parse_movielens(datapath, usercount, moviecount):
    moviepath = os.path.join(datapath, 'movies.csv')
    fh = open(moviepath, 'r')
    lines = fh.readlines()[1:] # 排除标题行
    
    column_movie_dict = {}
    movie_column_dict = {}
    for i in range(moviecount):
        items = lines[i].split(',', maxsplit=1)
        column_movie_dict[i] = [int(items[0]), items[1].strip()]
        movie_column_dict[int(items[0])] = i
    
    ratingpath = os.path.join(datapath, 'ratings.csv')
    ratings_raw = np.genfromtxt(ratingpath, delimiter=',', skip_header=1)
    ratings = np.empty((usercount, moviecount), dtype=np.float64)
    mask = np.zeros_like(ratings)
    for r in ratings_raw:
        ratings[int(r[0]-1), int(movie_column_dict[r[1]])] = r[2]
        mask[int(r[0]-1), int(movie_column_dict[r[1]])] = 1
        
    return column_movie_dict, ratings, mask



def ratings_mean(ratings, mask):
    return np.sum(ratings, axis=0) / (np.sum(mask, axis=0) + np.finfo(np.float32).eps)


def euclidean_rank(feature, query):
    # Euclidean distance
    compares = np.sqrt(np.sum((feature - np.tile(query.reshape((1,-1)), (feature.shape[0], 1)))**2, axis=1))
    # 距离从小到大排序的结果
    rank = np.argsort(compares)
    # 因为feature中含有query，compares和rank的第一个元素与query本身有关，所以排除第一个元素
    return compares[1:], rank[1:]


def cosine_rank(feature, query):
    # Adjusted Cosine Similarity
    mu = feature.mean(axis=0)
    new_feature = feature - mu
    new_query = query - mu
    feature_norm = np.linalg.norm(new_feature, ord=2, axis=1)
    query_norm = np.linalg.norm(new_query, ord=2)
    compares = np.divide(np.matmul(new_feature, query.reshape((-1,1))), feature_norm[:,np.newaxis]) / query_norm        
    # 相似度从大到小排序的结果
    rank = np.argsort(compares[:,0])[::-1]
    return compares[1:], rank[1:]