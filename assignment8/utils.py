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
    """
    axis=0表示按列向量处理，求多个列向量的范数
    axis=1表示按行向量处理，求多个行向量的范数
    a
    Out[14]: 
    array([[1, 2],
           [3, 4]])
    
    np.sum(a, axis=0)
    Out[15]: array([4, 6])
    
    np.sum(a, axis=1)
    Out[16]: array([3, 7])
    """
    return np.sum(ratings, axis=0) / (np.sum(mask, axis=0) + np.finfo(np.float32).eps)


def euclidean_rank(feature, query):
    # Euclidean distance
    compares = np.sqrt(np.sum((feature - np.tile(query.reshape((1,-1)), (feature.shape[0], 1)))**2, axis=1))
    # 距离从小到大排序的结果
    """np.argsort(a, axis=-1, kind='quicksort', order=None)
        It returns an array of indices of the same shape as `a` that index data along the given axis in sorted order
        x = np.array([3, 1, 2])
        np.argsort(x)
        array([1, 2, 0])
    """
    rank = np.argsort(compares)
    # 因为feature中含有query，compares和rank的第一个元素与query本身有关，所以排除第一个元素
    return compares[1:], rank[1:]


def cosine_rank(feature, query):
    # Adjusted Cosine Similarity
    """
    axis=0表示按列向量处理，求多个列向量的范数
    axis=1表示按行向量处理，求多个行向量的范数
    a = np.array([[1, 2], [3, 4]])
    a
    Out[9]: 
    array([[1, 2],
           [3, 4]])
    
    np.mean(a, axis=0)
    Out[10]: array([2., 3.])
    
    np.mean(a, axis=1)
    Out[11]: array([1.5, 3.5])
    """
    mu = feature.mean(axis=0)
   
    new_feature = feature - mu
    """
    b
    Out[26]: 
    array([[1],
           [1]])
    
    a
    Out[27]: 
    array([[1, 2],
           [3, 4]])
    
    a-b
    Out[28]: 
    array([[0, 1],
           [2, 3]])
    """
    new_query = query - mu
    """
    axis=0表示按列向量处理，求多个列向量的范数
    axis=1表示按行向量处理，求多个行向量的范数
    a
    Out[20]: 
    array([[1, 2],
           [3, 4]])
    
    np.linalg.norm(a, ord=2, axis=0)
    Out[21]: array([3.16227766, 4.47213595])
    
    np.linalg.norm(a, ord=2, axis=1)
    Out[22]: array([2.23606798, 5.        ])
      """
    feature_norm = np.linalg.norm(new_feature, ord=2, axis=1)
    query_norm = np.linalg.norm(new_query, ord=2)
    compares = np.divide(np.matmul(new_feature, query.reshape((-1,1))), feature_norm[:,np.newaxis]) / query_norm        
    # 相似度从大到小排序的结果
    rank = np.argsort(compares[:,0])[::-1]
    return compares[1:], rank[1:]