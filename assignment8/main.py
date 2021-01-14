# -*- coding: utf-8 -*-
import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib import cm
import utils
from collaborative_filtering import CF


#%% 载入数据
datapath = 'ml-latest-small'
n_movie = 9125
n_user = 671

column_movie_dict, ratings, mask = \
    utils.parse_movielens(datapath, n_user, n_movie)


#%% mean normalization
ratings_mean = utils.ratings_mean(ratings, mask)
ratings -= ratings_mean# 广播



#%% 划分数据集
# 通过划分 mask 矩阵中值为1的元素实现对 ratings 的划分，用户和电影的数量没有变化
# 注：阅读 np.where 的文档理解其功能和返回值格式
ones = np.where(mask == 1)

np.random.seed(1)
idx = np.random.permutation(len(ones[0]))
# 留出2000个rating作为测试用
train_ones = [ones[0][idx[:-2000]], ones[1][idx[:-2000]]]
test_ones = [ones[0][idx[-2000:]], ones[1][idx[-2000:]]]

train_mask = np.zeros_like(mask)
train_mask[train_ones] = 1
          
test_mask = np.zeros_like(mask)
test_mask[test_ones] = 1



#%% 设定隐含因子数，创建模型
n_factor = 10
lambd = 10  # 规则化项的权重
model = CF(n_user, n_movie, n_factor, lambd)


#%% 训练模型
# 任务1：完成 CF 类中 predict 和 update 方法

n_iter = 10
for i in range(n_iter):
    model.update(ratings, train_mask)
    model.evaluate(ratings, test_mask)

utils.plot_loss(model.trainloss, model.testloss)



#%% 观察推荐系统的工作情况
# 为了不推荐用户已评分的电影，做如下操作
reverse_mask = np.ones_like(mask)
reverse_mask[ones] = 0
all_predictions = (model.predict() + ratings_mean) * reverse_mask
                             

#%% 给定用户，推荐电影。（User - Item）
userid = 0 # 以第一个用户为例
prediction = all_predictions[userid]
recommend_ids = np.argsort(prediction)[::-1]
# 用户可能喜欢的前10部电影
for i in range(10):
    print('{0}#: {1}. Predicted Rating:{2:.2f}'.format(i+1, column_movie_dict[recommend_ids[i]][1], prediction[recommend_ids[i]]))



#%% 找喜好相似的用户，根据这些用户对电影的评分进行推荐。（User - User）

# 尝试不同的“距离”计算方法
_, rank = utils.euclidean_rank(model.U, model.U[userid])
#_, rank = utils.cosine_rank(model.U, model.U[userid])

# 根据前两个喜好相似的用户进行推荐。
# 第一个相似用户
prediction = all_predictions[rank[0]]
recommend_ids = np.argsort(prediction)[::-1]
for i in range(10):
    print('{0}#: {1}. Predicted Rating:{2:.2f}'.format(i+1, column_movie_dict[recommend_ids[i]][1], prediction[recommend_ids[i]]))

print()
# 第二个相似用户
prediction = all_predictions[rank[1]]
recommend_ids = np.argsort(prediction)[::-1]
for i in range(10):
    print('{0}#: {1}. Predicted Rating:{2:.2f}'.format(i+1, column_movie_dict[recommend_ids[i]][1], prediction[recommend_ids[i]]))



#%% 以与用户作出好评的物品相似的物品做为推荐。（Item - Item）
prediction = all_predictions[userid]
preferred_item = np.argmax(prediction)

# 根据评分最高的物品查找其它相似物品
_, rank = utils.euclidean_rank(model.I, model.I[preferred_item])
#_, rank = utils.cosine_rank(model.U, model.U[userid])

for i in range(10):
    print('{0}#: {1}. Predicted Rating:{2:.2f}'.format(i+1, column_movie_dict[rank[i]][1], prediction[rank[i]]))
