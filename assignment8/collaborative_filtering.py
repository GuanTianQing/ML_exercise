# -*- coding: utf-8 -*-
import numpy as np
#import matplotlib.pyplot as plt

class CF(object):
    def __init__(self, n_user, n_item, n_factor, lambd):
        self.n_user = n_user
        self.n_item = n_item
        self.n_factor = n_factor
        self.lambd = lambd
        # 初始化用户偏好矩阵和物品(电影)特征矩阵
        self.U = np.random.normal(0, 0.01, (n_user, n_factor))
        self.I = np.random.normal(0, 0.01, (n_item,n_factor))
        
        self.trainloss = []
        self.testloss = []
        self.snapshot = []
        
        
    def predict(self):
        # 任务1a：根据self.U 和 self.I 计算模型预测的评分
        return np.matmul(self.U ,np.transpose(self.I))
    
    
    def mse(self, R,M):
        # Y is rating matrix
        # W is weight(or mask) matrix
        # 计算预测值和实际值的均方误差
        return np.sum( ((self.predict() - R)*M) **2) / M.sum()
    
    def optimize_user(self,R,U, I, nu, nf, r_lambda):
        iT = np.transpose(I)
        for u in range(nu):
            iT_I = np.matmul(iT,  I)
            lambdE = np.dot(r_lambda, np.identity(nf))
            Ru_I = np.matmul(R[u],I)
            self.U[u] = np.linalg.solve(iT_I + lambdE, Ru_I)

    def optimize_item(self,R,U,I, ni, nf, r_lambda):
        uT = np.transpose(U)
        for i in range(ni):           
            uT_U = np.matmul(uT, U)
            lambdE = np.dot(r_lambda, np.identity(nf))
            uT_Ri = np.matmul(uT, R[:,i])
            self.I[i] = np.linalg.solve(uT_U + lambdE, uT_Ri)
        
    def update(self,R, M):
        # Alternating Least Square
        # u is index; Wu is data
#        for u, Wu in enumerate(W):
#            # 更新self.U的每一行，即每个用户的特征
#            #np.linalg.solve:Solve a linear matrix equation, or system of linear scalar equations.
#            self.U[u] = np.linalg.solve(np.dot(self.I, np.dot(np.diag(Wu), self.I.T)) + self.lambd * np.eye(self.n_factor),\
#                                        np.dot(self.I, np.dot(np.diag(Wu), Y[u])))
#
#        for i, Wi in enumerate(W.T):
#            # 任务1b：根据教学内容和上面对self.U的更新，更新self.I的每一列
#            self.I[:, i] = np.linalg.solve(np.dot(self.U.T, np.dot(np.diag(Wi), self.U)) + self.lambd * np.eye(self.n_factor),\
#                                        np.dot(self.U.T, np.dot(np.diag(Wi), Y[:, i])))
        self.optimize_user(R*M,self.U,self.I,self.n_user,self.n_factor,self.lambd)
        self.optimize_item(R*M,self.U,self.I,self.n_item,self.n_factor,self.lambd)
        
        prediction_error = self.mse(R, M)
        self.trainloss.append(prediction_error)
        self.snapshot.append((self.U.copy(), self.I.copy()))
        print('training error:%.4f' % (prediction_error))#格式化输出浮点数(float)
        
    
    def evaluate(self, R, M):
        prediction_error = self.mse(R, M)
        self.testloss.append(prediction_error)
        print('testing error:%.4f' % (prediction_error))
        
