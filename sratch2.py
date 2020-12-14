import numpy as np
import pandas as pd
import random
from sklearn import preprocessing  # to normalise existing X
import os
import matplotlib.pyplot as plt
# import matplotlib
import sys
import gym
import matplotlib.pyplot
plt.subplots()
# matplotlib.use('tkagg')
# print(matplotlib.get_backend())
# #

# # print(sorted(list1, key=lambda l: len(l)))
# #
# # sr = pd.Series([10, 25, 3, 11, 24, 6])
# # result = sr.rolling(2).mean()
# # print(type(result))
# #
# # fig, axs = plt.subplots(1,4, figsize=(10,5))
# # print(axs.shape)
# # st = fig.suptitle("curves of each repetition",fontsize=14)
# # fig.subplots_adjust(top=0.85)
# # fig.tight_layout()
# # plt.show()
# m = np.array([[[3,4],
#                [2,3]],
#               [[4,6],agent.epsilon = 0.5
#                [5,7]]
#               ])
# print(m.shape)
# print(np.mean(m,axis=0))

# print(np.power([[1,2,3],[2,4]],2))
# from sympy.core.symbol import symbols
# from sympy.solvers.solveset import nonlinsolve
# from sympy import exp, solve
# import math
# x, y= symbols('x, y', real=True)
# solution=solve([exp(-(0+y)/x)-0.5, exp(-(10000+y)/x)-0.05], [x, y])
# print(solve([-x*5000],[x,y]))
# lr_max = 0.3
# lr_min = 0.01
# x, y= symbols('x, y', real=True)
# solution = solve([exp(-(0+y)/x)-lr_max, exp(-(1000+y)/x)-lr_min], [x, y])
# lr_func_a = solution[x]
# lr_func_b = solution[y]
# print(lr_func_a,lr_func_b,type(lr_func_a))
# se=pd.Series([1,2,3,4,4,5,6,6,7,7,8,89,3,3])
# d1=pd.Series.rolling(se, window=4, center=False).mean()
# plt.plot(np.arange(len(d1)), d1, color='black', alpha=0.3)
# plt.show()
# print(np.random.choice([1,2,3]))
# print(sys.path)

# path = "/Users/yuan/Downloads/maze2.txt"
# embedding = []
# with open(path, "r") as f:
#     content = f.readlines()
#     content = [x.strip() for x in content]
#     for item in content:
#         emd = [x for x in item.split()]
#         embedding.append(emd)
# print(embedding)
import scipy

# list1 = [[0, 1, 5], [3, 5, 6], [4, 0, 1]]
# print(np.cumsum(list1, axis=1))
# print("jiji",str(list1))
# print(np.argwhere(list1==3))
# # print(np.sum(list1[0]))
# list2 = np.array(list1)
# print(list2)
# print(np.argwhere(list2==3))
# list3 = np.array(list2)
# print(list3[1].shape)
#
# yerr = np.linspace(0.05, 0.2, 10)
# print(np.random.choice([0,1,2,3],p=np.array([0,0,0.5,0.5])))
# # print(scipy.special.softmax([1,2,3,4,-4]))
#

# list1 = np.random.randint(100,size=(3,4,3))
# print(list1)
# print(list1.flatten())

import pyglet
import gym
# import gym_puddle # Don't forget this extra line!
from gympuddleworld import gym_puddleworld

# gym.envs.register(
#     id='PuddleWorld-v0',
#     entry_point='gym_puddle.envs:PuddleEnv',
#     max_episode_steps=250,
#
# )
# env = gym.make('PuddleWorld-v0')
# print(env.action_space)
# env.reset()
# env.render()
# actions = [np.zeros(2) for i in range(5)]
# for i in range(4):
#     actions[i][i//2] = 0.05 * (i%2 * 2 - 1)
# print(actions)

# sales = { 'apple': 2, 'orange': 3, 'grapes': 4 }
# for k, i in sales.items():
#     print(k,i)
# import numpy as np
# import logging
# import scipy.spatial
# from sklearn.metrics.pairwise import cosine_similarity
# from scipy import sparse
# from sklearn import metrics
# from sklearn import preprocessing
# from sklearn.metrics.pairwise import euclidean_distances
# test_array = np.random.rand(3,100)
# X_normalized = preprocessing.normalize(test_array)
# euclidean_dist = euclidean_distances(X_normalized)
# print(euclidean_dist)
# squared_euclidean = np.square(euclidean_dist)
# print(squared_euclidean)
# adjusted_cosine_distance = 2 - 2*cosine_similarity(X_normalized)
# print(adjusted_cosine_distance)
i = {'set'}
print(i)
for i in range(4):
    print(i)
for i,j in enumerate(['a','b','c']):
    print(i,j)

o = np.array([['a','b','c'],['b','c','v']])
print(np.argwhere(o=='b'))
