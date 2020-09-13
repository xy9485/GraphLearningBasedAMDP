import numpy as np
import pandas as pd
import random
from sklearn import preprocessing  # to normalise existing X
import os
import matplotlib.pyplot as plt
import matplotlib
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

list1 = [[0, 1, 5], [3, 5], [4, 0, 1, 5]]
# for i in list1:
#     for a,b in enumerate(i):
#         i[a] = i[a]+1
# print(list1)
# list2 = np.array(list1)
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
# d=pd.Series.rolling(se, window=4, center=False).mean()
if (1,2) in (1,2):
    print("ja")
else:
    print("nein")