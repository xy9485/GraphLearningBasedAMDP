"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

"""

import numpy as np
import pandas as pd
import random


class WatkinsQLambda():
    def __init__(self, state_size, action_size, env, epsilon, lr, gamma, lam):
        self.state_size = state_size
        self.action_size = action_size
        self.env = env
        self.epsilon = epsilon
        self.states = []
        # self.q_table = np.append([env.state_str],np.random.rand(4))
        self.q_table1 = np.array([str(env.state)])
        self.q_table2 = np.random.rand(1,4)

        # self.e_table = np.append([env.state_str],np.zeros(4))
        self.e_table1 = np.array([str(env.state)])
        self.e_table2 = np.zeros((1,4))

        # self.initializeQandE()
        self.lr = lr
        self.gamma = gamma
        self.lam = lam

    # def initializeQandE(self):
    #     for i in range(self.state_size[0]):
    #         for j in range(self.state_size[1]):
    #             for k in range(2):
    #                 for l in range(2):
    #                     for m in range(2):
    #                         state_str = (i, j, k, l, m)
    #                         self.states.append(str(state_str))
    #
    #     self.q_table = pd.DataFrame(np.random.rand(len(self.states), self.action_size), index=self.states,columns=np.arange(4), dtype=np.float64)
    #     self.e_table = pd.DataFrame(np.zeros((len(self.states), self.action_size)), index=self.states, columns=np.arange(4), dtype=np.float64)

    def resetEligibility(self):
        # e_table_shape = self.e_table2.shape
        # # print("e_table_shape:", e_table_shape)
        # self.e_table2 = np.zeros(e_table_shape)
        self.e_table2.fill(0)

    def check_state_exist(self, state_str):  # in case q table is initialized as empty first
        if state_str not in self.q_table1:
            # append new state_str to q table
            to_append_q1 = state_str
            to_append_q2 = np.random.rand(4)
            to_append_e1 = state_str
            to_append_e2 = np.zeros(4)

            self.q_table1 = np.append(self.q_table1, to_append_q1)
            self.q_table2 = np.vstack((self.q_table2, to_append_q2))
            self.e_table1 = np.append(self.e_table1, to_append_e1)
            self.e_table2 = np.vstack((self.e_table2, to_append_e2))

    def policy(self, state, actions):
        self.check_state_exist(str(state))
        ##epsilon greedy policy
        ran = np.random.randint(100)
        if ran < self.epsilon * 100:
            # return actions[random.randint(0, len(actions) - 1)]
            return np.random.choice(actions)
        else:
            # index = np.where(self.q_table[:, 0] == state_str)
            # np_state = np.array(state_str)
            # temp_list = []
            # for a in actions:
            #     bool_index = self.q_table1 == str(state)
            #     print("bool_index.shape:",bool_index.shape)
            #     temp_list.append(self.q_table2[bool_index,a])
            # action_index = np.argmax(temp_list)
            # return actions[action_index]
            # print("self.q_table2:",self.q_table2)
            return actions[np.argmax([self.q_table2[self.q_table1 == str(state), a] for a in actions])]

    def policyNoRand(self, state, actions):
        ## Pure greedy policy used for displaying visual policy
        # np_state = np.array(state_str)
        # temp_list = []
        # for a in actions:
        #     bool_index = self.q_table1 == str(state)
        #     print("bool_index.shape:",bool_index.shape)
        #     temp_list.append(self.q_table2[bool_index, a])
        # action_index = np.argmax(temp_list)
        # return actions[action_index]
        return actions[np.argmax([self.q_table2[self.q_table1 == str(state), a] for a in actions])]

    def learn(self, s, a, s_, a_, a_star, reward):

        ## Update the eligible states according to Watkins Q-lambda

        q_predict = self.q_table2[self.q_table1 == str(s), a]
        if (s_[0], s_[1]) != self.env.goal:
            q_target = reward + self.gamma * self.q_table2[self.q_table1 == str(s_), a_star]
        else:
            q_target = reward

        delta = q_target - q_predict
        # print("delta:",delta,"q_max:",self.q_table.max())
        self.e_table2[self.e_table1 == str(s), a] = 1
        self.q_table2 += self.lr * delta * self.e_table2
        if a_ == a_star:
            self.e_table2 *= (self.gamma * self.lam)
            self.e_table2 = np.where(self.e_table2 > 0.01, self.e_table2, 0)
        else:
            # self.e_table.at[s, a] = 0
            self.resetEligibility()
