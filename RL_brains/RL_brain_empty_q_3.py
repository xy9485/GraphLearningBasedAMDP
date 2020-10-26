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
        self.q_table1 = [str(env.state)]
        self.q_table2 = [[random.random(),random.random(),random.random(),random.random()]]

        self.e_table = []

        # self.initializeQandE()
        self.lr = lr
        self.gamma = gamma
        self.lam = lam

    def resetEligibility(self):
        self.e_table = []

    def check_state_exist(self, state_str):  # in case q table is initialized as empty first
        if state_str not in self.q_table1:
            # append new state_str to q table
            to_append_q1 = state_str
            to_append_q2 = [random.random(),random.random(),random.random(),random.random()]
            self.q_table1.append(to_append_q1)
            self.q_table2.append(to_append_q2)

    def policy(self, state, actions):
        self.check_state_exist(str(state))
        ##epsilon greedy policy
        ran = np.random.randint(100)
        if ran < self.epsilon * 100:
            # return actions[random.randint(0, len(actions) - 1)]
            return np.random.choice(actions)
        else:
            return actions[np.argmax([self.q_table2[self.q_table1.index(str(state))][a] for a in actions])]

    def policyNoRand(self, state, actions):
        ## Pure greedy policy used for displaying visual policy
        return actions[np.argmax([self.q_table2[self.q_table1.index(str(state))][a] for a in actions])]

    def learn(self, s, a, s_, a_, a_star, reward):

        ## Update the eligible states according to Watkins Q-lambda

        q_predict = self.q_table2[self.q_table1.index(str(s))][a]
        if (s_[0], s_[1]) != self.env.goal:
            q_target = reward + self.gamma * self.q_table2[self.q_table1.index(str(s_))][a_star]
        else:
            q_target = reward

        delta = q_target - q_predict
        # print("delta:",delta,"q_max:",self.q_table.max())

        found = False
        for x in range(0, len(self.e_table)):
            if self.e_table[x][0] == s and self.e_table[x][1] == a:
                self.e_table[x] = (self.e_table[x][0], self.e_table[x][1], 1)
                found = True
        if not found:
            self.e_table.append((s, a, 1))
        # print(found)
        new_e_table = []
        for x in range(0, len(self.e_table)):
            s, a, v = self.e_table[x][0], self.e_table[x][1], self.e_table[x][2]
            self.q_table2[self.q_table1.index(str(s))][a] = self.q_table2[self.q_table1.index(str(s))][a] + self.lr * v * delta
            if a_ == a_star:
                self.e_table[x] = (self.e_table[x][0], self.e_table[x][1], self.e_table[x][2] * self.lam * self.gamma)
            else:
                self.e_table[x] = (self.e_table[x][0], self.e_table[x][1], 0)
            s, a, v = self.e_table[x]
            if v > 0.01:
                new_e_table.append((s, a, v))
        self.e_table = new_e_table