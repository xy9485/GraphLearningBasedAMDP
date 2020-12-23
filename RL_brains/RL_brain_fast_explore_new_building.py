"""
This part of code is the Q learning brain, which is action1 brain of the agent.
All decisions are made in here.

"""

import numpy as np
import pandas as pd
import scipy
import random



class WatkinsQLambda():
    def __init__(self, state_size, action_size, env, epsilon, lr, gamma, lam):
        self.state_size = state_size
        self.action_size = action_size
        self.env = env
        self.epsilon = epsilon
        self.states = []
        # self.q_table = np.zeros((self.state_size[0], self.state_size[1], 2, 2, 2, self.action_size))
        self.q_table = np.random.rand(self.state_size[0], self.state_size[1], 2, 2, 2, self.action_size)
        # self.q_table2 = np.zeros((self.state_size[0], self.state_size[1], self.action_size))
        # self.q_table2 = np.random.rand(self.state_size[0], self.state_size[1], self.action_size)
        self.q_table2 = np.random.rand(self.state_size[0], self.state_size[1], 2, 2, 2, self.action_size)

        self.q_init = 1
        self.q2_init = 1
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.resetEligibility()
        # 临时
        self.temp_delta = 0
        self.state_actions = []
        self.state_actions_long_life = np.zeros((self.state_size[0], self.state_size[1], 2, 2, 2, self.action_size))
        # self.state_actions_episodic = np.zeros((self.state_size[0], self.state_size[1],self.action_size))
        self.states_long_life = np.zeros((self.state_size[0], self.state_size[1], 2, 2, 2))
        self.states_episodic = np.zeros((self.state_size[0], self.state_size[1], 2, 2, 2))
        print("Rl brain explore building!")

    # def resetEligibility(self):
    #     # self.e_table = np.zeros((self.state_size[0], self.state_size[1], 2, 2, 2, self.action_size))
    #     self.e_table.fill(0)
    def resetEligibility(self):   ##可替换
        self.e_table = []
        self.e_table2 = []
        # self.state_actions_episodic = np.zeros((self.state_size[0], self.state_size[1], self.action_size))
        self.states_episodic = np.zeros((self.state_size[0], self.state_size[1], 2, 2, 2))

    def resetQ(self):
        if self.q_init:
            self.q_table = np.random.rand(self.state_size[0], self.state_size[1], 2, 2, 2, self.action_size)
        else:
            self.q_table = np.zeros((self.state_size[0], self.state_size[1], 2, 2, 2, self.action_size))

    def check_state_exist(self, state):  # in case q table is initialized as empty first
        if state not in self.q_table.index:
            # append new state1 to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
    def policy(self, state, actions):
        ##epsilon greedy policy
        ran = np.random.randint(100)
        if ran < self.epsilon * 100:
            # return actions[random.randint(0, len(actions) - 1)]
            return np.random.choice(actions)
        if self.q_init:
            return actions[np.argmax([self.q_table[state[0], state[1], state[2], state[3], state[4], a] for a in actions])]
        else:
            q_values = np.array([self.q_table[state[0], state[1], state[2], state[3], state[4], a] for a in actions])
            action = actions[np.random.choice(np.flatnonzero(q_values == q_values.max()))]
            return action


    def policyNoRand(self, state, actions):
        ## Pure greedy policy used for displaying visual policy
        if self.q_init:
            return actions[np.argmax([self.q_table[state[0], state[1], state[2], state[3], state[4], a] for a in actions])]
        else:
            q_values = np.array([self.q_table[state[0], state[1], state[2], state[3], state[4], a] for a in actions])
            action = actions[np.random.choice(np.flatnonzero(q_values == q_values.max()))]
            return action

    # def policy(self, state, actions):
    #     # print("yaya")
    #     ##epsilon greedy policy
    #     ran = np.random.randint(100)
    #     if ran < self.epsilon * 100:
    #         actions_pool = []
    #         for a in actions:
    #             if not (state[0], state[1], a) in self.state_actions:
    #                 actions_pool.append(a)
    #         if len(actions_pool) > 0:
    #             # print("yaya")
    #             action = np.random.choice(actions_pool)
    #             self.state_actions.append((state[0],state[1], action))
    #             return action
    #         # return actions[random.randint(0, len(actions) - 1)]
    #         return np.random.choice(actions)
    #     return actions[np.argmax([self.q_table[state[0], state[1], state[2], state[3], state[4], a] for a in actions])]

    def policy_explore(self, state, actions):
        counts = np.array([self.state_actions_long_life[state[0], state[1], state[2], state[3], state[4], a] for a in actions])
        action = actions[np.random.choice(np.flatnonzero(counts == counts.min()))]
        self.state_actions_long_life[state[0], state[1], state[2], state[3], state[4], action] += 1
        return action


    def policy_explore2(self, state, actions): # softmax choosing actions
        # print("yaya")
        # print(state)
        ##epsilon greedy policy
        ran = np.random.randint(100)
        if ran < self.epsilon * 100:
            action = np.random.choice(actions)
            self.state_actions_long_life[state[0], state[1], state[2], state[3], state[4], action] -= 1
            return action
        # sum = np.sum([np.exp(self.state_actions_long_life[state[0], state[1], a]) for a in actions])
        Probs = scipy.special.softmax([self.state_actions_long_life[state[0], state[1], state[2], state[3], state[4], a] for a in actions])
        # for a in actions:
        #     p = np.exp(self.state_actions_long_life[state[0], state[1], a])/sum
        #     Probs.append(p)
        # print(Probs)
        action = np.random.choice(actions, p=Probs)
        self.state_actions_long_life[state[0], state[1], state[2], state[3], state[4], action] -= 1
        return action

    def policy_explore_rl(self, state, actions):
        ran = np.random.randint(100)
        if ran < self.epsilon * 100:
            action = np.random.choice(actions)
            return action
        if self.q2_init:
            action = actions[np.argmax([self.q_table2[state[0], state[1], state[2], state[3], state[4], a] for a in actions])]
            return action
        else:
            q_values = np.array([self.q_table2[state[0], state[1], state[2], state[3], state[4], a] for a in actions])
            action = actions[np.random.choice(np.flatnonzero(q_values == q_values.max()))]
            return action

    def policyNoRand_explore_rl(self, state, actions):
        if self.q2_init:
            action = actions[np.argmax([self.q_table2[state[0], state[1], state[2], state[3], state[4], a] for a in actions])]
            return action
        else:
            q_values = np.array([self.q_table2[state[0], state[1], state[2], state[3], state[4], a] for a in actions])
            action = actions[np.random.choice(np.flatnonzero(q_values == q_values.max()))]
            return action

    # def learn(self, s, a, s_, a_, a_star, reward):
    #     ## Update the eligible states according to Watkins Q-lambda
    #     # print("self.e_table.dtype:", self.e_table.dtype)        # self.e_table.dtype: float64
    #     # print("self.q_table.dtype:", self.q_table.dtype)        # self.q_table.dtype: float64
    #     # print("self.e_table.max():",self.e_table.max())
    #     # print("self.q_table.max():",self.q_table.max())
    #
    #     q_predict = self.q_table[s[0], s[1], s[2], s[3], s[4], a]
    #
    #     if (s_[0],s_[1]) != self.env.goal:
    #         # print("not hit goal")
    #         q_target = reward + self.gamma * self.q_table[s_[0], s_[1], s_[2], s_[3], s_[4], a_star]
    #     else:
    #         # print("hit goal")
    #         q_target = reward   # 这里和original不一样
    #     # q_target = reward + self.gamma * self.q_table[s_[0], s_[1], s_[2], s_[3], s_[4], a_star]
    #
    #     delta = q_target - q_predict
    #     # print("delta:",delta)
    #     # if abs(delta) > self.temp_delta:
    #     #     self.temp_delta = abs(delta)
    #     #     print("self.temp_delta:",self.temp_delta)
    #     # delta = np.around(delta,decimals=4)
    #     # print("delta,type(delta):",delta,type(delta))
    #     #delta,type(delta): 44.585166377063615 <class 'numpy.float64'>
    #
    #     self.e_table[s[0], s[1], s[2], s[3], s[4], a] = 1
    #     # self.e_table = np.around(self.e_table, decimals=4)
    #     # self.q_table = np.around(self.q_table, decimals=4)
    #     # print("self.q_table.shape, self.e_table.shape:", self.q_table.shape, self.e_table.shape)
    #     #self.q_table.shape, self.e_table.shape: (20, 20, 2, 2, 2, 4) (20, 20, 2, 2, 2, 4)
    #     # td = self.lr * delta * self.e_table
    #     # print("td.shape:",td.shape)
    #     # self.q_table += self.lr * delta * self.e_table
    #     self.q_table += self.lr * delta * self.e_table
    #     if a_ == a_star:
    #         self.e_table *= (self.gamma * self.lam)
    #         self.e_table = np.where(self.e_table > 0.01, self.e_table, 0)
    #     else:
    #         # self.e_table[s[0], s[1], s[2], s[3], s[4], a] = 0  here is a mistake to avoid
    #         self.resetEligibility()



    def learn(self, state1, action1, state2, action2, action_star, reward):   #可替换
        ## Update the eligible states according to Watkins Q-lambda
        # print("self.q_table.max():",self.q_table.max())

        found = False
        for x in range(0, len(self.e_table)):
            if self.e_table[x][0] == state1 and self.e_table[x][1] == action1:
                self.e_table[x] = (self.e_table[x][0], self.e_table[x][1], 1)
                found = True
        if not found:
            self.e_table.append((state1, action1, 1))

        if (state2[0], state2[1]) != self.env.goal:
            maxValue = self.q_table[state2[0], state2[1], state2[2], state2[3], state2[4], action_star]
            delta = reward + (self.gamma * maxValue) - self.q_table[state1[0], state1[1], state1[2], state1[3], state1[4], action1]
            # print("delta:",delta)
        else:
            delta = reward - self.q_table[state1[0], state1[1], state1[2], state1[3], state1[4], action1]

        # maxValue = self.q_table[state2[0], state2[1], state2[2], state2[3], state2[4], action_star]
        # delta = reward + (self.gamma * maxValue) - self.q_table[
        #     state1[0], state1[1], state1[2], state1[3], state1[4], action1]

        newE = []  ## remove eligibility traces that are too low by rebuilding
        for x in range(0, len(self.e_table)):
            s, a, v = self.e_table[x][0], self.e_table[x][1], self.e_table[x][2]
            self.q_table[s[0], s[1], s[2], s[3], s[4], a] = self.q_table[s[0], s[1], s[2], s[3], s[4], a] + self.lr * v * delta
            if action2 == action_star:
                self.e_table[x] = (self.e_table[x][0], self.e_table[x][1], self.e_table[x][2] * self.lam * self.gamma)
            else:
                self.e_table[x] = (self.e_table[x][0], self.e_table[x][1], 0)

            s, a, v = self.e_table[x]
            if v > 0.01:
                newE.append((s, a, v))
        self.e_table = newE

    def learn_explore(self, state1, action1, state2, action2, action_star, reward):   #可替换
        ## Update the eligible states according to Watkins Q-lambda
        # print("self.q_table.max():",self.q_table.max())

        found = False
        for x in range(0, len(self.e_table2)):
            if self.e_table2[x][0] == state1 and self.e_table2[x][1] == action1:
                self.e_table2[x] = (self.e_table2[x][0], self.e_table2[x][1], 1)
                found = True
        if not found:
            self.e_table2.append((state1, action1, 1))

        # if (state2[0], state2[1]) != self.env.goal:
        #     maxValue = self.q_table2[state2[0], state2[1], action_star]
        #     delta = reward + (self.gamma * maxValue) - self.q_table2[state1[0], state1[1], action1]
        #     # print("delta:",delta)
        # else:
        #     delta = reward - self.q_table2[state1[0], state1[1], action1]

        maxValue = self.q_table2[state2[0], state2[1], state2[2], state2[3], state2[4], action_star]
        delta = reward + (self.gamma * maxValue) - self.q_table2[state1[0], state1[1], state1[2], state1[3], state1[4], action1]

        # maxValue = self.q_table[state2[0], state2[1], state2[2], state2[3], state2[4], action_star]
        # delta = reward + (self.gamma * maxValue) - self.q_table[
        #     state1[0], state1[1], state1[2], state1[3], state1[4], action1]

        newE = []  ## remove eligibility traces that are too low by rebuilding
        for x in range(0, len(self.e_table2)):
            s, a, v = self.e_table2[x][0], self.e_table2[x][1], self.e_table2[x][2]
            self.q_table2[s[0], s[1], s[2], s[3], s[4], a] = self.q_table2[s[0], s[1], s[2], s[3], s[4], a] + self.lr * v * delta
            if action2 == action_star:
                self.e_table2[x] = (self.e_table2[x][0], self.e_table2[x][1], self.e_table2[x][2] * self.lam * self.gamma)
            else:
                self.e_table2[x] = (self.e_table2[x][0], self.e_table2[x][1], 0)

            s, a, v = self.e_table2[x]
            if v > 0.01:
                newE.append((s, a, v))
        self.e_table2 = newE

    def learn_explore_sarsa(self, state1, action1, state2, action2, action_star, reward):   #可替换
        maxValue = self.q_table2[state2[0], state2[1], state2[2], state2[3], state2[4], action2]
        delta = reward + (self.gamma * maxValue) - self.q_table2[state1[0], state1[1], state2[2], state2[3], state2[4], action1]
        self.q_table2[state1[0],state1[1],state2[2], state2[3], state2[4], action1] = self.q_table2[state1[0],state1[1]
        ,state2[2], state2[3], state2[4], action1] + self.lr * delta
