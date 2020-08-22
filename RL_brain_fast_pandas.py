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
        self.q_table = pd.DataFrame(columns=np.arange(self.action_size), dtype=np.float64)
        self.e_table = self.q_table.copy()
        # self.initialize_q_e()
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
    def initialize_empty_q_e(self):
        self.q_table = pd.DataFrame(columns=np.arange(self.action_size), dtype=np.float64)
        self.e_table = self.q_table.copy()
    def initialize_q_e(self):
        for i in range(self.state_size[0]):
            for j in range(self.state_size[1]):
                for k in range(2):
                    for l in range(2):
                        for m in range(2):
                            state = (i, j, k, l, m)
                            self.states.append(str(state))

        self.q_table = pd.DataFrame(np.random.rand(len(self.states), self.action_size), index=self.states,
                                    columns=np.arange(4), dtype=np.float64)
        # self.e_table = pd.DataFrame(np.zeros(len(self.states), self.action_size),index=self.states ,columns=np.arange(4), dtype=np.float64)

    def resetEligibility(self):
        # self.e_table = pd.DataFrame(np.zeros((len(self.states), self.action_size)), index=self.states, columns=np.arange(4), dtype=np.float64)
        for col in self.e_table:
            self.e_table[col].values[:] = 0


    def check_state_exist(self, state_str):  # in case q table is initialized as empty first
        if state_str not in self.q_table.index:
            # append new state to q table
            to_be_append1 = pd.Series(
                [0.0] * self.action_size,
                index=self.e_table.columns,
                name=state_str,
            )
            to_be_append2 = pd.Series(
                np.random.rand(self.action_size),
                index=self.q_table.columns,
                name=state_str,
            )
            self.q_table = self.q_table.append(to_be_append2)
            self.e_table = self.e_table.append(to_be_append1)

    def policy(self, state, actions):
        self.check_state_exist(state)
        ##epsilon greedy policy
        ran = np.random.randint(100)
        if ran < self.epsilon * 100:
            # return actions[random.randint(0, len(actions) - 1)]
            return np.random.choice(actions)
        return actions[np.argmax([self.q_table.loc[state, a] for a in actions])]

    def policyNoRand(self, state, actions):
        ## Pure greedy policy used for displaying visual policy
        return actions[np.argmax([self.q_table.loc[str(state), a] for a in actions])]

    def learn(self, s, a, s_, a_, a_star, reward):
        ## Update the eligible states according to Watkins Q-lambda

        q_predict = self.q_table.loc[s, a]
        if s_ != str(self.env.goal):    # 有问题！！！
            print("s_:",s_,"self.env.goal:",self.env.goal)
            q_target = reward + self.gamma * self.q_table.loc[s_, a_star]
        else:
            q_target = reward

        delta = q_target - q_predict

        self.e_table.loc[s, a] = 1
        self.q_table += self.lr * delta * self.e_table
        if a_ == a_star:
            self.e_table *= (self.gamma * self.lam)
        else:
            self.resetEligibility()

        self.e_table = self.e_table.where(self.e_table > 0.01, 0)
