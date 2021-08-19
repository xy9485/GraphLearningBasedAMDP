"""
This part of code is the Q learning brain, which is action1 brain of the agent.
All decisions are made in here.

"""

import numpy as np
import pandas as pd
import scipy
import random


class Brain:
    def __init__(self, env):
        self.env = env
        self.state_size = env.size
        self.action_size = env.num_of_actions


class ExploreStateBrain:
    def __init__(self, env, explore_config: dict):
        self.env = env
        self.state_size = env.size
        self.action_size = env.num_of_actions
        # self.explore_config = explore_config
        self.epsilon = explore_config['epsilon_e']
        self.lr = explore_config['lr']
        self.gamma = explore_config['gamma']
        self.e_mode = explore_config['e_mode']

        if self.e_mode == 'sarsa':      # only support sarsa so far
            # self.q_table2 = np.zeros((self.state_size[0], self.state_size[1], 2, 2, 2, self.action_size))
            self.q_table2 = np.random.rand(self.state_size[0], self.state_size[1], 2, 2, 2, self.action_size)
            self.q2_init = 1
            self.states_long_life = np.zeros((self.state_size[0], self.state_size[1], 2, 2, 2))
            self.states_episodic = np.zeros((self.state_size[0], self.state_size[1], 2, 2, 2))
        elif self.e_mode == 'softmax':
            self.state_actions_long_life = np.zeros((self.state_size[0], self.state_size[1], 2, 2, 2, self.action_size))
            self.state_actions_episodic = np.zeros((self.state_size[0], self.state_size[1], 2, 2, 2, self.action_size))
        else:
            raise Exception("invalid e_mode")

    def reset_episodic_staff(self):
        # self.state_actions_episodic = np.zeros((self.state_size[0], self.state_size[1], self.action_size))
        if self.e_mode == 'sarsa':
            self.states_episodic = np.zeros((self.state_size[0], self.state_size[1], 2, 2, 2))
        elif self.e_mode == 'softmax':
            self.state_actions_episodic = np.zeros((self.state_size[0], self.state_size[1], 2, 2, 2, self.action_size))
        else:
            raise Exception("invalid e_mode")

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

    def learn_explore_sarsa(self, state1, action1, state2, action2, reward):
        max_value = self.q_table2[state2[0], state2[1], state2[2], state2[3], state2[4], action2]

        delta = reward + (self.gamma * max_value) - self.q_table2[state1[0], state1[1], state1[2], state1[3], state1[4], action1]

        self.q_table2[state1[0], state1[1], state1[2], state1[3], state1[4], action1] += self.lr * delta

    def policy_explore_softmax(self, state, actions):
        ##epsilon greedy policy
        ran = np.random.randint(100)
        if ran < self.epsilon * 100:
            action = np.random.choice(actions)
            # self.state_actions_long_life[state[0], state[1], action] -= 1
            return action
        # sum = np.sum([np.exp(self.state_actions_long_life[state[0], state[1], a]) for a in actions])
        probs = scipy.special.softmax([self.state_actions_long_life[state[0], state[1], state[2], state[3], state[4], a]
                                       for a in actions])
        action = np.random.choice(actions, p=probs)
        # self.state_actions_long_life[state[0], state[1], action] -= 1
        return action

class ExploreCoordBrain:
    def __init__(self, env, explore_config: dict):
        self.env = env
        self.state_size = env.size
        self.action_size = env.num_of_actions
        self.explore_config = explore_config
        self.epsilon = explore_config['epsilon_e']
        self.lr = explore_config['lr']
        self.gamma = explore_config['gamma']
        self.e_mode = explore_config['e_mode']

        if self.e_mode == 'sarsa':
            # self.q_table2 = np.zeros((self.state_size[0], self.state_size[1], self.action_size))
            self.q_table2 = np.random.rand(self.state_size[0], self.state_size[1], self.action_size)
            self.q2_init = 1
            self.states_long_life = np.zeros((self.state_size[0], self.state_size[1]))
            self.states_episodic = np.zeros((self.state_size[0], self.state_size[1]))
        elif self.e_mode == 'softmax':
            self.state_actions_long_life = np.zeros((self.state_size[0], self.state_size[1], self.action_size))
            self.state_actions_episodic = np.zeros((self.state_size[0], self.state_size[1], self.action_size))
        else:
            raise Exception("invalid e_mode")

    def reset_episodic_staff(self):
        # self.state_actions_episodic = np.zeros((self.state_size[0], self.state_size[1], self.action_size))
        if self.e_mode == 'sarsa':
            self.states_episodic = np.zeros((self.state_size[0], self.state_size[1]))
        elif self.e_mode == 'softmax':
            self.state_actions_episodic = np.zeros((self.state_size[0], self.state_size[1], self.action_size))
        else:
            raise Exception("invalid e_mode")

    def policy_explore_rl(self, state, actions):
        ran = np.random.randint(100)
        if ran < self.epsilon * 100:
            # classic
            action = np.random.choice(actions)
            # softmax
            # Probs = scipy.special.softmax([self.state_actions_long_life[state[0], state[1], a] for a in actions])
            # action = np.random.choice(actions, p=Probs)
            return action
        elif self.q2_init:
            action = actions[np.argmax([self.q_table2[state[0], state[1], a] for a in actions])]
            return action
        else:
            q_values = np.array([self.q_table2[state[0], state[1], a] for a in actions])
            action = actions[np.random.choice(np.flatnonzero(q_values == q_values.max()))]
            return action

    def policyNoRand_explore_rl(self, state, actions):
        if self.q2_init:
            action = actions[np.argmax([self.q_table2[state[0], state[1], a] for a in actions])]
            return action
        else:
            q_values = np.array([self.q_table2[state[0], state[1], a] for a in actions])
            action = actions[np.random.choice(np.flatnonzero(q_values == q_values.max()))]
            return action

    def learn_explore_sarsa(self, state1, action1, state2, action2, reward):  # 可替换

        maxValue = self.q_table2[state2[0], state2[1], action2]
        delta = reward + (self.gamma * maxValue) - self.q_table2[state1[0], state1[1], action1]
        self.q_table2[state1[0], state1[1], action1] = self.q_table2[state1[0], state1[1], action1] + self.lr * delta

    def policy_explore_softmax(self, state, actions):
        ##epsilon greedy policy
        ran = np.random.randint(100)
        if ran < self.epsilon * 100:
            action = np.random.choice(actions)
            # self.state_actions_long_life[state[0], state[1], action] -= 1
            return action
        # sum = np.sum([np.exp(self.state_actions_long_life[state[0], state[1], a]) for a in actions])
        probs = scipy.special.softmax([self.state_actions_long_life[state[0], state[1], a] for a in actions])
        action = np.random.choice(actions, p=probs)
        # self.state_actions_long_life[state[0], state[1], action] -= 1
        return action

    def policy_explore(self, state, actions):  # deterministic policy to explore
        counts = np.array([self.state_actions_long_life[state[0], state[1], a] for a in actions])
        action = actions[np.random.choice(np.flatnonzero(counts == counts.max()))]
        # self.state_actions_long_life[state[0], state[1], action] -= 1
        return action


class QLambdaBrain:
    def __init__(self, env, ground_learning_config):
        self.env = env
        self.state_size = env.size
        self.action_size = env.num_of_actions
        self.epsilon = None
        self.lr = ground_learning_config['lr']
        self.gamma = ground_learning_config['gamma']
        self.lambda_ = ground_learning_config['lambda']

        self.states = []
        self.states_long_life = np.zeros((self.state_size[0], self.state_size[1], 2, 2, 2))
        self.states_episodic = np.zeros((self.state_size[0], self.state_size[1], 2, 2, 2))
        # self.q_table = np.zeros((self.state_size[0], self.state_size[1], 2, 2, 2, self.action_size))
        self.q_table = np.random.rand(self.state_size[0], self.state_size[1], 2, 2, 2, self.action_size)

        self.q_init = 1
        self.reset_eligibility()

    def reset_eligibility(self):
        self.e_table = []

    def reset_episodic_staff(self):
        self.states_episodic = np.zeros((self.state_size[0], self.state_size[1], 2, 2, 2))


    def policy(self, state, actions):
        ##epsilon greedy policy
        ran = np.random.randint(100)
        if ran < self.epsilon * 100:
            # return actions[random.randint(0, len(actions) - 1)]
            return np.random.choice(actions)
        if self.q_init:
            return actions[
                np.argmax([self.q_table[state[0], state[1], state[2], state[3], state[4], a] for a in actions])]
        else:
            q_values = np.array([self.q_table[state[0], state[1], state[2], state[3], state[4], a] for a in actions])
            action = actions[np.random.choice(np.flatnonzero(q_values == q_values.max()))]
            return action

    def policyNoRand(self, state, actions):
        ## Pure greedy policy used for displaying visual policy
        if self.q_init:
            return actions[
                np.argmax([self.q_table[state[0], state[1], state[2], state[3], state[4], a] for a in actions])]
        else:
            q_values = np.array([self.q_table[state[0], state[1], state[2], state[3], state[4], a] for a in actions])
            action = actions[np.random.choice(np.flatnonzero(q_values == q_values.max()))]
            return action



    def learn(self, state1, action1, state2, action2, action_star, reward):  # 可替换
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
            delta = reward + (self.gamma * maxValue) - self.q_table[
                state1[0], state1[1], state1[2], state1[3], state1[4], action1]
            # print("delta:",delta)
        else:
            delta = reward - self.q_table[state1[0], state1[1], state1[2], state1[3], state1[4], action1]

        # maxValue = self.q_table[state2[0], state2[1], state2[2], state2[3], state2[4], action_star]
        # delta = reward + (self.gamma * maxValue) - self.q_table[
        #     state1[0], state1[1], state1[2], state1[3], state1[4], action1]

        new_e = []  ## remove eligibility traces that are too low by rebuilding
        for x in range(0, len(self.e_table)):
            s, a, v = self.e_table[x][0], self.e_table[x][1], self.e_table[x][2]
            self.q_table[s[0], s[1], s[2], s[3], s[4], a] = self.q_table[s[0], s[1], s[2], s[3], s[4], a] + self.lr * v * delta
            if action2 == action_star:
                self.e_table[x] = (self.e_table[x][0], self.e_table[x][1], self.e_table[x][2] * self.lambda_ * self.gamma)
            else:
                self.e_table[x] = (self.e_table[x][0], self.e_table[x][1], 0)

            s, a, v = self.e_table[x]
            if v > 0.01:
                new_e.append((s, a, v))
        self.e_table = new_e



