import sys
import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy
import random
import copy
import math
import statistics
import random
import pandas as pd
import time
import os
from itertools import chain
from collections import Counter
from os import makedirs

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.figure as figure
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# plt.rcParams['xtick.labelsize'] = 17
# plt.rcParams['ytick.labelsize'] = 17

from mpl_toolkits.axes_grid1 import make_axes_locatable
from pprint import pprint
from envs.maze_env_general_all_approaches import Maze
from abstractions.abstraction_all_approaches import AMDP_General
from RL_brains.RL_brain_all_approaches import QLambdaBrain
from gensim_operations.gensim_operation_all_approaches import GensimOperator_General
# from main_across_all_approaches import PlotMaker

class MDP:
    def __init__(self, env):
        self.env = env
        # self.room_layout = env.room_layout
        self.state_size = env.size
        self.action_size = env.num_of_actions
        self.list_of_states = self.get_list_of_states()     # containing tuples
        self.set_of_states = set(self.list_of_states)
        # print(len(self.list_of_states), len(self.set_of_states))
        self.set_adjacencies()
        self.set_transition_and_reward()


    def get_list_of_states(self):
        list_of_states = []
        for i in range(len(self.env.room_layout)):
            for j in range(len(self.env.room_layout[0])):
                if not self.env.room_layout[i, j] == 'w':
                    list_of_states.append((i, j))
        return list_of_states

    def set_adjacencies(self):
        print("start setting adjacencies")
        adjacencies = []
        for i in range(len(self.env.room_layout)-1):
            # print("next line")
            for j in range(len(self.env.room_layout[0])-1):
                current = (i, j)
                down = (i+1, j)
                right = (i, j+1)
                # print(self.env.room_layout[i, j])
                if not self.env.room_layout[i, j] == "w" and not self.env.room_layout[i+1, j] == "w":
                    if (current, down) not in adjacencies:
                        adjacencies.append((current, down))
                        adjacencies.append((down, current))
                if not self.env.room_layout[i, j] == "w" and not self.env.room_layout[i, j+1] == "w":
                    if (current, right) not in adjacencies:
                        adjacencies.append((current, right))
                        adjacencies.append((right, current))
            j = len(self.env.room_layout[0])-1
            current = (i, j)
            down = (i+1, j)
            if not self.env.room_layout[i, j] == "w" and not self.env.room_layout[i+1, j] == "w":
                if (current, down) not in adjacencies:
                    adjacencies.append((current, down))
                    adjacencies.append((down, current))
        i = len(self.env.room_layout)-1
        for j in range(len(self.env.room_layout[0]) - 1):
            current = (i, j)
            right = (i, j+1)
            if not self.env.room_layout[i, j] == "w" and not self.env.room_layout[i, j+1] == "w":
                if (current, right) not in adjacencies:
                    adjacencies.append((current, right))
                    adjacencies.append((right, current))
        print("len(adjacencies): ", len(adjacencies))
        adjacencies = set(adjacencies)
        adjacency_for_each_state = []
        self.dict_adjacency_for_each_state = {}
        for a in self.list_of_states:
            # adj = [a]
            adj = []
            for b in self.list_of_states:
                if (a, b) in adjacencies:
                    adj.append(b)
            # adjacency_for_each_state.append(adj)
            self.dict_adjacency_for_each_state[str(a)] = adj
        # adjacency_for_each_state.append(["bin", "bin"])
        self.dict_adjacency_for_each_state['bin'] = ['bin']
        print("len(self.dict_adjacency_for_each_state): ", len(self.dict_adjacency_for_each_state))
        # print("adjacency_for_each_state:", adjacency_for_each_state)
        # self.adjacency_for_each_state = adjacency_for_each_state

    def get_actions(self, state):
        actions = self.dict_adjacency_for_each_state[str(state)]
        if state == self.env.goal:
            actions.append("to_goal")
        return actions

    def set_transition_and_reward(self):
        print("start setting transition and reward")
        self.list_of_states.append('bin')
        self.num_states = len(self.list_of_states)
        print("self.num_states:", self.num_states)
        transition = np.zeros(shape=(self.num_states, self.num_states, self.num_states))
        rewards = np.zeros(shape=(self.num_states, self.num_states, self.num_states))
        # rewards = np.full(shape=(self.num_states, self.num_states, self.num_states), fill_value=-1)
        # rewards = np.empty((self.num_states, self.num_states, self.num_states), float).fill(-1)
        print("transition and rewards initialized")
        for i in range(self.num_states):
            # print(self.list_of_states[i])
            if self.list_of_states[i] == self.env.goal:
                transition[-1, i, -1] = 1
                rewards[-1, i, -1] = 1000
                # continue
            adjacency_i = self.get_actions(self.list_of_states[i])
            for j in range(self.num_states):
                if self.list_of_states[j] in adjacency_i:
                    transition[j, i, j] = 1
                    if str(list(self.list_of_states[i])) in self.env.traps:    #self.list_of_states[j] or self.list_of_states[i] different results
                        print("traps----")
                        rewards[j, i, j] = -100     # -130 for v1;


        self.transition = transition
        self.rewards = rewards

    def set_transition_and_reward2(self):
        print("start setting transition and reward")
        self.num_states = len(self.list_of_states)
        print("self.num_states:", self.num_states)
        transition = np.zeros(shape=(self.num_states, self.num_states, self.num_states))
        rewards = np.zeros(shape=(self.num_states, self.num_states, self.num_states))-1
        # rewards = np.full(shape=(self.num_states, self.num_states, self.num_states), fill_value=-1)
        # rewards = np.empty((self.num_states, self.num_states, self.num_states), float).fill(-1)
        print("transition and rewards initialized")
        index_goal = self.list_of_states.index(self.env.goal)
        for i in range(self.num_states):
            # print(self.list_of_states[i])
            if self.list_of_states[i] == self.env.goal:
                transition[i, i, i] = 1
                continue
            adjacency_i = self.get_actions(self.list_of_states[i])
            for j in range(self.num_states):
                if self.list_of_states[j] in adjacency_i:
                    transition[j, i, j] = 1
                    if str(list(self.list_of_states[j])) in self.env.traps:
                        print("traps----")
                        rewards[j, i, j] = -1000
                if self.env.goal in adjacency_i:
                    rewards[index_goal, i, index_goal] = 3000
        self.transition = transition
        self.rewards = rewards

    def solve_mdp(self, synchronous=0, monitor=0):   # streamlined solve_amdp
        print("start solving mdp")
        print('length of self.list_of_abstract_states:', len(self.list_of_states))
        print('self.list_of_abstract_states:', self.list_of_states)
        values = np.zeros(len(self.list_of_states))
        if synchronous:
            values2 = copy.deepcopy(values)
        print("len(values):", len(values))
        delta = 0.2
        theta = 0.1
        print("Value Iteration delta values:")
        while delta > theta:
            delta = 0
            for i in range(0, len(values)):
                v = values[i]
                list_of_values = []
                for a in range(len(values)):
                    if self.transition[a, i, a] != 0:
                        value = self.transition[a, i, a] * (self.rewards[a, i, a] + 0.99 * values[a])
                        list_of_values.append(value)
                if synchronous:
                    values2[i] = max(list_of_values)
                    delta = max(delta, abs(v - values2[i]))
                else:
                    values[i] = max(list_of_values)
                    delta = max(delta, abs(v - values[i]))
            print("delta:", delta)
            if synchronous:
                values = copy.deepcopy(values2)
            if monitor:
                self.plot_current_values(self.env, values)           # plot current values

        print("np.min(values), np.max(values), second_min_values: ", np.min(values), np.max(values), np.partition(values, 1)[1])
        # print(values)
        self.plot_current_values(self.env, values, version=1)
        # values -= np.min(values[:-1])
        # second_smallest = np.partition(values, 1)[1]
        # values -= second_smallest

        self.dict_s_v = dict(zip((str(i) for i in self.list_of_states), values))
        print("self.dict_s_v:")
        pprint(self.dict_s_v)

    def get_value(self, state):
        value = self.dict_s_v[str(state)]
        return value

    def plot_current_values(self, env, values, version=1, plot_text=0, show=1, save=1):
        fig, ax = plt.subplots(figsize=(13, 12))
        my_cmap = copy.copy(plt.cm.get_cmap('hot'))
        values_max = np.amax(values)
        vmax = values_max
        # values_min = np.partition(values, 1)[1]
        values_min = np.min(values[:-1])
        vmin = values_min - 0.1 * (values_max - values_min)
        # vmin = -0.1 * vmax
        print(vmax, vmin)
        my_cmap.set_under('lime')
        my_cmap.set_bad('lime')
        my_cmap.set_over('dodgerblue')
        asp = 'equal'

        dict_ = dict(zip((str(i) for i in self.list_of_states), values))
        # print(dict_)
        plate = []
        plate2 = []
        for i in range(env.size[0]):
            row = []
            row2 = []
            for j in range(env.size[1]):
                current_state = (i, j)
                if current_state in env.valid_states:
                    if current_state == (env.start_state[0], env.start_state[1]):
                        row.append(vmin-1)
                        row2.append(dict_[str(current_state)])
                    elif current_state == env.goal:
                        row.append(vmax+1)
                        row2.append(dict_[str(current_state)])
                    else:
                        row.append(dict_[str(current_state)])
                        row2.append(dict_[str(current_state)])
                elif str([i, j]) in env.walls:
                    row.append(vmin)
                    row2.append('w')
            plate.append(row)
            plate2.append(row2)

        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        # plate = np.array(plate)-vmin+0.001
        # im = ax.imshow(plate, norm = colors.LogNorm(vmin=np.amin(plate)+1, vmax=np.amax(plate)-1), aspect=asp, cmap=my_cmap)

        # print(plate)
        if plot_text:
            for i in range(env.size[0]):
                for j in range(env.size[1]):
                    if (i % 3 == 0) and (j % 3 ==0):
                        current_state = (i, j)
                        if current_state in env.valid_states:
                            text_ = round(plate2[i][j])
                            ax.text(j, i, f"{text_}", horizontalalignment='center', verticalalignment='center',
                                    fontsize=9, fontweight='semibold', color='k')
        # fig.subplots_adjust(right=0.85)
        # cax = fig.add_axes([0.9, 0.23, 0.03, 0.5])
        # fig.colorbar(im, cax=cax)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = fig.colorbar(im, cax=cax)
        cax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        cax.tick_params(axis='both', which='both', labelsize=24)
        cax.yaxis.get_offset_text().set_fontsize(24)
        # fig.colorbar(im, ax=ax, shrink=0.75)
        if show:
            fig.show()
        if save:
            fig.savefig(f"./naive_solved_mdp/{env.maze_name}_big{env.big}_v{version}_equal",
                        dpi=200, bbox_inches='tight', transparent=False, pad_inches=0.1)


class PlotMakerNaive:
    def __init__(self, num_of_repetitions, std_factor, num_approaches):
        self.num_of_repetitions = num_of_repetitions
        self.num_approaches = num_approaches

        self.fig_each_rep, self.axs_each_rep = plt.subplots(num_of_repetitions, 5,
                                                            figsize=(5 * 5, num_of_repetitions * 4))
        self.fig_each_rep.set_tight_layout(True)

        self.fig_mean_performance, self.axs_mean_performance = plt.subplots(1, 2, figsize=(5 * 2, 4 * 1))
        self.fig_mean_performance.set_tight_layout(True)
        self.current_approach_mean_performance = self.num_approaches
        self.std_factor = std_factor
        self.max_steps = 0  # for plotting curve of mean performance: reward against move_count

        self.fig_time_consumption, self.ax_time_consumption = plt.subplots()
        self.fig_time_consumption.set_tight_layout(True)
        self.current_approach_time_consumption = self.num_approaches
        self.highest_bar_height = 0

        pass

    @staticmethod
    def plot_maze(env: Maze, version=1, show=1, save=1):
        fontsize = 20 if env.big == 0 else 4.5
        fontweight = 'semibold'
        cmap = ListedColormap(["black", "lightgrey", "yellow", "green", "red"])
        maze_to_plot = np.where(env.room_layout == 'w', 0, 1)
        # maze_to_plot[env.start_state[0], env.start_state[1]] = 4
        # maze_to_plot[env.goal[0], env.goal[1]] = 3
        # w, h = figure.figaspect(maze_to_plot)
        # print("w, h:", w, h)
        # fig1, ax1 = plt.subplots(figsize=(w, h))
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        # ax1.text(env.start_state[1] + 0.5, env.start_state[0] + 0.55, 'S', ha="center", va="center", color="k", fontsize=fontsize,
        #          fontweight=fontweight)
        # ax1.text(env.goal[1] + 0.5, env.goal[0] + 0.55, 'G', ha="center", va="center", color="k", fontsize=fontsize,
        #          fontweight=fontweight)
        ax1.text(6 + 0.5, 5 + 0.55, 'X', ha="center", va="center", color="r", fontsize=fontsize+9,
                 fontweight=fontweight)
        ax1.text(8 + 0.5, 5 + 0.55, 'X', ha="center", va="center", color="r", fontsize=fontsize+9,
                 fontweight=fontweight)
        ax1.pcolor(maze_to_plot, cmap=cmap, vmin=0, vmax=4, edgecolors='k', linewidth=2)
        ax1.set_aspect('equal')
        ax1.invert_yaxis()
        ax1.axis('off')
        fig1.tight_layout()
        if show:
            fig1.show()
        if save:
            fig1.savefig(f"./img_mazes/{env.maze_name}_big{env.big}_v{version}_XX.png", dpi=200,
                         transparent=False, bbox_inches='tight', pad_inches=0.1)

    @staticmethod
    def plot_maze_trap(env: Maze, version=1, show=1, save=0):
        fontsize = 20 if env.big == 0 else 4.5
        fontweight = 'semibold'
        cmap = ListedColormap(["black", "lightgrey", "yellow", "green", "red", "purple"])
        maze_to_plot = np.where(env.room_layout == 'w', 0, 1)
        maze_to_plot[env.start_state[0], env.start_state[1]] = 4
        maze_to_plot[env.goal[0], env.goal[1]] = 3

        # w, h = figure.figaspect(maze_to_plot)
        # print("w, h:", w, h)
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        # fig, ax1 = plt.subplots()
        ax1.text(env.start_state[1] + 0.5, env.start_state[0] + 0.55, 'S', ha="center", va="center", color="k", fontsize=fontsize,
                 fontweight=fontweight)
        ax1.text(env.goal[1] + 0.5, env.goal[0] + 0.55, 'G', ha="center", va="center", color="k", fontsize=fontsize,
                 fontweight=fontweight)
        for trap in env.traps:
            trap = eval(trap)
            maze_to_plot[trap[0], trap[1]] = 5
            ax1.text(trap[1] + 0.5, trap[0] + 0.55, 'T', ha="center", va="center", color="k", fontsize=fontsize,
                     fontweight=fontweight)
        # for flag in env.flags:
        #     # print(flag)
        #     maze_to_plot[flag[0], flag[1]] = 2
        #     ax1.text(flag[1] + 0.5, flag[0] + 0.55, 'F', ha="center", va="center", color="k", fontsize=fontsize,
        #              fontweight=fontweight)
        # print(maze_to_plot)
        ax1.pcolor(maze_to_plot, cmap=cmap, vmin=0, vmax=5, edgecolors='k', linewidth=2)
        ax1.set_aspect('equal')
        ax1.invert_yaxis()
        ax1.axis('off')
        fig1.tight_layout()
        if show:
            fig1.show()
        if save:
            fig1.savefig(f"./img_mazes/{env.maze_name}_big{env.big}_irregular_traps_v{version}.png", dpi=200,
                         transparent=False, bbox_inches='tight', pad_inches=0.1)

    def plot_each_heatmap(self, agent_e, rep, ax_title, save_path):
        # plot heatmap
        fig, ax = plt.subplots(figsize=(13,12))
        im = ax.imshow(agent_e.states_long_life, aspect='equal', cmap='hot')
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        cax.tick_params(axis='both', labelsize=24)
        cax.yaxis.get_offset_text().set_fontsize(24)
        fig.show()
        fig.savefig(f"{save_path}/visit_count_rep{rep}.png", dpi=200, bbox_inches='tight', transparent=False, pad_inches=0.1)

        # im = self.axs_each_rep[rep, 0].imshow(agent_e.states_long_life, aspect='equal', cmap='hot')
        # self.fig_each_rep.colorbar(im, ax=self.axs_each_rep[rep, 0])
        # self.axs_each_rep[rep, 0].set_title(ax_title)

        # divider = make_axes_locatable(self.axs_each_rep[rep, 0])
        # cax = divider.append_axes("right", size="5%", pad=0.1)
        # self.fig_each_rep.colorbar(im, cax=cax)

        # self.fig_each_rep.show()

    def plot_each_reward_movecount(self, reward_episodes, move_count_episodes, rep, curve_label):
        rolling_window_size = int(len(reward_episodes)/30)
        if curve_label.startswith('t'):
            ls = '-'
        elif curve_label.startswith('u'):
            ls = '--'
        elif curve_label.startswith('g'):
            ls = '-'

        d1 = pd.Series(reward_episodes)
        rolled_d1 = pd.Series.rolling(d1, window=rolling_window_size, center=False).mean()
        self.axs_each_rep[rep, 3].plot(np.arange(len(rolled_d1)), rolled_d1, alpha=1, linestyle=ls, label=curve_label)
        self.axs_each_rep[rep, 3].set_ylabel("reward")
        self.axs_each_rep[rep, 3].set_xlabel("Episode No.")
        self.axs_each_rep[rep, 3].set_title(f"reward curve of rep{rep}")
        self.axs_each_rep[rep, 3].legend(loc=4)
        self.axs_each_rep[rep, 3].grid(True)
        # axs[rep, 1].axvspan(0, num_explore_episodes, facecolor='green', alpha=0.5 / num_of_repetitions)
        # axs[rep, 1].axvspan(num_explore_episodes, second_evolution, facecolor='blue',alpha=0.5/num_of_repetitions)
        self.axs_each_rep[rep, 3].axis([0, None, None, 35000])

        d1 = pd.Series(move_count_episodes)
        rolled_d1 = pd.Series.rolling(d1, window=rolling_window_size, center=False).mean()
        self.axs_each_rep[rep, 4].plot(np.arange(len(rolled_d1)), rolled_d1, alpha=1, linestyle=ls, label=curve_label)
        self.axs_each_rep[rep, 4].set_ylabel("move_count")
        self.axs_each_rep[rep, 4].set_xlabel("Episode No.")
        self.axs_each_rep[rep, 4].set_title(f"move_count curve of rep{rep}")
        self.axs_each_rep[rep, 4].legend(loc=1)
        self.axs_each_rep[rep, 4].grid(True)
        # axs[rep, 2].axvspan(0, num_explore_episodes, facecolor='green', alpha=0.5 / num_of_repetitions)
        # axs[rep, 2].axvspan(num_explore_episodes, second_evolution, facecolor='blue', alpha=0.5/num_of_repetitions)
        self.axs_each_rep[rep, 4].axis([0, None, None, None])

        # self.fig_each_rep.show()

    def plot_each_cluster_layout_and_values(self, env, amdp, amdp_mode, rep, ax_title=None, save_path=None, version=1, text_cluster_and_values=0, show=0, save=0):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        # fig, axs = plt.subplots(1, 2, figsize=(15*2, 12))
        my_cmap1 = "gist_ncar"

        my_cmap = copy.copy(plt.cm.get_cmap('hot'))
        my_cmap.set_under('darkred')
        my_cmap.set_bad('lime')
        my_cmap.set_over('dodgerblue')
        asp = 'equal'

        if amdp_mode == "uniform":
            sqrt_ = np.sqrt((len(amdp.list_of_abstract_states) - 1))
            vmax1 = sqrt_ * 10 + sqrt_
            vmin1 = -vmax1 * 0.16
        elif amdp_mode == 'general':
            vmax1 = len(amdp.list_of_abstract_states)-1
            vmin1 = -vmax1 * 0.16

        values_max = np.amax(amdp.values)
        vmax2 = values_max
        # values_min = np.partition(values, 1)[1]
        values_min = np.min(amdp.values[:-1])
        vmin2 = values_min - 0.1 * (values_max - values_min)
        # vmin = -0.1 * vmax

        dict_as2v = dict(zip((i for i in amdp.list_of_abstract_states), amdp.values))
        # print(dict_)

        plate = []      # for plotting graphic
        plate2 = []
        plate3 = []
        plate4 = []
        for i in range(env.size[0]):
            row1 = []        # for plotting cluster_layout
            row2 = []
            row3 = []       # for plotting values
            row4 = []
            for j in range(env.size[1]):
                current_state = (i, j)
                if current_state in env.valid_states:
                    a_state = amdp.get_abstract_state(current_state)
                    if amdp_mode == 'uniform':
                        a_state_int = int(a_state[1]) * 10 + int(a_state[4])
                    elif amdp_mode == 'general':
                        a_state_int = a_state
                    if current_state == (env.start_state[0], env.start_state[1]):
                        row1.append(a_state_int)
                        row2.append(str(a_state))
                        row3.append(np.nan)
                        row4.append(dict_as2v[a_state])

                    elif current_state == env.goal:
                        row1.append(a_state_int)
                        row2.append(str(a_state))
                        row3.append(vmax2 + 1)
                        row4.append(dict_as2v[a_state])
                    else:
                        row1.append(a_state_int)
                        row2.append(str(a_state))
                        row3.append(dict_as2v[a_state])
                        row4.append(dict_as2v[a_state])

                elif str([i, j]) in env.walls:
                    row1.append(vmin1/2)
                    row2.append('w')
                    row3.append(vmin2)
                    row4.append('w')
            plate.append(row1)
            plate2.append(row2)
            plate3.append(row3)
            plate4.append(row4)

        def contour_rect_slow(plate, ax):
            """Clear version"""

            pad = np.pad(plate, [(1, 1), (1, 1)])  # zero padding

            im0 = np.abs(np.diff(pad, n=1, axis=0))[:, 1:]
            im1 = np.abs(np.diff(pad, n=1, axis=1))[1:, :]

            lines = []

            for ii, jj in np.ndindex(im0.shape):
                if im0[ii, jj] > 0:
                    lines += [([ii - .5, ii - .5], [jj - .5, jj + .5])]
                if im1[ii, jj] > 0:
                    lines += [([ii - .5, ii + .5], [jj - .5, jj - .5])]

            for line in lines:
                ax.plot(line[1], line[0], color='k', alpha=1, linewidth=2)


        fig, ax = plt.subplots(figsize=(12,12))
        im = ax.imshow(plate, vmin=vmin1, vmax=vmax1, aspect=asp, cmap=my_cmap1)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        # using pcolor to plot
        # ax.pcolor(plate, cmap=my_cmap1, vmin=vmin1, vmax=vmax1, edgecolors='k', linewidth=2)
        # ax.set_aspect('equal')
        # ax.invert_yaxis()
        # ax.axis('off')

        # x = np.arange(len(plate))
        # y = np.arange(len(plate[0]))
        # X, Y = np.meshgrid(x, y)
        # im = ax.contourf(plate, levels=17, aspect=asp, cmap=my_cmap1, origin='upper')
        # im = ax.contour(plate, levels=17, colors='k', origin='upper')
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.1)
        # plt.colorbar(im, cax=cax)
        plate2 = np.array(plate2)
        for a_state in amdp.list_of_abstract_states:
            coords = np.argwhere(plate2 == str(a_state))
            if len(coords) > 0:
                mean = np.mean(coords, axis=0)
                v_ = round(dict_as2v[a_state])
                # ax.text(mean[1], mean[0], f"{str(a_state)}", horizontalalignment='center', verticalalignment='center',
                #                           fontsize=24, fontweight='semibold', color='k')
        contour_rect_slow(np.array(plate), ax)
        fig.show()
        fig.savefig(f"{save_path}/cluster_layout_rep{rep}.png", dpi=200, bbox_inches='tight', transparent=False, pad_inches=0.1)
        # self.axs_each_rep[rep, 1].imshow(plate, vmin=vmin1, vmax=vmax1, aspect=asp, cmap=my_cmap1)
        # self.axs_each_rep[rep, 1].set_title(ax_title)

        fig, ax = plt.subplots(figsize=(13, 12))
        im = ax.imshow(plate3, vmin=vmin2, vmax=vmax2, aspect=asp, cmap=my_cmap)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax)
        cax.tick_params(axis='both', labelsize=24)
        # ax.text(env.start_state[1], env.start_state[0], 'S', ha="center", va="center", color="k", fontsize=24,
        #         fontweight='semibold')
        # ax.text(env.goal[1], env.goal[0], 'G', ha="center", va="center", color="k", fontsize=24,
        #         fontweight='semibold')
        # cb.ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        # cax.yaxis.get_offset_text().set_fontsize(24)
        # print(np.around(plate3, 1))
        # plt.imshow(plate3, vmin=vmin2, vmax=vmax2, aspect=asp, cmap=my_cmap)
        plate2 = np.array(plate2)
        for a_state in amdp.list_of_abstract_states:
            coords = np.argwhere(plate2 == str(a_state))
            if len(coords) > 0:
                mean = np.mean(coords, axis=0)
                v_ = round(dict_as2v[a_state],1)
                # ax.text(mean[1], mean[0], f"{str(a_state)}", horizontalalignment='center', verticalalignment='center',
                #          fontsize=24, fontweight='semibold', color='k')
                # ax.text(mean[1], mean[0], f"{str(v_)}", horizontalalignment='center', verticalalignment='center',
                #         fontsize=15, fontweight='semibold', color='k')
        contour_rect_slow(np.array(plate), ax)

        fig.show()
        fig.savefig(f"{save_path}/solved_values_rep{rep}.png", dpi=200, bbox_inches='tight', transparent=False, pad_inches=0.1)
        # im = self.axs_each_rep[rep, 2].imshow(plate3, vmin=vmin2, vmax=vmax2, aspect=asp, cmap=my_cmap)
        # divider = make_axes_locatable(self.axs_each_rep[rep, 2])
        # cax = divider.append_axes("right", size="5%", pad=0.1)
        # self.fig_each_rep.colorbar(im, cax=cax)
        # self.fig_each_rep.colorbar(im, ax=self.axs_each_rep[rep, 2])

        if text_cluster_and_values:
            plate2 = np.array(plate2)
            for a_state in amdp.list_of_abstract_states:
                coords = np.argwhere(plate2 == str(a_state))
                if len(coords) > 0:
                    mean = np.mean(coords, axis=0)
                    v_ = round(dict_as2v[a_state])
                    self.axs_each_rep[rep, 1].text(mean[1], mean[0], f"{str(a_state)}", horizontalalignment='center', verticalalignment='center',
                                              fontsize=12, fontweight='semibold', color='k')
                    self.axs_each_rep[rep, 2].text(mean[1], mean[0], f"{str(a_state)}", horizontalalignment='center', verticalalignment='center',
                                                   fontsize=12, fontweight='semibold', color='k')
                    # self.axs_each_rep[rep, 2].text(mean[1], mean[0], f"{str(v_)}", horizontalalignment='center', verticalalignment='center',
                    #                         fontsize=10, fontweight='semibold', color='k')


        if show:
            self.fig_each_rep.show()
        if save:
            self.fig_each_rep.savefig(f"./naive_each_rep/{env.maze_name}_big{env.big}_v{version}")

    def plot_mean_performance_across_reps(self, reward_episodes_repetitions,
                                          move_count_episodes_repetitions,
                                          curve_label, ax_title=None):
        self.current_approach_mean_performance -= 1
        rolling_window_size = int(len(reward_episodes_repetitions[0])/30)
        if curve_label.startswith('t'):
            ls = '-'
        elif curve_label.startswith('u'):
            ls = '--'
        elif curve_label.startswith('g'):
            ls = '-'

        print("============Reward plotting============")
        mean_by_rep_reward = np.mean(reward_episodes_repetitions, axis=0)
        std_by_rep_reward = np.std(reward_episodes_repetitions, axis=0)
        print("mean_by_rep_reward.shape:", mean_by_rep_reward.shape)
        print("std_by_rep_reward.shape", std_by_rep_reward.shape)
        confidence_interval = std_by_rep_reward * self.std_factor
        # plt.rcParams['agg.path.chunksize'] = 10000
        d = pd.Series(mean_by_rep_reward)
        s = pd.Series(confidence_interval)
        rolled_d = pd.Series.rolling(d, window=rolling_window_size, center=False).mean()
        rolled_s = pd.Series.rolling(s, window=rolling_window_size, center=False).mean()
        self.axs_mean_performance[0].plot(np.arange(len(rolled_d)), rolled_d, linestyle=ls, label=curve_label)
        self.axs_mean_performance[0].fill_between(np.arange(len(rolled_d)), rolled_d - rolled_s, rolled_d + rolled_s, alpha=0.25)
        if self.current_approach_mean_performance == 0:
            self.axs_mean_performance[0].set_ylabel("reward")
            self.axs_mean_performance[0].set_xlabel("Episode No.")
            # self.axs_mean_performance[1].legend(loc=4)
            self.axs_mean_performance[0].grid(True)
            # axs[0].set_title(ax_title)
            # axs[0].axvspan(0, num_explore_episodes, facecolor='green', alpha=0.5)
            # axs[1].axvspan(num_explore_episodes, second_evolution, facecolor='blue', alpha=0.5)
            # axs[1].set(xlim=(0, num_total_episodes))
            self.axs_mean_performance[0].axis([0, None, None, 35000])

        print("============Reward against time steps plotting============")
        mean_by_rep_reward = np.mean(reward_episodes_repetitions, axis=0)
        std_by_rep_reward = np.std(reward_episodes_repetitions, axis=0)
        mean_by_rep_move_count = np.mean(move_count_episodes_repetitions, axis=0)
        cum_mean_by_rep_move_count = np.cumsum(mean_by_rep_move_count)
        # max_mean_by_rep_move_count = np.amax(mean_by_rep_move_count)
        # print("max_mean_by_rep_move_count:",max_mean_by_rep_move_count)
        print("mean_by_rep_reward.shape:", mean_by_rep_reward.shape)
        print("std_by_rep_reward.shape:", std_by_rep_reward.shape)
        print("mean_by_rep_move_count:", cum_mean_by_rep_move_count.shape)
        confidence_interval = std_by_rep_reward * self.std_factor
        # plt.rcParams['agg.path.chunksize'] = 10000
        d = pd.Series(mean_by_rep_reward)
        p = pd.Series(cum_mean_by_rep_move_count)
        s = pd.Series(confidence_interval)
        rolled_d = pd.Series.rolling(d, window=rolling_window_size, center=False).mean()
        rolled_p = pd.Series.rolling(p, window=rolling_window_size, center=False).mean()
        rolled_s = pd.Series.rolling(s, window=rolling_window_size, center=False).mean()
        self.axs_mean_performance[1].plot(rolled_p, rolled_d, linestyle=ls, label=curve_label)
        self.axs_mean_performance[1].fill_between(rolled_p, rolled_d - rolled_s, rolled_d + rolled_s, alpha=0.25)
        if rolled_p.max() > self.max_steps:
            self.max_steps = rolled_p.max()
        if self.current_approach_mean_performance == 0:
            self.axs_mean_performance[1].set_ylabel("reward")
            self.axs_mean_performance[1].set_xlabel("steps")
            self.axs_mean_performance[1].legend(loc=4)
            self.axs_mean_performance[1].grid(True)
            self.axs_mean_performance[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            # axs[1].set_title(f"reward against steps over {'big' if env.big==1 else 'small'} {env.maze_name}")
            # axs[3].axvspan(0, num_explore_episodes, facecolor='green', alpha=0.5)
            # axs[3].axvspan(num_explore_episodes, second_evolution, facecolor='blue', alpha=0.5)
            # axs[1].set(xlim=(0, num_total_episodes))
            self.axs_mean_performance[1].axis([0, self.max_steps * 1.05, None, 35000])

    def plot_mean_time_comparison(self, experiment_time_repetitions, solve_amdp_time_repetitions,
                                  ground_learning_time_repetitions, exploration_time_repetitions=[0],
                                  solve_word2vec_time_repetitions=[0], solve_kmeans_time_repetitions=[0], bar_label=None):
        def autolabel(rects, fontsize):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                self.ax_time_consumption.annotate('{}'.format(height),
                             xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom', fontsize=fontsize, fontweight=400)

        mean_by_rep_experiment_time = np.mean(experiment_time_repetitions)
        mean_by_rep_exploration_time = np.mean(exploration_time_repetitions)
        mean_by_rep_word2vec_time = np.mean(solve_word2vec_time_repetitions)
        mean_by_rep_kmeans_time = np.mean(solve_kmeans_time_repetitions)
        mean_by_rep_amdp_time = np.mean(solve_amdp_time_repetitions)
        mean_by_rep_q_time = np.mean(ground_learning_time_repetitions)
        labels = ['Total', 'Exploration', 'Word2vec', 'K-means', 'AMDP', 'Q-learning']
        data = [mean_by_rep_experiment_time,
                mean_by_rep_exploration_time,
                mean_by_rep_word2vec_time,
                mean_by_rep_kmeans_time,
                mean_by_rep_amdp_time,
                mean_by_rep_q_time]
        data = [round(item, 1) for item in data]
        print("data:", data)
        x = np.arange(len(data))
        if self.num_approaches == 4:
            width = 0.2
            fontsize = 6
            if self.current_approach_time_consumption == 4:
                rects = self.ax_time_consumption.bar(x - width*3/2, data, width, label=bar_label)
            elif self.current_approach_time_consumption == 3:
                rects = self.ax_time_consumption.bar(x - width/2, data, width, label=bar_label)
            elif self.current_approach_time_consumption == 2:
                rects = self.ax_time_consumption.bar(x + width/2, data, width, label=bar_label)
            elif self.current_approach_time_consumption == 1:
                rects = self.ax_time_consumption.bar(x + width*3/2, data, width, label=bar_label)
        elif self.num_approaches == 3:
            width = 0.3
            fontsize = 7
            if self.current_approach_time_consumption == 3:
                rects = self.ax_time_consumption.bar(x - width, data, width, label=bar_label)
            elif self.current_approach_time_consumption == 2:
                rects = self.ax_time_consumption.bar(x, data, width, label=bar_label)
            elif self.current_approach_time_consumption == 1:
                rects = self.ax_time_consumption.bar(x + width, data, width, label=bar_label)
        elif self.num_approaches == 2:
            width = 0.4
            fontsize = 8
            if self.current_approach_time_consumption == 2:
                rects = self.ax_time_consumption.bar(x - width/2, data, width, label=bar_label)
            elif self.current_approach_time_consumption == 1:
                rects = self.ax_time_consumption.bar(x + width / 2, data, width, label=bar_label)
        elif self.num_approaches == 1:
            width = 0.5
            fontsize = 10
            rects = self.ax_time_consumption.bar(x, data, width, label=bar_label)

        autolabel(rects, fontsize)
        if data[0] > self.highest_bar_height:
            self.highest_bar_height = data[0]
        self.current_approach_time_consumption -= 1

        if self.current_approach_time_consumption == 0:
            self.ax_time_consumption.set_xticks(x)
            self.ax_time_consumption.set_xticklabels(labels)
            self.ax_time_consumption.set_ylabel("time taken in sec")
            self.ax_time_consumption.legend(loc=9)
            self.ax_time_consumption.set_ylim(top=self.highest_bar_height * 1.1)

class MazeNaive:
    def __init__(self, maze='basic', big=0):
        print("env oh yeah!!!!!")
        self.maze_name = maze
        self.big = big
        self.num_of_actions = 4
        self.room_layout = self.get_room_layout() # return a nparray
        self.size = (len(self.room_layout), len(self.room_layout[0]))
        print("maze.size:", self.size)
        self.reset()
        self.start_state = self.state
        self.set_valid_states()
        print("len(env.valid_states)", len(self.valid_states))
        self.print_maze_info()


    def print_maze_info(self):
        print("env.name:", self.maze_name)
        print("env.big:", self.big)
        print("env.state:", self.state)

    def get_room_layout(self):

        A = 0
        B = 1
        C = 2
        D = 3
        E = 4
        F = 5
        G = 6
        H = 7
        I = 8
        J = 9
        K = 10
        L = 11
        M = 12
        N = 13
        O = 14
        P = 15
        T = "t"
        W = "w"
        if self.maze_name == 'basic':
            ## "True" layout determined by doorways.
            room_layout = [[C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
                           [C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
                           [C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
                           [C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
                           [W, W, W, W, C, W, W, W, W, W, W, D, D, D, D, W, F, F, F, F, F],
                           [B, B, B, B, B, B, B, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
                           [B, B, B, B, B, B, W, E, E, E, E, D, D, D, D, W, F, F, F, F, F],
                           [B, B, B, B, B, B, W, E, E, E, W, D, D, D, D, W, T, T, T, F, F],
                           [W, A, W, W, W, W, W, E, E, E, W, D, D, D, D, W, T, T, T, F, F],
                           [A, A, A, A, A, A, W, E, E, E, W, D, D, D, D, W, T, T, T, F, F],
                           [A, A, A, A, A, A, W, E, E, E, W, D, D, D, D, W, T, T, T, F, F],
                           [T, T, T, T, A, A, W, W, E, W, W, W, W, W, W, W, F, F, F, F, F],
                           [T, T, T, T, A, A, W, G, G, G, G, G, G, G, G, W, F, F, F, F, F],
                           [T, T, T, T, A, A, W, G, G, G, G, G, G, G, G, W, F, F, F, F, F],
                           [A, A, A, A, A, A, W, G, G, G, G, G, G, G, G, F, F, F, F, F, F],
                           [A, A, A, A, A, A, W, G, G, G, G, G, G, G, G, W, F, F, F, F, F]]
            room_layout = np.array(room_layout)

        elif self.maze_name == 'basic2':
            ## "True" layout determined by doorways.
            room_layout = [[C, C, C, C, C, C, W, D, D, D, T, T, T, T, T, W, F, F, F, F, F],
                           [C, C, C, C, C, C, W, D, D, D, D, T, T, T, T, W, F, F, F, F, F],
                           [C, C, C, C, C, C, W, D, D, D, D, D, T, T, T, W, F, F, F, F, F],
                           [C, C, C, C, C, C, W, D, D, D, D, D, D, T, T, W, F, F, F, F, F],
                           [W, W, W, W, C, W, W, W, W, W, W, D, D, D, T, W, F, F, F, F, F],
                           [B, B, B, B, B, B, B, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
                           [B, B, B, B, B, B, W, E, E, E, E, D, D, D, D, W, F, F, F, F, F],
                           [B, B, B, B, B, B, W, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
                           [W, A, W, W, W, W, W, E, E, E, W, D, D, D, D, W, T, T, T, F, F],
                           [A, A, A, A, A, A, W, E, E, E, W, D, D, D, D, W, T, T, T, F, F],
                           [A, A, A, A, A, A, W, E, E, E, W, D, D, D, D, W, T, T, T, F, F],
                           [A, A, T, T, A, A, W, W, E, W, W, W, W, W, W, W, T, T, T, F, F],
                           [A, T, T, T, T, A, W, G, G, G, G, G, G, G, G, W, T, T, T, F, F],
                           [A, T, T, T, T, A, W, G, G, G, G, G, G, G, G, W, T, T, T, F, F],
                           [A, A, T, T, A, A, W, G, G, G, G, G, G, G, G, F, F, F, F, F, F],
                           [A, A, A, A, A, A, W, G, G, G, G, G, G, G, G, W, F, F, F, F, F]]
            room_layout = np.array(room_layout)

        elif self.maze_name == 'basic3':
            ## "True" layout determined by doorways.
            room_layout = [[C, C, C, C, C, C, W, D, D, D, D, T, T, T, T, T, F, F, F, F, F],
                           [C, C, C, C, C, C, W, D, D, D, D, D, T, T, T, T, F, F, F, F, F],
                           [C, C, C, C, C, C, W, D, D, D, D, D, D, T, T, T, F, F, F, F, F],
                           [C, C, C, C, C, C, W, D, D, D, D, D, D, D, T, T, F, F, F, F, F],
                           [W, W, W, W, C, W, W, W, W, W, W, D, D, D, D, T, F, F, F, F, F],
                           [B, B, B, B, B, B, B, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
                           [B, B, B, B, B, B, W, E, E, E, E, D, D, D, D, W, F, F, F, F, F],
                           [B, B, B, B, B, B, W, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
                           [W, A, W, W, W, W, W, E, E, E, W, D, D, D, D, W, T, T, T, F, F],
                           [A, A, A, A, A, A, W, E, E, E, W, D, D, D, D, W, T, T, T, F, F],
                           [A, A, A, A, A, A, W, E, E, E, W, D, D, D, D, W, T, T, T, F, F],
                           [A, A, T, T, A, A, W, W, E, W, W, W, W, W, W, W, T, T, T, F, F],
                           [A, T, T, T, T, A, W, G, G, G, G, G, G, G, G, W, T, T, T, F, F],
                           [A, T, T, T, T, A, W, G, G, G, G, G, G, G, G, W, T, T, T, F, F],
                           [A, A, T, T, A, A, W, G, G, G, G, G, G, G, G, F, F, F, F, F, F],
                           [A, A, A, A, A, A, W, G, G, G, G, G, G, G, G, W, F, F, F, F, F]]
            room_layout = np.array(room_layout)

        elif self.maze_name == 'simple':
            room_layout = [[A, A, A, W, H, H, H, I, I, I, I, I, I, I, W, J, T, T, T, T],  ## simple
                           [A, A, A, W, H, H, H, W, I, I, I, I, I, I, J, J, T, T, T, T],
                           [A, A, A, W, H, H, H, W, I, I, I, I, I, I, W, J, J, J, J, J],
                           [A, A, A, W, H, W, W, W, W, W, W, W, W, W, W, J, J, J, J, J],
                           [A, A, A, W, G, G, G, G, G, G, G, W, P, P, W, J, J, J, J, J],
                           [A, A, A, W, G, G, G, G, G, G, G, G, P, P, W, W, W, W, B, B],
                           [A, A, A, W, G, G, G, G, G, G, G, G, P, P, P, T, T, W, B, B],
                           [A, A, A, W, W, W, W, G, G, G, G, W, P, P, P, T, T, W, B, B],
                           [A, A, A, A, A, A, W, W, W, W, W, W, P, P, P, T, T, B, B, B],
                           [A, A, A, A, A, A, F, F, F, F, F, F, P, P, P, T, T, B, B, B],
                           [A, A, A, A, A, A, W, F, F, F, F, F, P, P, P, T, T, B, B, B]]
            room_layout = np.array(room_layout)

        elif self.maze_name == 'simple2':
            room_layout = [[H, H, H, I, I, I, I, I, I, I, W, J, J, J, J, J],  ## simple2
                           [H, H, H, W, I, I, I, I, I, I, J, J, J, J, J, J],
                           # [H, H, H, W, I, I, I, I, I, I, W, T, T, T, J, J],
                           # [H, W, W, W, W, W, W, W, W, W, W, T, T, T, J, J],
                           [H, H, H, W, I, I, I, I, I, I, W, J, J, J, J, J],
                           [H, W, W, W, W, W, W, W, W, W, W, J, J, J, J, J],
                           [G, G, G, G, G, G, G, W, P, P, W, J, J, J, J, J],
                           [G, G, G, G, G, G, G, G, P, P, W, J, J, J, J, J],
                           [G, G, G, G, G, G, G, G, P, P, W, W, W, W, J, J],
                           [W, W, W, G, G, G, G, W, P, P, T, T, B, W, J, J],
                           [A, A, W, W, W, W, W, W, P, P, T, T, B, B, J, J],
                           [A, A, F, F, F, F, F, F, P, P, T, T, B, B, J, J],
                           [A, A, W, F, F, F, F, F, P, P, T, T, B, B, J, J]]
            room_layout = np.array(room_layout)
        
        elif self.maze_name == 'simple3':
            room_layout = [[P, P, P, J, J, J, J, J],
                           [P, P, W, J, J, J, J, J],
                           [P, P, W, J, J, J, J, J],
                           [P, P, W, W, W, W, J, J],
                           [P, P, P, P, P, W, J, J],
                           [P, P, P, P, P, W, J, J]]
            room_layout = np.array(room_layout)

        elif self.maze_name == 'simple3_ala':
            room_layout = [[P, P, P, J, J, J, J, J, J, J],
                           [P, P, W, J, J, J, J, J, J, J],
                           [P, P, W, J, J, J, J, J, J, J],
                           [P, P, W, W, W, W, W, W, J, J],
                           [P, P, P, P, P, P, P, W, J, J],
                           [P, P, P, P, P, P, P, W, J, J]]
            room_layout = np.array(room_layout)

        elif self.maze_name == 'simple4':
            room_layout = [[H, H, H, I, I, I, I, I, I, W, J, J, J, J, J],  ## simple2
                           [H, H, H, W, I, I, I, I, I, J, J, J, J, J, J],
                           # [H, H, H, W, I, I, I, I, I, W, T, T, T, J, J],
                           # [H, W, W, W, W, W, W, W, W, W, T, T, T, J, J],
                           [H, H, H, W, I, I, I, I, I, W, J, J, J, J, J],
                           [H, W, W, W, W, W, W, W, W, W, J, J, J, J, J],
                           [G, G, G, G, G, G, W, P, P, W, J, J, J, J, J],
                           [G, G, G, G, G, G, G, P, P, W, J, J, J, J, J],
                           [G, G, G, G, G, G, G, P, P, W, W, W, W, J, J],
                           [W, W, W, G, G, G, W, P, P, P, P, B, W, J, J],
                           [A, A, W, W, W, W, W, P, P, P, P, B, W, J, J],
                           [A, A, F, F, F, F, F, P, P, P, P, B, W, J, J],
                           [A, A, W, F, F, F, F, P, P, P, P, B, W, J, J]]
            room_layout = np.array(room_layout)

        elif self.maze_name == 'strips':
            room_layout = [[A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],  ## Strips
                          [A, A, W, B, B, B, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, B, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, F, G],
                          [A, A, W, B, B, W, C, C, C, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, C, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, A, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, E, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, E, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, D, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G]]
            room_layout = np.array(room_layout)

        elif self.maze_name == 'strips2':
            room_layout = [[A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],  ## Strips
                          [A, A, W, B, B, B, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, B, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, F, G, G],
                          [A, A, W, B, B, W, C, C, C, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, C, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, A, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, E, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, E, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, D, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G]]
            room_layout = np.array(room_layout)

        elif self.maze_name == 'spiral':
            room_layout = [[C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, B, B],  ## Spiral
                          [C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, B, B],
                          [D, D, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, B, B],
                          [D, D, W, G, G, G, G, G, G, G, G, G, G, G, G, F, F, W, B, B],
                          [D, D, W, G, G, G, G, G, G, G, G, G, G, G, G, F, F, W, B, B],
                          [D, D, W, H, H, W, W, W, W, W, W, W, W, W, W, F, F, W, B, B],
                          [D, D, W, H, H, W, K, K, K, K, K, K, J, J, W, F, F, W, B, B],
                          [D, D, W, H, H, W, K, K, K, K, K, K, J, J, W, F, F, W, B, B],
                          [D, D, W, H, H, W, L, L, W, W, W, W, J, J, W, F, F, W, B, B],
                          [D, D, W, H, H, W, L, L, M, M, M, W, J, J, W, F, F, W, B, B],
                          [D, D, W, H, H, W, L, L, M, M, M, W, J, J, W, F, F, W, B, B],
                          [D, D, W, H, H, W, W, W, W, W, W, W, J, J, W, F, F, W, B, B],
                          [D, D, W, H, H, I, I, I, I, I, I, I, I, I, W, F, F, W, B, B],
                          [D, D, W, H, H, I, I, I, I, I, I, I, I, I, W, F, F, W, B, B],
                          [D, D, W, W, W, W, W, W, W, W, W, W, W, W, W, F, F, W, B, B],
                          [D, D, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W, B, B],
                          [D, D, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W, B, B],
                          [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, B, B],
                          [A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
                          [A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A]]
            room_layout = np.array(room_layout)

        elif self.maze_name == 'open_space':
            room_layout = [[C, C, C, C, C, C, C, C, C, C, C, C, C, E, E, E, E, W, W, W],  ## Open Space
                          [W, W, W, W, W, C, C, C, C, C, C, C, C, E, E, W, E, E, W, W],
                          [W, W, W, W, W, C, C, C, C, W, W, W, D, E, E, E, W, E, E, W],
                          [W, W, W, W, W, C, C, D, D, D, D, D, D, E, E, E, W, W, E, E],
                          [B, B, B, B, B, B, D, D, D, D, D, D, D, E, E, E, E, W, W, E],
                          [B, B, B, B, B, B, D, D, D, D, D, D, D, E, E, E, E, E, E, E],
                          [B, B, B, B, B, B, D, D, D, W, D, D, D, E, E, E, E, E, E, E],
                          [B, B, B, B, B, B, D, D, W, W, W, D, D, E, E, E, E, E, E, E],
                          [B, B, B, B, B, B, D, W, W, W, W, W, F, F, F, F, F, F, W, W],
                          [W, W, W, B, B, B, H, H, W, W, W, F, F, F, F, F, F, W, W, W],
                          [W, W, W, B, B, B, H, H, H, W, H, F, F, F, F, F, F, F, W, W],
                          [A, A, A, B, B, B, H, H, H, H, H, H, W, W, G, G, G, G, G, G],
                          [A, A, A, B, B, B, H, H, H, H, H, H, W, W, G, G, G, G, G, G],
                          [A, A, A, B, B, B, H, H, H, H, H, G, G, G, G, G, G, G, G, G],
                          [A, A, A, B, B, B, H, H, H, H, H, G, G, G, G, G, G, G, G, G],
                          [A, A, A, W, W, W, W, H, H, H, W, G, G, G, G, G, G, G, G, G],
                          [A, A, A, W, W, W, W, H, H, H, W, G, G, G, G, W, W, G, G, G],
                          [A, A, A, W, W, W, W, H, H, H, W, G, G, G, G, W, W, G, G, G],
                          [A, A, A, W, W, W, W, H, H, H, W, G, G, G, G, W, W, G, G, G],
                          [A, A, A, A, A, A, A, H, H, H, G, G, G, G, G, W, W, G, G, G]]
            room_layout = np.array(room_layout)

        elif self.maze_name == 'high_connectivity':
            room_layout = [[A, A, A, W, H, H, H, I, I, I, I, I, I, I, W, J, J, J, J, J],  ##High Connectivity
                          [A, A, A, W, H, H, H, W, I, I, I, I, I, I, J, J, J, J, J, J],
                          [A, A, A, H, H, H, H, W, I, I, I, I, I, I, W, J, J, J, J, J],
                          [A, A, A, W, W, W, H, W, W, W, I, W, P, W, W, J, J, J, J, J],
                          [A, A, A, W, G, G, G, G, G, W, K, W, P, P, W, J, J, J, J, J],
                          [A, A, A, W, G, G, G, G, G, K, K, W, P, P, W, W, W, J, J, J],
                          [A, A, A, G, G, G, G, G, G, W, K, W, P, P, P, P, W, P, W, W],
                          [A, A, A, W, W, W, W, G, G, W, K, W, P, P, P, P, P, P, P, P],
                          [A, A, A, A, A, A, W, F, W, W, W, W, P, P, P, P, P, P, P, P],
                          [A, A, A, A, A, A, F, F, F, F, F, W, P, P, P, P, P, P, P, P],
                          [A, A, A, A, A, A, W, F, F, F, F, P, P, P, P, P, P, P, P, P],
                          [W, W, W, B, W, W, W, F, W, W, W, W, P, W, W, W, W, P, W, W],
                          [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
                          [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, O, N, N, N],
                          [B, B, B, B, E, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
                          [B, B, B, B, E, W, E, E, E, W, W, O, W, W, O, O, W, N, N, N],
                          [W, C, W, W, E, W, W, W, E, W, L, L, L, W, O, W, W, W, N, W],
                          [C, C, C, W, W, W, D, D, D, W, L, L, L, W, W, M, M, M, M, M],
                          [C, C, C, C, D, D, D, D, D, W, L, L, L, L, M, M, M, M, M, M],
                          [C, C, C, W, W, W, D, D, D, L, L, L, L, L, W, M, M, M, M, M]]
            room_layout = np.array(room_layout)

        elif self.maze_name == 'low_connectivity':
            room_layout = [[A, A, A, W, H, H, H, I, I, I, I, I, I, I, W, J, J, J, J, J],  ##Low Connectivity
                          [A, A, A, W, H, H, H, W, I, I, I, I, I, I, J, J, J, J, J, J],
                          [A, A, A, W, H, H, H, W, I, I, I, I, I, I, W, J, J, J, J, J],
                          [A, A, A, W, W, W, H, W, W, W, I, W, W, W, W, J, J, J, J, J],
                          [A, A, A, W, G, G, G, G, G, W, K, W, P, P, W, J, J, J, J, J],
                          [A, A, A, W, G, G, G, G, G, W, K, W, P, P, W, W, W, J, J, J],
                          [A, A, A, W, G, G, G, G, G, W, K, W, P, P, P, P, W, T, T, T],
                          [A, A, A, W, W, W, W, G, G, W, K, W, P, P, P, P, P, P, P, P],
                          [A, A, A, A, A, A, W, F, W, W, W, W, P, P, P, P, P, P, P, P],
                          [A, A, A, A, A, A, W, F, F, F, F, W, P, P, P, P, P, P, P, P],
                          [A, A, A, A, A, A, W, F, F, F, F, P, P, P, P, P, P, P, P, P],
                          [W, W, W, B, W, W, W, T, T, T, W, W, P, W, W, W, W, W, W, W],
                          [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
                          [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, O, N, N, N],
                          [B, B, B, B, E, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
                          [B, B, B, B, E, W, E, E, E, W, W, W, W, W, O, O, W, N, N, N],
                          [W, C, W, W, E, W, W, W, E, W, L, L, L, W, O, W, W, W, N, W],
                          [C, C, C, W, W, W, D, D, D, W, L, L, L, W, W, M, M, M, M, M],
                          [C, C, C, W, D, D, D, D, D, W, L, L, L, L, M, M, M, M, M, M],
                          [C, C, C, W, W, W, D, D, D, L, L, L, L, L, W, M, M, M, M, M]]
            room_layout = np.array(room_layout)

        elif self.maze_name == 'low_connectivity2':
            room_layout = [[A, A, A, W, H, H, H, I, I, I, I, I, I, I, W, J, J, J, J, J],  ##Low Connectivity
                          [A, A, A, W, H, H, H, W, I, I, I, I, I, I, J, J, J, J, J, J],
                          [A, A, A, W, H, H, H, W, I, I, I, I, I, I, W, J, J, J, J, J],
                          [A, A, A, W, W, W, H, W, W, W, I, W, W, W, W, J, J, J, J, J],
                          [A, A, A, W, G, G, G, G, G, W, K, W, P, P, W, J, J, J, J, J],
                          [A, A, A, W, G, G, G, G, G, W, K, W, P, P, W, W, W, J, J, J],
                          [A, A, A, W, G, G, G, G, G, W, K, W, P, P, P, P, W, W, W, W],
                          [A, A, A, W, W, W, W, G, G, W, K, W, P, P, P, P, P, P, P, P],
                          [A, A, A, A, A, A, W, F, W, W, W, W, P, P, P, P, P, P, P, P],
                          [A, A, A, A, A, A, W, F, F, F, F, W, P, P, P, P, P, P, P, P],
                          [A, A, A, A, A, A, W, F, F, F, F, P, P, P, P, P, P, P, P, P],
                          [W, W, W, B, W, W, W, W, W, W, W, W, P, W, W, W, W, W, W, W],
                          [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
                          [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, O, N, N, N],
                          [B, B, B, B, E, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
                          [B, B, B, B, E, W, E, E, E, W, W, W, W, W, O, O, W, N, N, N],
                          [W, C, W, W, E, W, W, W, E, W, L, L, L, W, O, W, W, W, N, W],
                          [C, C, C, W, W, W, D, D, D, W, L, L, L, W, W, M, M, M, M, M],
                          [C, C, C, W, D, D, D, D, D, W, L, L, L, L, M, M, M, M, M, M],
                          [C, C, C, W, W, W, D, D, D, L, L, L, L, L, W, M, M, M, M, M]]
            room_layout = np.array(room_layout)

        elif self.maze_name.startswith('external'):
            # path = f"/Users/yuan/PycharmProjects/Masterthesis/external_mazes/{self.maze_name}.txt"
            path = f"/home/xue/projects/masterthesis/external_mazes/{self.maze_name}.txt"
            # path = f"external_mazes/{self.maze_name}.txt"
            room_layout = []
            with open(path, "r") as f:
                content = f.readlines()
                content = [x.strip() for x in content]
                for item in content:
                    item = item.replace('#', 'w')
                    # item = item.replace('.', '0')
                    row = [x for x in item]
                    room_layout.append(row)
            room_layout = np.array(room_layout)
            # self.walls = np.argwhere(roomLayout == 'w').tolist()

        else:
            raise Exception("invalide maze name")

        if self.big:
            newLayout = []
            for l in room_layout:
                nextLine = []
                for x in l:
                    nextLine.extend([x, x, x])
                newLayout.append(nextLine)
                newLayout.append(nextLine)
                newLayout.append(nextLine)
            room_layout = np.array(newLayout)
            walls = np.argwhere(room_layout == 'w').tolist()
            self.walls = {str(i) for i in walls}
            traps = np.argwhere(room_layout == 't').tolist()
            self.traps = {str(i) for i in traps}
            return room_layout
        else:
            walls = np.argwhere(room_layout == 'w').tolist()
            self.walls = {str(i) for i in walls}
            traps = np.argwhere(room_layout == 't').tolist()
            self.traps = {str(i) for i in traps}
            print("self.traps:",self.traps)
            return room_layout


    def isTerminal(self, state):
        return state == self.goal  # self.goal is a tuple

    def reset(self):
        self.flags_found_order = []
        self.flags_collected = 0
        self.flags_collected2 = [0, 0, 0]
        if self.maze_name == 'basic':
            # self.state = (6, 4)          #v1
            # self.goal = (14, 1)

            # self.state = (6, 1)        # v2
            # self.goal = (14, 1)

            # self.state = (6, 1)        # v3
            # self.goal = (14, 1)

            # self.state = (6, 1)        # v4
            # self.goal = (15, 18)

            self.state = (15, 0)    # v5
            self.goal = (1, 16)
        elif self.maze_name == 'basic2':
            self.state = (15, 3)
            self.goal = (1, 16)
        elif self.maze_name == 'basic3':
            self.state = (15, 3)
            self.goal = (1, 16)
        elif self.maze_name == 'simple':
            self.state = (1, 1)
            self.goal = (9, 18)
        elif self.maze_name == 'simple2':
            self.state = (9, 0)
            self.goal = (9, 14)
        elif self.maze_name == 'simple3':
            self.state = (5, 4)
            self.goal = (5, 7)
        elif self.maze_name == 'simple3_ala':
            self.state = (5, 4)
            self.goal = (5, 7)
        elif self.maze_name == 'simple4':
            self.state = (9, 0)
            self.goal = (9, 14)
        elif self.maze_name == 'strips':
            self.state = (0, 0)
            self.flags = [(15, 11), (19, 0), (4, 19)]
            self.goal = (18, 1)
        elif self.maze_name == 'strips2':
            self.state = (0, 0)
            self.flags = [(18, 3), (15, 11), (19, 19)]
            self.goal = (18, 1)
        elif self.maze_name == 'spiral':
            # self.state = (19, 0)       #v1
            # self.flags = [(0, 19), (15, 6), (6, 6)]
            # self.goal = (13, 13)

            # self.state = (19, 0)       #v2
            # self.flags = [(0, 19), (15, 6), (10, 10)]
            # self.goal = (0, 0)

            self.state = (19, 0)         # v3
            self.flags = [(3, 3), (16, 16), (10, 10)]
            self.goal = (16, 18)
        elif self.maze_name == 'open_space':
            self.state = (19, 0)
            self.flags = [(0, 0), (2, 17), (13, 14)]
            self.goal = (19, 3)
            # self.goal = (5, 3)   # 纯属试一试
        elif self.maze_name == 'high_connectivity':
            self.state = (19, 0)
            self.flags = [(0, 1), (2, 18), (5, 6)]
            self.goal = (15, 0)
        elif self.maze_name == 'low_connectivity':
            self.state = (19, 0)
            self.flags = [(0, 1), (2, 18), (5, 6)]
            # self.goal = (15, 0)
            self.goal = (4, 19)
        elif self.maze_name == 'low_connectivity2':
            self.state = (19, 0)
            self.flags = [(0, 1), (2, 18), (5, 6)]
            # self.goal = (15, 0)
            self.goal = (4,19)

        elif self.maze_name == "external_maze21x21_1":
            self.state = (1, 1)
            self.goal = (1, 19)
            # self.goal = (19, 19)
            self.flags = [(10,1), (1,17), (17,3)]
        elif self.maze_name == "external_maze21x21_2":
            self.state = (0, 0)
            self.goal = (1, 19)
            # self.goal = (19, 19)
            self.flags = [(10, 1), (1, 17), (17, 3)]
        elif self.maze_name == "external_maze21x21_3":
            self.state = (0, 0)
            self.goal = (20, 20)
            # self.goal = (19, 19)
            self.flags = [(9, 1), (1, 17), (15, 11)]
        elif self.maze_name == "external_maze31x31_1":
            self.state = (0, 0)
            # self.goal = (1, 20)
            self.goal = (29, 29)
            self.flags = [(15, 2), (7, 27), (29, 15)]
        elif self.maze_name == "external_maze31x31_2":
            self.state = (0, 0)
            # self.goal = (1, 20)
            self.goal = (5, 20)
            self.flags = [(29, 2), (15, 16), (5, 23)]
        elif self.maze_name == "external_maze61x61_1":
            self.state = (0, 0)
            self.goal = (1, 57)
            self.flags = [(9, 1), (45, 58), (59, 25)]

        elif self.maze_name == "external_low_connectivity_1":
            self.state = (0, 0)
            self.goal = (1, 18)
            self.flags = [(10, 16), (12, 16), (15, 15)]
        else:
            print("no matched maze")
        if self.big:
            self.state = tuple([i*3 for i in self.state])
            self.goal = tuple([i*3 for i in self.goal])
            # flags_big = []
            # for item in self.flags:
            #     item_prime = tuple([i*3 for i in item])
            #     flags_big.append(item_prime)
            # self.flags = flags_big

    def set_valid_states(self):
        valid_states = []
        template = self.room_layout
        templateX = len(template[0])  # num of columns
        templateY = len(template)  # num of rows
        for i in range(templateY):
            for j in range(templateX):
                if template[i, j] != "w":
                    current_coord = (i, j)
                    valid_states.append(current_coord)

        self.valid_states = set(valid_states)

    def isMovable(self, state):
        # check if wall is in the way or already out of the bounds
        if state[0] < 0 or state[0] > (len(self.room_layout) - 1):
            return False
        elif state[1] < 0 or state[1] > (len(self.room_layout[0]) - 1):
            return False
        elif str([state[0], state[1]]) in self.walls:
            return False
        else:
            return True

    def actions(self, state):
        actions = []
        for a in range(4):
            if self.isMovable(self.step(state, a)):
                actions.append(a)
        return actions

    def step(self, state, action):
        if action == 0:  # right
            new_state = (state[0], state[1] + 1)
        elif action == 1:  # down
            new_state = (state[0] + 1, state[1])
        elif action == 2:  # left
            new_state = (state[0], state[1] - 1)
        elif action == 3:  # up
            new_state = (state[0] - 1, state[1])
        return new_state

    def reward(self, state, state_prime):
        # if str(list(state)) in self.traps:
        #     if str(list(state_prime)) in self.traps:
        #         return 0
        #     else:
        #         return -500000
        r_traps = -2000
        if str(list(state_prime)) in self.traps:
            if str(list(state)) not in self.traps:
                return r_traps
        if str(list(state)) in self.traps:
            if str(list(state_prime)) not in self.traps:
                return r_traps
        # if str(list(state_prime)) in self.traps:
        #     return r_traps    # -300000/-100000/-50000
        if state_prime == self.goal:
            return 0
        return -1  # normal steps


class AMDP_Naive:
    def __init__(self, env=None, uniform_mode=None, gensim_opt=None,):
        assert env != None, "env can't be None!"
        assert (uniform_mode == None) != (gensim_opt == None), "only one of uniform or gensim_opt can be assigned"
        self.env = env
        self.manuel_room_layout = env.room_layout  # np array
        self.goal = env.goal
        # self.flags = env.flags
        if uniform_mode:
            self.abstraction_layout = self.do_tiling_v2(uniform_mode)
        elif gensim_opt:
            self.gensim_opt  = gensim_opt
            self.abstraction_layout = np.array(self.gensim_opt.cluster_layout)
        else:
            raise Exception("invalide mode for AMDP_Topology_Uniform")
        print("print abstraction_layout from AMDP_Topology_Uniform:")
        print(self.abstraction_layout)
        self.goal_abstraction = self.get_abstract_state((self.goal[0], self.goal[1]))
        print("self.goal_abstraction from AMDP_Naive:", self.goal_abstraction)

        # self.flags_found = [0, 0, 0]
        self.list_of_abstract_states = None
        self.adjacencies = None
        self.adjacencies_for_each_astate = None
        self.transition_table = None
        self.rewards_table = None

        self.set_list_of_abstract_state()
        self.set_ajacencies()
        self.set_transitions_and_rewards()

    def do_tiling_v2(self, tiling_size: tuple, ignorewalls=True):
        columns = len(self.manuel_room_layout[0])
        rows = len(self.manuel_room_layout)
        tiling_layout = self.manuel_room_layout.tolist()
        tiling_label = (1, 1)
        for i in range(rows):
            if i != 0 and (i % tiling_size[0]) == 0:
                tiling_label = (tiling_label[0] + 1, 1)
            else:
                tiling_label = (tiling_label[0], 1)
            for j in range(columns):
                if j != 0 and (j % tiling_size[1]) == 0:
                    tiling_label = (tiling_label[0], tiling_label[1] + 1)
                if ignorewalls:
                    tiling_layout[i][j] = str(tiling_label)
                if not ignorewalls and not self.manuel_room_layout[i][j] == "w":
                    tiling_layout[i][j] = str(tiling_label)
        return np.array(tiling_layout)

    def get_abstract_state(self, state) -> str:
        abstract_state = self.abstraction_layout[state[0], state[1]]
        return abstract_state

    def set_list_of_abstract_state(self) -> list:
        # build all the potential abstract states
        list_of_abstract_states = []
        for i in range(len(self.abstraction_layout)):
            for j in range(len(self.abstraction_layout[0])):
                temp_astate = self.get_abstract_state((i, j))  # return a list
                if temp_astate not in list_of_abstract_states:
                    if not self.abstraction_layout[i][j] == "w":
                        list_of_abstract_states.append(temp_astate)
        self.list_of_abstract_states = list_of_abstract_states

    def set_ajacencies(self):
        adjacencies = []
        for i in range(len(self.abstraction_layout) - 1):
            for j in range(len(self.abstraction_layout[0]) - 1):
                current = self.get_abstract_state((i, j))
                down = self.get_abstract_state((i + 1, j))
                right = self.get_abstract_state((i, j + 1))
                if not current == down and not current[0] == "w" and not down[0] == "w":
                    if (current, down) not in adjacencies:
                        adjacencies.append((current, down))
                        adjacencies.append((down, current))
                if not current == right and not current[0] == "w" and not right[0] == "w":
                    if (current, right) not in adjacencies:
                        adjacencies.append((current, right))
                        adjacencies.append((right, current))
            # designed for the state on the right border of the maze
            j = len(self.abstraction_layout[0]) - 1
            current = self.get_abstract_state((i, j))
            down = self.get_abstract_state((i + 1, j))
            if not current == down and not current[0] == "w" and not down[0] == "w":
                if (current, down) not in adjacencies:
                    adjacencies.append((current, down))
                    adjacencies.append((down, current))
        # designed for the state on the bottom border of the maze
        i = len(self.abstraction_layout) - 1
        for j in range(len(self.abstraction_layout[0]) - 1):
            current = self.get_abstract_state((i, j))
            right = self.get_abstract_state((i, j + 1))
            if not current == right and not current[0] == "w" and not right[0] == "w":
                if (current, right) not in adjacencies:
                    adjacencies.append((current, right))
                    adjacencies.append((right, current))

        # self.adjacencies = adjacencies
        # get the adjacencies for each abstraction(tiling)
        # adjacencies_for_each_abstract_state = []
        self.dict_adjacency_for_each_state = {}
        for a in self.list_of_abstract_states:
            # adj = [a]
            adj = []
            for b in self.list_of_abstract_states:
                if (a, b) in adjacencies:
                    adj.append(b)
            self.dict_adjacency_for_each_state[str(a)] = adj
            # adjacencies_for_each_abstract_state.append(adj)
        self.dict_adjacency_for_each_state['bin'] = ['bin']
        # adjacencies_for_each_abstract_state.append(["bin", "bin"])
        # self.adjacencies_for_each_astate = adjacencies_for_each_abstract_state

    def get_abstract_actions(self, abstract_state):
        actions = self.dict_adjacency_for_each_state[str(abstract_state)]
        if abstract_state == self.goal_abstraction:
            actions.append("to_bin")
        return actions

    def set_transitions_and_rewards(self):
        self.list_of_abstract_states.append("bin")
        num_of_abstract_state = len(self.list_of_abstract_states)
        ## Initialise empty action, State, State' transitions and reward
        transition = np.zeros((num_of_abstract_state, num_of_abstract_state, num_of_abstract_state))
        rewards = np.zeros((num_of_abstract_state, num_of_abstract_state, num_of_abstract_state))

        for i in range(num_of_abstract_state):
            ajacency_of_i = self.get_abstract_actions(self.list_of_abstract_states[i])
            # print("len(ajacency_of_i)",len(ajacency_of_i))
            for j in range(num_of_abstract_state):
                # normal transition
                if self.list_of_abstract_states[j] in ajacency_of_i:
                    transition[j, i, j] = 1
                    rewards[j, i, j] = -1
                    # print("set normal transition:",[j, i, j])

            # # goal transition
            if "to_bin" in ajacency_of_i:  # 可修改
                transition[-1, i, -1] = 1
                # rewards[-1, i, -1] = 1000
                # print("goal transition set!!!")
                # print("set goal transition:",[-1, i, -1])
        rewards[-1, -1, -1] = 0

        self.transition_table = transition
        self.rewards_table = rewards

    def solve_amdp(self, synchronous=0, monitor=0):   # streamlined solve_amdp
        print('length of self.list_of_abstract_states:', len(self.list_of_abstract_states))
        print('self.list_of_abstract_states:', self.list_of_abstract_states)
        values = np.zeros(len(self.list_of_abstract_states))
        if synchronous:
            values2 = copy.deepcopy(values)
        print("len(values):", len(values))
        # print("self.transition_table:",np.argwhere(self.transition_table))
        # print("self.rewards_table:",np.argwhere(self.rewards_table))
        delta = 0.2
        theta = 0.1
        print("Value Iteration delta values:")
        while delta > theta:
            delta = 0
            for i in range(0, len(values)):
                v = values[i]
                list_of_values = []
                for a in range(len(values)):
                    if self.transition_table[a, i, a] != 0:
                        value = self.transition_table[a, i, a] * (self.rewards_table[a, i, a] + 0.99 * values[a])
                        list_of_values.append(value)
                if synchronous:
                    values2[i] = max(list_of_values)
                    delta = max(delta, abs(v - values2[i]))
                else:
                    values[i] = max(list_of_values)
                    delta = max(delta, abs(v - values[i]))
            print("delta:", delta)
            if synchronous:
                values = copy.deepcopy(values2)
            if monitor:
                self.plot_current_values(self.env, values)           # plot current values

        # self.plot_current_values(self.env, values)
        # values -= min(values[:-1])
        self.values = values
        self.dict_as2v = dict(zip((i for i in self.list_of_abstract_states), self.values))
        print("self.dict_as2v:")
        print(self.dict_as2v)

    def get_value(self, astate):
        value = self.dict_as2v[str(astate)]
        return value

    def plot_current_values(self, env, values, version=1, text_all_values=0, text_cluster_values=1, show=1, save=1):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig, ax = plt.subplots(figsize=(12, 12))
        my_cmap = copy.copy(plt.cm.get_cmap('hot'))
        values_max = np.amax(values)
        vmax = values_max
        # values_min = np.partition(values, 1)[1]
        values_min = np.min(values[:-1])
        vmin = values_min - 0.1 * (values_max - values_min)
        # vmin = -0.1 * vmax
        print(vmax, vmin)
        my_cmap.set_under('darkred')
        my_cmap.set_bad('lime')
        my_cmap.set_over('dodgerblue')
        asp = 'equal'

        dict_ = dict(zip((i for i in self.list_of_abstract_states), values))
        # print(dict_)
        plate = []      # for plotting graphic
        plate2 = []     # for texting label and values of clusters
        plate3 = []     # for texting values of clusters in each cell
        for i in range(env.size[0]):
            row = []
            row2 = []
            row3 = []
            for j in range(env.size[1]):
                current_state = (i, j)
                a_state = self.get_abstract_state(current_state)
                if current_state in env.valid_states:
                    if current_state == (env.start_state[0], env.start_state[1]):
                        row.append(np.nan)
                        row2.append(a_state)
                        row3.append(dict_[a_state])
                    elif current_state == env.goal:
                        row.append(vmax+1)
                        row2.append(a_state)
                        row3.append(dict_[a_state])
                    else:
                        row.append(dict_[a_state])
                        row2.append(a_state)
                        row3.append(dict_[a_state])
                elif str([i, j]) in env.walls:
                    row.append(vmin)
                    row2.append('w')
                    row3.append('w')
            plate.append(row)
            plate2.append(row2)
            plate3.append(row3)

        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
        # print(plate)
        if text_all_values:
            for i in range(env.size[0]):
                for j in range(env.size[1]):
                    current_state = (i, j)
                    if current_state in env.valid_coords:
                        text_ = round(plate3[i][j])
                        ax.text(j, i, f"{text_}", horizontalalignment='center', verticalalignment='center',
                                fontsize=9, fontweight='semibold', color='k')
        if text_cluster_values:
            plate2 = np.array(plate2)
            for a_state in self.list_of_abstract_states:
                coords = np.argwhere(plate2 == a_state)
                if len(coords) > 0:
                    a_state_head = a_state
                    mean = np.mean(coords, axis=0)
                    v_ = round(dict_[a_state])
                    ax.text(mean[1], mean[0], f"{str(a_state_head)}\n{str(v_)}", horizontalalignment='center', verticalalignment='center',
                            fontsize=10, fontweight='semibold', color='k')

        # fig.subplots_adjust(right=0.85)
        # cax = fig.add_axes([0.9, 0.23, 0.03, 0.5])
        # fig.colorbar(im, cax=cax)
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.1)
        # fig.colorbar(im, cax=cax)
        # fig.colorbar(im, ax=ax, shrink=0.75)
        if show:
            fig.show()
        if save:
            fig.savefig(f"./naive_solved_mdp/{env.maze_name}_big{env.big}_v{version}_uniform", dpi=300)


class AMDP_General_Naive:
    def __init__(self, sentences_period_complete, env=None, gensim_opt=None, save_path=None):  # tiling_mode is same with tiling_size
        self.sentences_period_complete = sentences_period_complete
        assert (env!=None) and (gensim_opt!=None), "env and gensim_opt need to be assigned"
        self.save_path = save_path
        self.env = env
        # self.manuel_layout = env.room_layout     # np array
        self.goal = env.goal
        # self.flags = env.flags

        self.gensim_opt: GensimOperator_General = gensim_opt
        print("self.gensim_opt.sentences[:5]:", self.gensim_opt.sentences[:5])

        self.list_of_abstract_states = [i for i in range(self.gensim_opt.num_clusters)]
        self.dict_gstates_astates = self.gensim_opt.dict_gstates_astates
        # self.dict_gstates_astates = dict(zip(self.gensim_opt.words, self.gensim_opt.cluster_labels.tolist()))
        print("len(gensim_opt.words), len(gensim_opt.cluster_labels):", len(self.gensim_opt.words), len(self.gensim_opt.cluster_labels.tolist()))

        print("start setting amdp transition and reward...")
        self.set_transition_and_rewards3()

    def get_abstract_state(self, state):
        if not isinstance(state, str):
            state = str(state)
        return self.dict_gstates_astates[state]

    def set_transition_and_rewards(self):
        self.list_of_abstract_states.append("bin")
        num_abstract_states = len(self.list_of_abstract_states)    #+1 for absorbing abstract state
        transition = np.zeros(shape=(num_abstract_states, num_abstract_states, num_abstract_states))
        transition2 = np.zeros(shape=(num_abstract_states, num_abstract_states, num_abstract_states))
        rewards = np.zeros(shape=(num_abstract_states, num_abstract_states, num_abstract_states))
        for sentence in self.sentences_period_complete:
            for i in range(len(sentence)):
                if i < (len(sentence)-1):
                    # index1 = self.list_of_ground_states.index(sentence[i])
                    # index2 = self.list_of_ground_states.index(sentence[i+1])
                    # cluster_label1 = self.list_of_abstract_states[index1]
                    # cluster_label2 = self.list_of_abstract_states[index2]
                    cluster_label1 = self.get_abstract_state(sentence[i])
                    cluster_label2 = self.get_abstract_state(sentence[i+1])
                    if not cluster_label1 == cluster_label2:
                        transition[cluster_label2, cluster_label1, cluster_label2] += 1
                        transition2[cluster_label2, cluster_label1, cluster_label2] += 1
                        # transition[cluster_label1, cluster_label2, cluster_label1] = 1
                        # rewards[cluster_label2, cluster_label1, cluster_label2] = -1
                else:
                    # index1 = self.list_of_ground_states.index(sentence[i])
                    # cluster_label1 = self.list_of_abstract_states[index1]
                    cluster_label1 = self.get_abstract_state(sentence[i])
                state_in_tuple = eval(sentence[i])
                if state_in_tuple == (self.goal[0], self.goal[1]):
                    transition[-1, cluster_label1, -1] = 1
                    # transition2[-1, cluster_label1, -1] += 1
                    rewards[-1, cluster_label1, -1] = 1000
        transition[-1, -1, -1] = 1

        valid_transitions = transition[transition>1]
        print("valid_transitions:", valid_transitions)

        bad_valid_transitions = np.partition(valid_transitions, 2)[:10]
        print("bad_transitions:", bad_valid_transitions)

        smallest_valid_trasition = np.partition(valid_transitions, 2)[2]
        print("smallest_valid_trasition:", smallest_valid_trasition)
        transition = np.where((1<transition) & (transition<smallest_valid_trasition), 0, transition)
        transition = np.where(transition >= smallest_valid_trasition, 1, transition)

        for i in range(len(self.list_of_abstract_states)):
            # print(transition2[:, i, :])
            nz = np.nonzero(transition2[:, i, :])
            print(f"t-from-{self.list_of_abstract_states[i]}: {nz}, {transition2[:, i, :][nz]}")

        self.num_abstract_states = num_abstract_states
        self.transition = transition
        self.rewards = rewards

    def set_transition_and_rewards2(self):
        self.list_of_abstract_states.append("bin")
        num_abstract_states = len(self.list_of_abstract_states)    #+1 for absorbing abstract state
        transition = np.zeros(shape=(num_abstract_states, num_abstract_states, num_abstract_states))
        transition2 = np.zeros(shape=(num_abstract_states, num_abstract_states, num_abstract_states))
        rewards = np.zeros(shape=(num_abstract_states, num_abstract_states, num_abstract_states))
        for sentence in self.sentences_period_complete:
            for i in range(len(sentence)):
                state_in_tuple = eval(sentence[i])
                if i < (len(sentence)-1):
                    # index1 = self.list_of_ground_states.index(sentence[i])
                    # index2 = self.list_of_ground_states.index(sentence[i+1])
                    # cluster_label1 = self.list_of_abstract_states[index1]
                    # cluster_label2 = self.list_of_abstract_states[index2]
                    cluster_label1 = self.get_abstract_state(sentence[i])
                    cluster_label2 = self.get_abstract_state(sentence[i+1])
                    if str(list(state_in_tuple)) in self.env.traps:
                        cluster_label1_bad = True
                    else:
                        cluster_label1_bad = False
                    if not cluster_label1 == cluster_label2:
                        transition[cluster_label2, cluster_label1, cluster_label2] = 1
                        transition2[cluster_label2, cluster_label1, cluster_label2] += 1
                        if cluster_label1_bad:
                            rewards[cluster_label2, cluster_label1, cluster_label2] = -100
                        # transition[cluster_label1, cluster_label2, cluster_label1] = 1
                        # rewards[cluster_label2, cluster_label1, cluster_label2] = -1
                else:
                    # index1 = self.list_of_ground_states.index(sentence[i])
                    # cluster_label1 = self.list_of_abstract_states[index1]
                    cluster_label1 = self.get_abstract_state(sentence[i])
                if state_in_tuple == (self.goal[0], self.goal[1]):
                    transition[-1, cluster_label1, -1] = 1
                    # transition2[-1, cluster_label1, -1] += 1
                    rewards[-1, cluster_label1, -1] = 1000
        transition[-1, -1, -1] = 1

        for i in range(len(self.list_of_abstract_states)):
            # print(transition2[:, i, :])
            nz = np.nonzero(transition2[:, i, :])
            print(f"t-from-{self.list_of_abstract_states[i]}: {nz}, {transition2[:, i, :][nz]}")

        self.num_abstract_states = num_abstract_states
        self.transition = transition
        self.rewards = rewards

    def set_transition_and_rewards3(self):
        self.list_of_abstract_states.append("bin")
        num_abstract_states = len(self.list_of_abstract_states)    #+1 for absorbing abstract state
        transition = np.zeros(shape=(num_abstract_states, num_abstract_states, num_abstract_states))
        transition2 = np.zeros(shape=(num_abstract_states, num_abstract_states, num_abstract_states))
        rewards = np.zeros(shape=(num_abstract_states, num_abstract_states, num_abstract_states))
        for sentence in self.sentences_period_complete:
            for i in range(len(sentence)):
                state_in_tuple = eval(sentence[i])
                if i < (len(sentence)-1):
                    # index1 = self.list_of_ground_states.index(sentence[i])
                    # index2 = self.list_of_ground_states.index(sentence[i+1])
                    # cluster_label1 = self.list_of_abstract_states[index1]
                    # cluster_label2 = self.list_of_abstract_states[index2]
                    cluster_label1 = self.get_abstract_state(sentence[i])
                    cluster_label2 = self.get_abstract_state(sentence[i+1])
                    if str(list(state_in_tuple)) in self.env.traps:
                        cluster_label1_bad = True
                    else:
                        cluster_label1_bad = False
                    if not cluster_label1 == cluster_label2:
                        transition[cluster_label2, cluster_label1, cluster_label2] = 1
                        transition2[cluster_label2, cluster_label1, cluster_label2] += 1
                        if cluster_label1_bad:
                            rewards[cluster_label2, cluster_label1, cluster_label2] = -10
                        else:
                            rewards[cluster_label2, cluster_label1, cluster_label2] = -1
                        # transition[cluster_label1, cluster_label2, cluster_label1] = 1
                        # rewards[cluster_label2, cluster_label1, cluster_label2] = -1
                else:
                    # index1 = self.list_of_ground_states.index(sentence[i])
                    # cluster_label1 = self.list_of_abstract_states[index1]
                    cluster_label1 = self.get_abstract_state(sentence[i])
                if state_in_tuple == (self.goal[0], self.goal[1]):
                    transition[-1, cluster_label1, -1] = 1
                    # transition2[-1, cluster_label1, -1] += 1
                    # rewards[-1, cluster_label1, -1] = 1000
        transition[-1, -1, -1] = 1

        for i in range(len(self.list_of_abstract_states)):
            # print(transition2[:, i, :])
            nz = np.nonzero(transition2[:, i, :])
            print(f"t-from-{self.list_of_abstract_states[i]}: {nz}, {transition2[:, i, :][nz]}")

        self.num_abstract_states = num_abstract_states
        self.transition = transition
        self.rewards = rewards

    def solve_amdp(self, synchronous=0, monitor=0):
        values = np.zeros(self.num_abstract_states)
        if synchronous:
            values2 = copy.deepcopy(values)
        print("len(values):", len(values))
        delta = 0.2
        theta = 0.1
        print("Value Iteration delta values:")
        while delta > theta:
            delta = 0
            for i in range(0, len(values)):
                v = values[i]
                list_of_values = []
                for a in range(len(values)):
                    if self.transition[a, i, a] != 0:
                        value = self.transition[a, i, a] * (self.rewards[a, i, a] + 0.99 * values[a])
                        list_of_values.append(value)
                if synchronous:
                    values2[i] = max(list_of_values)
                    delta = max(delta, abs(v - values2[i]))
                else:
                    values[i] = max(list_of_values)
                    delta = max(delta, abs(v - values[i]))
            print("delta:", delta)
            if synchronous:
                values = copy.deepcopy(values2)
            if monitor:
                self.plot_current_values(self.env, values)            # plot current values
        # print(V)
        # self.plot_current_values(self.env, values)
        # values -= min(values[:-1])
        self.values = values
        # print("self.values_of_abstract_states:")
        # print(self.values_of_abstract_states)
        # print(len(self.list_of_abstract_states), len(self.values_of_abstract_states))
        self.dict_as_v = dict(zip((str(i) for i in self.list_of_abstract_states), self.values))
        print("self.dict_as_v:")
        print(self.dict_as_v)

    def get_value(self, astate):
        assert isinstance(astate, int), "astate has to be int"
        value = self.values[astate]
        # print("value:",value)
        return value

    def plot_current_values(self, env, values, version=1, text_all_values=0, text_cluster_values=0, show=1, save=0):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig, ax = plt.subplots(figsize=(15, 12))
        my_cmap = copy.copy(plt.cm.get_cmap('hot'))
        values_max = np.amax(values)
        vmax = values_max
        # values_min = np.partition(values, 1)[1]
        values_min = np.min(values[:-1])
        vmin = values_min - 0.1 * (values_max - values_min)
        # vmin = -0.1 * vmax
        print(vmax, vmin)
        my_cmap.set_under('darkred')
        my_cmap.set_bad('lime')
        my_cmap.set_over('dodgerblue')
        asp = 'equal'

        dict_ = dict(zip((i for i in self.list_of_abstract_states), values))
        # print(dict_)
        plate = []      # for plotting graphic
        plate2 = []     # for texting label and values of clusters
        plate3 = []     # for texting values of clusters in each cell
        for i in range(env.size[0]):
            row = []
            row2 = []
            row3 = []
            for j in range(env.size[1]):
                current_state = (i, j)
                if current_state in env.valid_states:
                    a_state = self.get_abstract_state(current_state)
                    if current_state == (env.start_state[0], env.start_state[1]):
                        row.append(np.nan)
                        row2.append(str(a_state))
                        row3.append(dict_[a_state])
                    elif current_state == env.goal:
                        row.append(vmax+1)
                        row2.append(str(a_state))
                        row3.append(dict_[a_state])
                    else:
                        row.append(dict_[a_state])
                        row2.append(str(a_state))
                        row3.append(dict_[a_state])
                elif str([i, j]) in env.walls:
                    row.append(vmin)
                    row2.append('w')
                    row3.append('w')
            plate.append(row)
            plate2.append(row2)
            plate3.append(row3)

        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
        # print(plate)
        if text_all_values:
            for i in range(env.size[0]):
                for j in range(env.size[1]):
                    current_state = (i, j)
                    if current_state in env.valid_coords:
                        text_ = round(plate3[i][j])
                        ax.text(j, i, f"{text_}", horizontalalignment='center', verticalalignment='center',
                                fontsize=9, fontweight='semibold', color='k')
        if text_cluster_values:
            plate2 = np.array(plate2)
            for a_state in self.list_of_abstract_states:
                coords = np.argwhere(plate2 == str(a_state))
                if len(coords) > 0:
                    # a_state_head = a_state
                    mean = np.mean(coords, axis=0)
                    v_ = round(dict_[a_state])
                    ax.text(mean[1], mean[0], f"{str(a_state)}\n{str(v_)}", horizontalalignment='center', verticalalignment='center',
                            fontsize=10, fontweight='semibold', color='k')

        # fig.subplots_adjust(right=0.85)
        # cax = fig.add_axes([0.9, 0.23, 0.03, 0.5])
        # fig.colorbar(im, cax=cax)
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.1)
        # fig.colorbar(im, cax=cax)
        # fig.colorbar(im, ax=ax, shrink=0.75)
        if show:
            fig.show()
        if save:
            fig.savefig(f"./{self.save_path}/solved_AMDP")


class ExploreBrainNaive:
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
            action = np.random.choice(actions)
            return action
        if self.q2_init:
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

    def learn_explore_sarsa(self, state1, action1, state2, action2, reward):
        max_value = self.q_table2[state2[0], state2[1], action2]

        delta = reward + (self.gamma * max_value) - self.q_table2[state1[0], state1[1], action1]

        self.q_table2[state1[0], state1[1], action1] += self.lr * delta

    def policy_explore_softmax(self, state, actions):
        ##epsilon greedy policy
        ran = np.random.randint(100)
        if ran < self.epsilon * 100:
            action = np.random.choice(actions)
            # self.state_actions_long_life[state[0], state[1], action] -= 1
            return action
        # sum = np.sum([np.exp(self.state_actions_long_life[state[0], state[1], a]) for a in actions])
        probs = scipy.special.softmax([self.state_actions_long_life[state[0], state[1], a]
                                       for a in actions])
        action = np.random.choice(actions, p=probs)
        # self.state_actions_long_life[state[0], state[1], action] -= 1
        return action


class ExperimentMakerNaive:
    def __init__(self, env_name, big, q_eps, interpreter, print_to_file, plot_maker: PlotMakerNaive):
        self.env_name = env_name
        self.big = big

        self.interpreter = interpreter
        self.print_to_file = print_to_file
        self.env = MazeNaive(maze=self.env_name, big=self.big)
        self.plot_maker = plot_maker

        plt.rcParams['agg.path.chunksize'] = 10000
        plt.rcParams['lines.linewidth'] = 1

        self.flags_episodes_repetitions = []
        self.reward_episodes_repetitions = []
        self.move_count_episodes_repetitions = []
        # self.flags_found_order_episodes_repetitions = []
        # self.path_episodes_repetitions = []
        self.experiment_time_repetitions = []
        self.solve_amdp_time_repetitions = []
        self.ground_learning_time_repetitions = []

        self.ground_learning_config = {
            'q_mode': 'Q(Lambda)',
            'q_eps': q_eps,
            'lr': 0.1,
            'lambda': 0.9,
            'gamma': 0.999,
            'omega': 100,
            'epsilon_q_max': 1,
            'epsilon_q_min': 0.1,
            'epsilon_q_max2': 1,  # used when evo2 comes
            'epsilon_q_min2': 0.1,  # used when evo2 comes
        }

    def _build_and_solve_amdp(self, tiling_size: tuple = None, gensim_opt = None, general=0):
        print("-----Begin build and solve amdp-----")
        assert (tiling_size == None) != (gensim_opt == None), "tiling_size and gensim_opt can't be both None"
        if tiling_size:
            amdp = AMDP_Naive(env=self.env, uniform_mode=tiling_size, gensim_opt=None)
        elif gensim_opt and general == 0:
            amdp = AMDP_Naive(env=self.env, uniform_mode=None, gensim_opt=gensim_opt)
        elif gensim_opt and general == 1:
            amdp = AMDP_General(env=self.env, gensim_opt=gensim_opt)
        else:
            raise Exception("wrong parameters")
        print("start solving amdp...")
        start_amdp = time.time()
        amdp.solve_amdp()
        end_amdp = time.time()
        solve_amdp_time = end_amdp - start_amdp
        print("solve_amdp_time:", solve_amdp_time)
        self.solve_amdp_time_repetitions.append(solve_amdp_time)
        print("-----Finish build and solve amdp-----")
        return amdp

    def _solve_amdp(self, amdp, syn=0):
        print("-----Begin solving amdp-----")
        start_amdp = time.time()
        amdp.solve_amdp(synchronous=syn, monitor=0)
        end_amdp = time.time()
        solve_amdp_time = end_amdp - start_amdp
        print("solve_amdp_time:", solve_amdp_time)
        self.solve_amdp_time_repetitions.append(solve_amdp_time)
        print("-----Finish solving amdp-----")

    def _ground_learning(self, amdp):
        print("-----Begin Ground Learning-----")
        start_ground_learning = time.time()
        agent_q = QLambdaBrain(env=self.env, ground_learning_config=self.ground_learning_config)
        env = self.env
        for ep in range(self.ground_learning_config['q_eps']):
            if (ep + 1) % 100 == 0:
                print(f"episode_100: {ep} | avg_move_count: {int(np.mean(self.move_count_episodes[-100:]))} | "
                      f"avd_reward: {int(np.mean(self.reward_episodes[-100:]))} | "
                      f"env.state: {env.state} | "
                      f"env.flagcollected: {env.flags_collected} | "
                      f"agent.epsilon: {agent_q.epsilon} | "
                      f"agent.lr: {agent_q.lr}")
            # set epsilon
            temp_epsilon = self.ground_learning_config['epsilon_q_max'] - (self.ground_learning_config['epsilon_q_max'] / self.ground_learning_config['q_eps']) * ep
            if temp_epsilon > 0.1:
                agent_q.epsilon = round(temp_epsilon, 5)

            env.reset()
            agent_q.reset_eligibility()
            episode_reward = 0
            move_count = 0
            track = [str((env.state[0], env.state[1]))]
            a = agent_q.policy(env.state, env.actions(env.state))
            while not env.isTerminal(env.state):
                abstract_state = amdp.get_abstract_state(env.state)
                new_state = env.step(env.state, a)
                move_count += 1
                track.append(str((new_state[0], new_state[1])))
                new_abstract_state = amdp.get_abstract_state(new_state)
                a_prime = agent_q.policy(new_state, env.actions(new_state))  ##Greedy next-action selected
                a_star = agent_q.policyNoRand(new_state, env.actions(new_state))
                r = env.reward(env.state, a, new_state)  ## ground level reward
                episode_reward += r

                value_new_abstract_state = amdp.get_value(new_abstract_state)
                value_abstract_state = amdp.get_value(abstract_state)
                shaping = self.ground_learning_config['gamma'] * value_new_abstract_state * \
                          self.ground_learning_config['omega'] - value_abstract_state * self.ground_learning_config['omega']
                # shaping = 0
                agent_q.learn(env.state, a, new_state, a_prime, a_star, r + shaping)

                env.state = new_state
                a = a_prime

            self.reward_episodes.append(episode_reward)
            self.flags_episodes.append(env.flags_collected)
            self.move_count_episodes.append(move_count)
            self.flags_found_order_episodes.append(env.flags_found_order)

            self.epsilons_episodes.append(agent_q.epsilon)
            self.gamma_episodes.append(agent_q.gamma)
            self.lr_episodes.append(agent_q.lr)

        end_ground_learning = time.time()
        ground_learning_time = end_ground_learning - start_ground_learning
        print("ground_learning_time:", ground_learning_time)
        self.ground_learning_time_repetitions.append(ground_learning_time)

        print("-----Finish Ground Learning-----")

        return agent_q


class UniformExpMakerNaive(ExperimentMakerNaive):
    def __init__(self, env_name: str, big: int, tiling_size: tuple, q_eps: int, repetitions: int, interpreter: str,
                 print_to_file: int, plot_maker: PlotMakerNaive, path_results: str):

        super().__init__(env_name, big, q_eps, interpreter, print_to_file, plot_maker)

        self.tiling_size = tiling_size
        self.repetitions = repetitions
        # self.num_total_eps = self.ground_learning_config['q_eps']
        self.path_results = path_results
        # self.rolling_window_size = int(self.ground_learning_config['q_eps'] / 30)

        # self.experiment_time_repetitions = []
        # self.exploration_time_repetitions = []
        # self.solve_word2vec_time_repetitions = []
        # self.solve_amdp_time_repetitions = []
        # self.ground_learning_time_repetitions = []

    def _print_before_start(self):
        print("+++++++++++start UniformExpMaker.run()+++++++++++")
        print("PID: ", os.getpid())
        print("=path_results=:", self.path_results)
        print(f"maze:{self.env_name} | big:{self.big} | repetitions:{self.repetitions} | interpreter:{self.interpreter} |"
              f"print_to_file: {self.print_to_file}")
        print(f"=explore_config=: Nothing to show")
        print(f"=w2v_config=: Nothing to show")
        print(f"=ground_learning_config=: {self.ground_learning_config}")

    def _results_upload(self):
        print("============upload experiments details to google sheets============")
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials

        scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
                 "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

        creds = ServiceAccountCredentials.from_json_keyfile_name("google_creds.json", scope)
        client = gspread.authorize(creds)
        # sheet = client.open("experiments_result").sheet1  # Open the spreadhseet
        sheet = client.open("experiments_result").worksheet("OOP")
        # gspread api to get worksheet
        ### worksheet = sh.get_worksheet(0)
        ### worksheet = sh.worksheet("January")

        mean_by_rep_experiment_time = np.mean(self.experiment_time_repetitions)
        mean_by_rep_exploration_time = 0
        mean_by_rep_word2vec_time = 0
        mean_by_rep_kmeans_time = 0
        mean_by_rep_amdp_time = np.mean(self.solve_amdp_time_repetitions)
        mean_by_rep_q_time = np.mean(self.ground_learning_time_repetitions)
        data = [mean_by_rep_experiment_time,
                mean_by_rep_exploration_time,
                mean_by_rep_word2vec_time,
                mean_by_rep_kmeans_time,
                mean_by_rep_amdp_time,
                mean_by_rep_q_time]
        data = [round(item, 1) for item in data]

        rolling_window_size = int(self.ground_learning_config['q_eps'] / 50)
        final_reward = statistics.mean(np.mean(self.reward_episodes_repetitions, axis=0)[-rolling_window_size:])

        total_steps = np.cumsum(np.mean(self.move_count_episodes_repetitions, axis=0))[-1]
        abstraction_mode = 'uniform'
        insert_row = [self.env.maze_name, self.big, abstraction_mode, '--', '--', '--', '--', '--', data[1],
                      '--', '--',
                      '--', '--', '--', '--', data[2], '--', '--',
                      data[4], self.ground_learning_config['q_eps'], data[5], self.repetitions, self.interpreter, data[0],
                      round(final_reward, 2), total_steps, self.ground_learning_config['lr'], self.ground_learning_config['gamma'],
                      self.ground_learning_config['lambda'], self.ground_learning_config['omega'], '--',
                      "1-0.1", self.plot_maker.std_factor, self.path_results]

        sheet.append_row(insert_row)
        print("uniform approach uploaded to google sheet")
        print(" FINISHED!")

    def run(self, time_comparison=0):
        if not os.path.isdir(self.path_results):
            makedirs(self.path_results)
        self._print_before_start()
        # curve_label = f"u-{self.tiling_size[0]}x{self.tiling_size[1]}"
        curve_label = f"u-{math.ceil(self.env.size[0]/self.tiling_size[0]) * math.ceil(self.env.size[1]/self.tiling_size[1])}"
        for rep in range(self.repetitions):
            print(f"+++++++++ Begin repetition: {rep} +++++++++")
            # experiment timing start
            start_experiment = time.time()

            self.flags_episodes = []
            self.reward_episodes = []
            self.move_count_episodes = []
            self.flags_found_order_episodes = []
            self.epsilons_episodes = []
            self.lr_episodes = []
            self.gamma_episodes = []

            self.sentences_collected = []
            self.sentences_period = []

            # build and solve amdp
            # amdp = AMDP_Topology_Uniform(env=self.env, uniform_mode=self.tiling_size, gensim_opt=None)
            amdp = AMDP_Naive(env=self.env, uniform_mode=self.tiling_size, gensim_opt=None)
            self._solve_amdp(amdp)
            self.plot_maker.plot_each_cluster_layout_and_values(self.env, amdp, "uniform", rep,
                                                                ax_title=None, save_path=self.path_results, save=0, show=1)

            # ground learning
            # self._ground_learning(amdp)

            # experiment timing ends and saved
            end_experiment = time.time()
            experiment_time = end_experiment-start_experiment
            self.experiment_time_repetitions.append(experiment_time)

            # plot flags, reward, move_count for each rep
            self.plot_maker.plot_each_reward_movecount(self.reward_episodes,
                                                       self.move_count_episodes,
                                                       rep, curve_label)

            # save performance of each rep
            self.flags_episodes_repetitions.append(self.flags_episodes)
            self.reward_episodes_repetitions.append(self.reward_episodes)
            self.move_count_episodes_repetitions.append(self.move_count_episodes)

        # plot mean performance among all reps
        # ax_title = f"flags collection in {'big' if self.big == 1 else 'small'} {self.env.maze_name}"

        self.plot_maker.plot_mean_performance_across_reps(self.reward_episodes_repetitions,
                                                          self.move_count_episodes_repetitions, curve_label)
        if time_comparison:
            self.plot_maker.plot_mean_time_comparison(self.experiment_time_repetitions, self.solve_amdp_time_repetitions,
                                                      self.ground_learning_time_repetitions, bar_label=curve_label)

        # self._results_upload()


class GeneralExpMakerNaive(ExperimentMakerNaive):
    def __init__(self, env_name: str, big: int, e_mode: str, e_start: str, e_eps: int, mm: int, ds_factor, ds_repetitions,
                 rep_size: int, win_size: int, sg: int, num_clusters: int, k_means_pkg: str, q_eps: int,
                 repetitions: int, interpreter: str, print_to_file: int, plot_maker: PlotMakerNaive, path_results: str):

        super().__init__(env_name, big, q_eps, interpreter, print_to_file, plot_maker)

        self.explore_config = {
            'e_mode': e_mode,
            'e_start': e_start,
            'e_eps': e_eps,
            'max_move_count': mm,
            'ds_factor': ds_factor,
            'ds_repetitions': ds_repetitions,
            'lr': 0.1,
            'gamma': 0.999,
            'epsilon_e': 0.01,
        }

        self.w2v_config = {
            'rep_size': rep_size,
            'win_size': win_size,
            'sg': sg,
            'workers': 32
        }

        self.num_clusters = num_clusters
        self.k_means_pkg = k_means_pkg
        self.repetitions = repetitions
        # self.num_total_eps = self.explore_config['e_eps'] + self.ground_learning_config['q_eps']
        self.path_results = path_results
        # self.rolling_window_size = int(self.ground_learning_config['q_eps'] / 30)

        # self.experiment_time_repetitions = []
        self.exploration_time_repetitions = []
        self.solve_word2vec_time_repetitions = []
        self.solve_kmeans_time_repetitions = []

        self.longlife_exploration_std_repetitions = []
        self.longlife_exploration_mean_repetitions = []

    def _print_before_start(self):
        print("+++++++++++start GeneralExpMaker.run()+++++++++++")
        print("PID: ", os.getpid())
        print("=path_results=:", self.path_results)
        print(f"maze:{self.env_name} | big:{self.big} | repetitions:{self.repetitions} | interpreter:{self.interpreter} |"
              f"print_to_file: {self.print_to_file}")
        print(f"=explore_config=: {self.explore_config}")
        print(f"=w2v_config=: {self.w2v_config}")
        print(f"=ground_learning_config=: {self.ground_learning_config}")
        print(f"num_clusters: {self.num_clusters} | k_means_pkg: {self.k_means_pkg}")

    def _explore(self):
        print("-----Begin Exploration-----")
        start_exploration = time.time()
        agent_e = ExploreBrainNaive(env=self.env, explore_config=self.explore_config)
        env = self.env
        valid_states_ = tuple(env.valid_states)
        for ep in range(self.explore_config['e_eps']):
            # if (ep+1) % int(self.explore_config['e_eps']/10) == 0:
            #     self.plot_maker.plot_each_heatmap(agent_e, 0, None, None)
            if (ep + 1) % 100 == 0:
                print(f"episode_100: {ep} | avg_move_count: {int(np.mean(self.move_count_episodes[-100:]))} | "
                      f"avd_reward: {int(np.mean(self.reward_episodes[-100:]))} | "
                      f"env.state: {env.state} | "
                      f"env.flagcollected: {env.flags_collected} | "
                      f"agent.epsilon: {agent_e.epsilon} | "
                      f"agent.lr: {agent_e.lr}")
            move_count = 0
            episode_reward = 0

            agent_e.epsilon = self.explore_config['epsilon_e']

            if self.explore_config['e_start'] == 'random':
                env.reset()
                start_state = random.choice(valid_states_)
                env.state = start_state
            elif self.explore_config['e_start'] == 'last':
                start_state = env.state
                env.reset()
                env.state = start_state
            elif self.explore_config['e_start'] == 'origin':
                env.reset()
            else:
                raise Exception("Invalid self.explore_config['e_start']")

            agent_e.reset_episodic_staff()

            if self.explore_config['e_mode'] == 'sarsa':
                track = [str(env.state)]
                a = agent_e.policy_explore_rl(env.state, env.actions(env.state))
                while move_count < self.explore_config['max_move_count']:
                    agent_e.states_episodic[env.state[0], env.state[1]] += 1
                    agent_e.states_long_life[env.state[0], env.state[1]] += 1
                    new_state = env.step(env.state, a)
                    move_count += 1
                    track.append(str(new_state))

                    r1 = -agent_e.states_long_life[new_state[0], new_state[1]]
                    # r2 = -agent.states_episodic[new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]]
                    # beta = ep / num_explore_episodes
                    # r = (1 - beta) * r1 + (beta) * r2
                    # r = (1 - beta) * r2 + (beta) * r1
                    r2 = env.reward(env.state, new_state)
                    if r2 < -1:
                        r = r1*1 + r2
                        # print("trap, r:", r)
                    else:
                        r = r1
                        r *= 1
                    # r = r1
                    # r *= 10
                    a_prime = agent_e.policy_explore_rl(new_state, env.actions(new_state))
                    # a_star = agent_e.policyNoRand_explore_rl(new_state, env.actions(new_state))
                    agent_e.learn_explore_sarsa(env.state, a, new_state, a_prime, r)
                    env.state = new_state
                    a = a_prime
            elif self.explore_config['e_mode'] == 'softmax':
                track = [str(env.state)]
                a = agent_e.policy_explore_softmax(env.state, env.actions(env.state))
                while move_count < self.explore_config['max_move_count']:
                    agent_e.state_actions_long_life[env.state[0], env.state[1], a] -= 1
                    new_state = env.step(env.state, a)
                    move_count += 1
                    track.append(str(new_state))
                    a_prime = agent_e.policy_explore_softmax(new_state, env.actions(new_state))
                    env.state = new_state
                    a = a_prime
            else:
                raise Exception("Invalid self.explore_config['e_mode']")

            self.reward_episodes.append(episode_reward)
            self.flags_episodes.append(env.flags_collected)
            self.move_count_episodes.append(move_count)
            self.flags_found_order_episodes.append(env.flags_found_order)

            self.epsilons_episodes.append(agent_e.epsilon)
            self.gamma_episodes.append(agent_e.gamma)
            self.lr_episodes.append(agent_e.lr)

            if self.explore_config['ds_factor'] == 1:
                self.sentences_period.append(track)
                self.sentences_period_complete.append(track)
            else:
                for _ in range(self.explore_config['ds_repetitions']):
                    down_sampled = [track[index] for index in sorted(random.sample(range(len(track)),
                                    math.floor(len(track) * self.explore_config['ds_factor'])))]
                    self.sentences_period.append(down_sampled)
                self.sentences_period_complete.append(track)

            # print("np.std(agent.states_episodic):", np.std(agent.states_episodic[agent.states_episodic > 0]))

        end_exploration = time.time()
        exploration_time = end_exploration - start_exploration
        print("exploration_time:", exploration_time)
        self.exploration_time_repetitions.append(exploration_time)

        self.longlife_exploration_std_repetitions.append(np.std(agent_e.states_long_life[agent_e.states_long_life > 0]))
        print("longlife_exploration_std:", self.longlife_exploration_std_repetitions)
        self.longlife_exploration_mean_repetitions.append(np.mean(agent_e.states_long_life[agent_e.states_long_life > 0]))
        print("longlife_exploration_mean:", self.longlife_exploration_mean_repetitions)
        print("longlife_exploration_sum:", np.sum(agent_e.states_long_life[agent_e.states_long_life > 0]))

        # check sentences_period
        print("len of self.sentences_period:", len(self.sentences_period))
        flatten_list = list(chain.from_iterable(self.sentences_period))
        counter_dict = Counter(flatten_list)
        print("min counter value:", min(counter_dict.values()))
        under5 = [k for k, v in counter_dict.items() if v < 5]
        print("under5:", under5)
        print("under5 length:", len(under5))
        print("len(flatten_list):", len(flatten_list))
        print("unique len(flatten_list)", len(set(flatten_list)))

        self.sentences_collected.extend(self.sentences_period)
        self.sentences_period = []

        print("-----Finish Exploration-----")
        return agent_e

    def _w2v_and_kmeans(self):
        print("-----Begin w2v and k-means-----")
        random.shuffle(self.sentences_collected)
        gensim_opt = GensimOperator_General(self.env)
        solve_wor2vec_time, solve_kmeans_time = gensim_opt.get_cluster_labels(sentences=self.sentences_collected,
                                                 size=self.w2v_config['rep_size'],
                                                 window=self.w2v_config['win_size'],
                                                 clusters=self.num_clusters,
                                                 skip_gram=self.w2v_config['sg'],
                                                 workers=self.w2v_config['workers'],
                                                 package=self.k_means_pkg)
        self.solve_word2vec_time_repetitions.append(solve_wor2vec_time)
        self.solve_kmeans_time_repetitions.append(solve_kmeans_time)
        print("-----Finish w2v and k-means-----")
        return gensim_opt


    def _results_upload(self):
        print("============upload experiments details to google sheets============")
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials

        scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
                 "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

        creds = ServiceAccountCredentials.from_json_keyfile_name("google_creds.json", scope)
        client = gspread.authorize(creds)
        # sheet = client.open("experiments_result").sheet1  # Open the spreadhseet
        sheet = client.open("experiments_result").worksheet("OOP")
        # gspread api to get worksheet
        ### worksheet = sh.get_worksheet(0)
        ### worksheet = sh.worksheet("January")

        mean_by_rep_experiment_time = np.mean(self.experiment_time_repetitions)
        mean_by_rep_exploration_time = np.mean(self.exploration_time_repetitions)
        mean_by_rep_word2vec_time = np.mean(self.solve_word2vec_time_repetitions)
        mean_by_rep_kmeans_time = np.mean(self.solve_kmeans_time_repetitions)
        mean_by_rep_amdp_time = np.mean(self.solve_amdp_time_repetitions)
        mean_by_rep_q_time = np.mean(self.ground_learning_time_repetitions)
        data = [mean_by_rep_experiment_time,
                mean_by_rep_exploration_time,
                mean_by_rep_word2vec_time,
                mean_by_rep_kmeans_time,
                mean_by_rep_amdp_time,
                mean_by_rep_q_time]
        data = [round(item, 1) for item in data]

        rolling_window_size = int(self.ground_learning_config['q_eps'] / 50)
        final_reward = statistics.mean(np.mean(self.reward_episodes_repetitions, axis=0)[-rolling_window_size:])
        total_steps = np.cumsum(np.mean(self.move_count_episodes_repetitions, axis=0)[self.explore_config['e_eps']:])[-1]
        w2v = 'SG' if self.w2v_config['sg'] == 1 else 'CBOW'
        negative = 5
        abstraction_mode = 'general'
        insert_row = [self.env.maze_name, self.big, abstraction_mode, self.explore_config['e_mode'], self.explore_config['e_start'],
                      self.explore_config['e_eps'], self.explore_config['max_move_count'], self.explore_config['ds_factor'], data[1],
                      round(statistics.mean(self.longlife_exploration_mean_repetitions), 2),
                      round(statistics.mean(self.longlife_exploration_std_repetitions), 2),
                      self.w2v_config['rep_size'], self.w2v_config['win_size'], w2v, negative, data[2], f"{data[3]}-{self.k_means_pkg}", self.num_clusters,
                      data[4], self.ground_learning_config['q_eps'], data[5], self.repetitions, self.interpreter, data[0],
                      round(final_reward, 2), total_steps, self.ground_learning_config['lr'], self.ground_learning_config['gamma'],
                      self.ground_learning_config['lambda'], self.ground_learning_config['omega'], self.explore_config['epsilon_e'],
                      "1-0.1", self.plot_maker.std_factor, self.path_results]
        sheet.append_row(insert_row)
        print("general approach uploaded to google sheet")
        print(" FINISHED!")

    def run(self, p_heatmap=1, p_cluster_layout_and_values=1, time_comparison=0):
        if not os.path.isdir(self.path_results):
            makedirs(self.path_results)
        self._print_before_start()
        curve_label = f"g-{self.num_clusters}"
        for rep in range(self.repetitions):
            print(f"+++++++++ Begin repetition: {rep} +++++++++")
            # experiment timing start
            start_experiment = time.time()
            # self.experiment_time = 0
            # self.exploration_time = 0
            # self.solve_word2vec_time = 0
            # self.solve_amdp_time = 0
            # self.solve_q_time = 0

            self.flags_episodes = []
            self.reward_episodes = []
            self.move_count_episodes = []
            self.flags_found_order_episodes = []
            self.epsilons_episodes = []
            self.lr_episodes = []
            self.gamma_episodes = []

            self.sentences_collected = []
            self.sentences_period = []      # for gensim w2v
            self.sentences_period_complete = []     # for building amdp transition

            # exploration
            agent_e = self._explore()
            if p_heatmap:
                self.plot_maker.plot_each_heatmap(agent_e, rep, ax_title=f"exploration-{self.explore_config['e_start']}", save_path=self.path_results)

            # solve w2v and k-means to get clusters and save cluster file
            gensim_opt = self._w2v_and_kmeans()

            # build and solve amdp
            amdp = AMDP_General_Naive(self.sentences_period_complete, env=self.env, gensim_opt=gensim_opt, save_path=self.path_results)
            self._solve_amdp(amdp)
            if p_cluster_layout_and_values:
                ax_title = f"clusters{self.num_clusters}mm{self.explore_config['max_move_count']}s" \
                           f"{self.w2v_config['rep_size']}w{self.w2v_config['win_size']}sg{self.w2v_config['sg']}"
                self.plot_maker.plot_each_cluster_layout_and_values(self.env, amdp, "general", rep, ax_title=ax_title, save_path=self.path_results, save=0, show=0)

            # ground learning
            # self._ground_learning(amdp)

            # experiment timing ends and saved
            end_experiment = time.time()
            experiment_time = end_experiment-start_experiment
            self.experiment_time_repetitions.append(experiment_time)

            # plot flags, reward, move_count for each rep
            self.plot_maker.plot_each_reward_movecount(self.reward_episodes[self.explore_config['e_eps']:],
                                                       self.move_count_episodes[self.explore_config['e_eps']:],
                                                       rep, curve_label)

            # save performance of each rep
            self.flags_episodes_repetitions.append(self.flags_episodes)
            self.reward_episodes_repetitions.append(self.reward_episodes)
            self.move_count_episodes_repetitions.append(self.move_count_episodes)

        # plot mean performance among all reps
        ### ax_title = f"flags collection in {'big' if self.big==1 else 'small'} {self.env.maze_name}"

        sliced_f_ep_rep = np.array(self.flags_episodes_repetitions)[:, self.explore_config['e_eps']:]
        sliced_r_ep_rep = np.array(self.reward_episodes_repetitions)[:, self.explore_config['e_eps']:]
        sliced_m_ep_rep = np.array(self.move_count_episodes_repetitions)[:, self.explore_config['e_eps']:]
        self.plot_maker.plot_mean_performance_across_reps(sliced_r_ep_rep, sliced_m_ep_rep, curve_label)

        if time_comparison:
            self.plot_maker.plot_mean_time_comparison(self.experiment_time_repetitions, self.solve_amdp_time_repetitions,
                                                      self.ground_learning_time_repetitions, self.exploration_time_repetitions,
                                                      self.solve_word2vec_time_repetitions, self.solve_kmeans_time_repetitions,
                                                      bar_label=curve_label)

        # self._results_upload()


def compare_approaches():
    maze = 'basic3'  # low_connectivity2/external_maze21x21_1/external_maze31x31_2/strips2/spiral/basic/open_space/high_connectivity
    big = 1
    e_mode = 'sarsa'  # 'sarsa' or 'softmax'
    e_start = 'last'  # 'random' or 'last' or 'semi_random'
    e_eps = 5000        # 3000 / 20000
    mm = 100
    ds_factor = 0.5
    ds_repetitions = 4

    q_eps = 500
    repetitions = 4
    rep_size = 128
    win_size = 50
    sg = 1  # 'SG' or 'CBOW'
    ng = 5
    # numbers_of_clusters = [9, 16, 25, 36]     # number of abstract states for Uniform will be matched with the number of clusters
    numbers_of_clusters = [16]  # number of abstract states for Uniform will be matched with the number of clusters

    k_means_pkg = 'sklearn'  # 'sklearn' or 'nltk'
    interpreter = 'R'  # L or R
    std_factor = 1 / np.sqrt(10)

    print_to_file = 0
    show = 1
    save = 0
    for i in range(len(numbers_of_clusters)):
        # set directory to store imgs and files
        # path_results = f"./cluster_layout/{maze}_big={big}" \
        #                f"/topology-vs-uniform{numbers_of_clusters}-oop/v4_rp{repetitions}_{e_start}{e_eps}+{q_eps}_mm{mm}_" \
        #                f"ds{ds_factor}_win{win_size}_rep{rep_size}_sg{sg}_{k_means_pkg}_{interpreter}/k[{numbers_of_clusters[i]}]"
        path_results = f"./naive/{maze}_big={big}_irregular_traps" \
                       f"/uniform{numbers_of_clusters}-oop/rp{repetitions}_{e_start}{e_eps}+{q_eps}_mm{mm}_" \
                       f"ds{ds_factor}_win{win_size}_rep{rep_size}_sg{sg}_ng{ng}_{k_means_pkg}_{interpreter}/" \
                       f"k[{numbers_of_clusters[i]}]_trap_inout-2k_equal"
        if not os.path.isdir(path_results):
            makedirs(path_results)
        if print_to_file == 1:
            sys.stdout = open(f"{path_results}/output.txt", 'w')
            sys.stderr = sys.stdout

        plot_maker = PlotMakerNaive(repetitions, std_factor, 1)  # third argument should match num of approaches below

        # ===topology approach===
        # topology_maker = TopologyExpMaker(env_name=maze, big=big, e_mode=e_mode, e_start=e_start, e_eps=e_eps, mm=mm, ds_factor=ds_factor,
        #                                   rep_size=rep_size, win_size=win_size, sg=sg, num_clusters=numbers_of_clusters[i], k_means_pkg=k_means_pkg,
        #                                   q_eps=q_eps, repetitions=repetitions, interpreter=interpreter, print_to_file=print_to_file,
        #                                   plot_maker=plot_maker, path_results=path_results)
        # topology_maker.run(time_comparison=0)

        # ===uniform approach===
        # ---match number of abstract state same with the one in topology approach, in order to be fair.
        env = MazeNaive(maze=maze, big=big)
        a = math.ceil(env.size[0] / np.sqrt(numbers_of_clusters[i]))
        b = math.ceil(env.size[1] / np.sqrt(numbers_of_clusters[i]))
        print("(a,b): ", (a, b))
        uniform_maker = UniformExpMakerNaive(env_name=maze, big=big, tiling_size=(a, b), q_eps=q_eps, repetitions=repetitions,
                                        interpreter=interpreter, print_to_file=print_to_file, plot_maker=plot_maker,
                                        path_results=path_results)
        uniform_maker.run(time_comparison=1)

        # ===general approach===
        # general_maker = GeneralExpMakerNaive(env_name=maze, big=big, e_mode=e_mode, e_start=e_start, e_eps=e_eps, mm=mm, ds_factor=ds_factor, ds_repetitions=ds_repetitions,
        #              rep_size=rep_size, win_size=win_size, sg=sg, num_clusters=numbers_of_clusters[i], k_means_pkg=k_means_pkg, q_eps=q_eps,
        #              repetitions=repetitions, interpreter=interpreter, print_to_file=print_to_file, plot_maker=plot_maker, path_results=path_results)
        # general_maker.run(p_heatmap=1, p_cluster_layout_and_values=1)

        # ===plot and save summary===
        print("saving fig_each_rep ...")
        if show:
            plot_maker.fig_each_rep.show()
        if save:
            plot_maker.fig_each_rep.savefig(f"{path_results}/plots_of_each_rep.png", dpi=300, facecolor='w', edgecolor='w',
                                            orientation='portrait', format=None,
                                            transparent=False, bbox_inches=None, pad_inches=0.1)

        # print("saving fig_mean_performance ...")
        # if show:
        #     plot_maker.fig_mean_performance.show()
        # if save:
        #     plot_maker.fig_mean_performance.savefig(f"{path_results}/mean_results.png",
        #                                             dpi=200, facecolor='w', edgecolor='w', orientation='portrait',
        #                                             format=None, transparent=False, bbox_inches=None, pad_inches=0.1)
        #
        # print("saving fig_time_consumption ...")
        # if show:
        #     plot_maker.fig_time_consumption.show()
        # if save:
        #     plot_maker.fig_time_consumption.savefig(f"{path_results}/time_consumption.png",
        #                                             dpi=200, facecolor='w', edgecolor='w', orientation='portrait',
        #                                             format=None, transparent=False, bbox_inches=None, pad_inches=0.1)

        if print_to_file == 1:
            sys.stdout.close()

if __name__ == "__main__":
    # env = MazeNaive(maze="basic", big=1)
    # mdp = MDP(env)
    # mdp.solve_mdp(synchronous=0, monitor=0)

    env = MazeNaive(maze="simple2", big=0)
    # PlotMakerNaive.plot_maze_trap(env, version=1, save=1)
    PlotMakerNaive.plot_maze(env, version=1, save=0)

    # compare_approaches()