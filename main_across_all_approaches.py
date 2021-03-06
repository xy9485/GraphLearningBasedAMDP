import sys
# print(sys.path)
import copy
import math
import statistics
import random
import pickle
# print(matplotlib.get_backend())
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pandas as pd
from os import makedirs
import numpy as np
import os
# from PIL import Image
import time
from itertools import chain
from collections import Counter
from abstractions.abstraction_all_approaches import AMDP_Topology_Uniform, AMDP_General
from envs.maze_env_general_all_approaches import Maze
from RL_brains.RL_brain_all_approaches import ExploreStateBrain, ExploreCoordBrain, QLambdaBrain
from gensim_operations.gensim_operation_all_approaches import GensimOperator_Topology,GensimOperator_General

class PlotMaker:
    def __init__(self, num_of_repetitions, std_factor, num_approaches):
        self.num_of_repetitions = num_of_repetitions
        self.num_approaches = num_approaches

        self.fig_each_rep, self.axs_each_rep = plt.subplots(num_of_repetitions, 5,
                                                            figsize=(5 * 5, num_of_repetitions * 4))
        self.fig_each_rep.set_tight_layout(True)

        self.fig_mean_performance, self.axs_mean_performance = plt.subplots(1, 3, figsize=(5 * 3, 4 * 1))
        self.fig_mean_performance.set_tight_layout(True)
        self.current_approach_mean_performance = self.num_approaches
        self.std_factor = std_factor
        self.max_steps = 0  # for plotting curve of mean performance: reward against move_count

        self.fig_time_consumption, self.ax_time_consumption = plt.subplots()
        self.fig_time_consumption.set_tight_layout(True)
        self.current_approach_time_consumption = self.num_approaches
        self.highest_bar_height = 0

    # def _initialize_plot_for_each_rep(self, num_of_repetitions):
    #     self.fig_each_rep, self.axs_each_rep = plt.subplots(num_of_repetitions, 5,
    #                                                         figsize=(5 * 5, num_of_repetitions * 4))
    #     self.fig_each_rep.set_tight_layout(True)
    @staticmethod
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
            ax.plot(line[1], line[0], color='k', alpha=1, linewidth=1.5)

    @staticmethod
    def plot_maze(env: Maze, version=1, show=1, save=1):
        fontsize = 12 if env.big == 0 else 4.5
        fontweight = 'semibold'
        cmap = ListedColormap(["black", "lightgrey", "yellow", "green", "red", "tab:cyan"])
        indice_z = np.where(env.room_layout == 'z')
        maze_to_plot = np.where(env.room_layout == 'w', 0, 1)
        for i in range(len(indice_z[0])):
            maze_to_plot[indice_z[0][i], indice_z[1][i]] = 5
        maze_to_plot[env.start_state[0], env.start_state[1]] = 4
        maze_to_plot[env.goal[0], env.goal[1]] = 3
        w, h = figure.figaspect(maze_to_plot)
        print("w, h:", w, h)
        fig1, ax1 = plt.subplots(figsize=(w, h))
        # fig, ax1 = plt.subplots()
        ax1.text(env.start_state[1] + 0.5, env.start_state[0] + 0.55, 'S', ha="center", va="center", color="k", fontsize=fontsize,
                 fontweight=fontweight)
        ax1.text(env.goal[1] + 0.5, env.goal[0] + 0.55, 'G', ha="center", va="center", color="k", fontsize=fontsize,
                 fontweight=fontweight)
        for flag in env.flags:
            # print(flag)
            maze_to_plot[flag[0], flag[1]] = 2
            ax1.text(flag[1] + 0.5, flag[0] + 0.55, 'F', ha="center", va="center", color="k", fontsize=fontsize,
                     fontweight=fontweight)
        # print(maze_to_plot)
        ax1.pcolor(maze_to_plot, cmap=cmap, vmin=0, vmax=5, edgecolors='k', linewidth=1)
        ax1.invert_yaxis()
        ax1.axis('off')
        fig1.tight_layout()
        if show:
            fig1.show()
        if save:
            fig1.savefig(f"./img_mazes/{env.maze_name}_big{env.big}_v{version}_temp.png", dpi=300, facecolor='w', edgecolor='w',
                         orientation='portrait', format=None,
                         transparent=False, bbox_inches=None, pad_inches=0.1)

    @staticmethod
    def plot_maze_trap(env: Maze, version=1, show=1, save=0):
        fontsize = 20 if env.big == 0 else 4.5
        fontweight = 'semibold'
        cmap = ListedColormap(["black", "lightgrey", "yellow", "green", "red", "purple"])
        maze_to_plot = np.where(env.room_layout == 'w', 0, 1)
        maze_to_plot[env.start_state[0], env.start_state[1]] = 4
        maze_to_plot[env.goal[0], env.goal[1]] = 3

        w, h = figure.figaspect(maze_to_plot)
        print("w, h:", w, h)
        fig1, ax1 = plt.subplots(figsize=(12, 12))
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
        ax1.invert_yaxis()
        ax1.axis('off')
        fig1.tight_layout()
        if show:
            fig1.show()
        if save:
            fig1.savefig(f"./img_mazes/{env.maze_name}_big{env.big}_traps_v{version}.png", dpi=200, facecolor='w', edgecolor='w',
                         orientation='portrait', format=None,
                         transparent=False, bbox_inches=None, pad_inches=0.1)

    @staticmethod
    def plot_manual_rooms(env: Maze, version, show=1, save=1, plot_label=1):
        room_layout = copy.deepcopy(env.room_layout).tolist()
        for row in room_layout:
            for index, item in enumerate(row):
                # print(type(row[index]), row[index])
                if row[index].isdigit():
                    row[index] = int(row[index])
                else:
                    row[index] = -1
        vmax = np.amax(room_layout)
        vmin = -vmax * 0.16
        print("vmax:", vmax)
        for row in room_layout:
            for index, item in enumerate(row):
                if row[index] < 0:
                    row[index] = vmin/2
        room_layout = np.array(room_layout)
        w, h = figure.figaspect(room_layout)
        # w = round(w*1.5, 2)
        # h = round(h*1.5, 2)
        print("w, h:", w, h)
        fig, ax = plt.subplots(figsize=(6, 4.8))
        # fig, ax = plt.subplots()
        my_cmap = plt.cm.get_cmap('gist_ncar')
        # my_cmap.set_under('k')
        ax.imshow(room_layout, vmax=vmax, vmin=vmin, aspect='equal', cmap=my_cmap)
        # ax.tick_params(axis='both', labelsize=16)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        PlotMaker.contour_rect_slow(room_layout, ax)
        if plot_label == 1:
            indice_room_centers = []
            # print(np.array(room_layout))
            for i in range(vmax+1):
                coords = np.argwhere(env.room_layout == str(i))
                indice_room_centers.append(np.mean(coords, axis=0))
            for index, cen in enumerate(indice_room_centers):
                # print(index, cen)
                ax.text(cen[1], cen[0], str(index), horizontalalignment='center', verticalalignment='center',
                                 fontsize=14, fontweight='semibold')
        if show:
            fig.show()
        if save:
            fig.savefig(f"./img_manual_room_layout/{env.maze_name}_big{env.big}_v{version}.png", dpi=300, bbox_inches='tight', transparent=False, pad_inches=0.1)

    def plot_each_heatmap(self, agent_e, rep, ax_title):
        # plot heatmap
        im = self.axs_each_rep[rep, 4].imshow(agent_e.states_long_life, aspect='equal', cmap='hot')
        self.fig_each_rep.colorbar(im, ax=self.axs_each_rep[rep, 4])
        self.axs_each_rep[rep, 4].set_title(ax_title)
        # self.fig_each_rep.show()

    def plot_each_heatmap_general(self, agent, rep, path_results, show=1, save=1,
                                  exploration=1, final_policy=0):
        if exploration + final_policy != 1:
            raise Exception("exploration or final_policy, only one of them can be 1")
        if exploration:
            heatmap_mode = "exploration"
        elif final_policy:
            heatmap_mode = "final_policy"
        # fig = plt.figure(figsize=(5 * 3, 4 * 4))
        fig = plt.figure(figsize=(5 * 3, 5 * 4))
        vmin = np.amin(agent.states_long_life)
        vmax = np.amax(agent.states_long_life)
        my_cmap = 'hot'
        asp = 'equal'
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    hm = agent.states_long_life[:, :, k, l, m]
                    print("hm.shape:", hm.shape)
                    if k == 0 and l == 0 and m == 0:
                        ax = fig.add_subplot(4, 3, 11)          # gist_ncar; rainbow
                        im = ax.imshow(hm, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                    elif k == 1 and l == 0 and m == 0:
                        ax = fig.add_subplot(4, 3, 7)
                        im = ax.imshow(hm, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                    elif k == 0 and l == 1 and m == 0:
                        ax = fig.add_subplot(4, 3, 8)
                        im = ax.imshow(hm, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                    elif k == 0 and l == 0 and m == 1:
                        ax = fig.add_subplot(4, 3, 9)
                        im = ax.imshow(hm, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                    elif k == 1 and l == 1 and m == 0:
                        ax = fig.add_subplot(4, 3, 4)
                        im = ax.imshow(hm, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                    elif k == 1 and l == 0 and m == 1:
                        ax = fig.add_subplot(4, 3, 5)
                        im = ax.imshow(hm, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                    elif k == 0 and l == 1 and m == 1:
                        ax = fig.add_subplot(4, 3, 6)
                        im = ax.imshow(hm, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                    elif k == 1 and l == 1 and m == 1:
                        ax = fig.add_subplot(4, 3, 2)
                        im = ax.imshow(hm, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                    ax.set_title(f"{k}-{l}-{m}", fontsize=15, fontweight='semibold')
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                    ax.tick_params(axis='both', which='both', length=0)
        # fig.set_tight_layout(False)
        fig.subplots_adjust(right=0.88)
        cax = fig.add_axes([0.9, 0.23, 0.03, 0.5])
        fig.colorbar(im, cax=cax)
        cax.tick_params(axis='both', labelsize=20)

        # fig.set_tight_layout(True)
        if show:
            fig.show()
        if save:
            os.makedirs(f"{path_results}/building", exist_ok=True)
            fig.savefig(f"{path_results}/building/building_{heatmap_mode}_rep{rep}.png", dpi=200, bbox_inches='tight', transparent=False, pad_inches=0.1)

    def plot_each_cluster_layout(self, gensim_opt, num_clusters, rep, ax_title, plot_label=1):
        copy_cluster_layout = copy.deepcopy(gensim_opt.cluster_layout)
        # vmax = num_clusters + 10
        # vmin = -(num_clusters + 10) / 30
        vmax = num_clusters
        vmin = -vmax * 0.16
        for row in copy_cluster_layout:
            for index, item in enumerate(row):
                if row[index].isdigit():
                    # row[index] = (int(row[index]) + 1) * 10
                    # row[index] = int(row[index]) + 10
                    row[index] = int(row[index])
                else:
                    row[index] = vmin/2
        self.axs_each_rep[rep, 3].imshow(np.array(copy_cluster_layout), vmax=vmax, vmin=vmin,
                                         aspect='auto', cmap="gist_ncar")

        if plot_label == 1:
            indice_center_clusters = []
            np_cluster_layout = np.array(gensim_opt.cluster_layout)
            for i in range(num_clusters):
                coords = np.argwhere(np_cluster_layout == str(i))
                indice_center_clusters.append(np.mean(coords, axis=0))
            for index, cen in enumerate(indice_center_clusters):
                # print(index, cen)
                self.axs_each_rep[rep, 3].text(cen[1], cen[0], str(index), horizontalalignment='center', verticalalignment='center',
                                 fontsize=13, fontweight='semibold')

        self.axs_each_rep[rep, 3].set_title(ax_title)
        # self.fig_each_rep.show()

    def plot_each_cluster_layout_general(self, gensim_opt, num_clusters, env, path_results, rep, plot_label=1, save=1, show=1):
        fig = plt.figure(figsize=(5 * 3, 4 * 4))
        my_cmap = 'gist_ncar'
        asp = 'auto'
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    print(f"k-l-m: {k}-{l}-{m}")
                    plate = []
                    for i in range(env.size[0]):
                        row = []
                        for j in range(env.size[1]):
                            current_state = (i, j, k, l, m)
                            if current_state in env.valid_states:
                                row.append(gensim_opt.dict_gstates_astates[str(current_state)])
                            elif str([i, j]) in env.walls:
                                row.append('W')
                            elif env.flags.index((i, j)) == 0 and k == 0:
                                row.append('X')
                            elif env.flags.index((i, j)) == 1 and l == 0:
                                row.append('X')
                            elif env.flags.index((i, j)) == 2 and m == 0:
                                row.append('X')
                        plate.append(row)
                    list_of_lists = copy.deepcopy(plate)
                    # ==to print==
                    # for row in range(len(list_of_lists)):
                    #     for col in range(len(list_of_lists[row])):
                    #         if len(str(list_of_lists[row][col])) == 1:
                    #             print('  ' + str(list_of_lists[row][col]), end=' ')
                    #         elif len(str(list_of_lists[row][col])) == 2:
                    #             print(' ' + str(list_of_lists[row][col]), end=' ')
                    #         else:
                    #             print(str(list_of_lists[row][col]), end=' ')
                    #     print(' ')  # To change lines
                    # ==to plot==
                    for row in range(len(list_of_lists)):
                        for col in range(len(list_of_lists[0])):
                            if isinstance(list_of_lists[row][col], int):
                                list_of_lists[row][col] += 10
                            else:
                                list_of_lists[row][col] = 0
                    vmax = num_clusters + 10
                    vmin = -(num_clusters + 10) / 30
                    if k == 0 and l == 0 and m == 0:
                        ax = fig.add_subplot(4, 3, 11)          # gist_ncar; rainbow
                        ax.imshow(np.array(list_of_lists), vmax=vmax, vmin=vmin, aspect=asp, cmap=my_cmap)
                        # ax.set_title(f"{k}-{l}-{m}")
                    elif k == 1 and l == 0 and m == 0:
                        ax = fig.add_subplot(4, 3, 7)
                        ax.imshow(np.array(list_of_lists), vmax=vmax, vmin=vmin, aspect=asp, cmap=my_cmap)
                        # ax.set_title(f"{k}-{l}-{m}")
                    elif k == 0 and l == 1 and m == 0:
                        ax = fig.add_subplot(4, 3, 8)
                        ax.imshow(np.array(list_of_lists), vmax=vmax, vmin=vmin, aspect=asp, cmap=my_cmap)
                        # ax.set_title(f"{k}-{l}-{m}")
                    elif k == 0 and l == 0 and m == 1:
                        ax = fig.add_subplot(4, 3, 9)
                        ax.imshow(np.array(list_of_lists), vmax=vmax, vmin=vmin, aspect=asp, cmap=my_cmap)
                        # ax.set_title(f"{k}-{l}-{m}")
                    elif k == 1 and l == 1 and m == 0:
                        ax = fig.add_subplot(4, 3, 4)
                        ax.imshow(np.array(list_of_lists), vmax=vmax, vmin=vmin, aspect=asp, cmap=my_cmap)
                        # ax.set_title(f"{k}-{l}-{m}")
                    elif k == 1 and l == 0 and m == 1:
                        ax = fig.add_subplot(4, 3, 5)
                        ax.imshow(np.array(list_of_lists), vmax=vmax, vmin=vmin, aspect=asp, cmap=my_cmap)
                        # ax.set_title(f"{k}-{l}-{m}")
                    elif k == 0 and l == 1 and m == 1:
                        ax = fig.add_subplot(4, 3, 6)
                        ax.imshow(np.array(list_of_lists), vmax=vmax, vmin=vmin, aspect=asp, cmap=my_cmap)
                        # ax.set_title(f"{k}-{l}-{m}")
                    elif k == 1 and l == 1 and m == 1:
                        ax = fig.add_subplot(4, 3, 2)
                        ax.imshow(np.array(list_of_lists), vmax=vmax, vmin=vmin, aspect=asp, cmap=my_cmap)
                        # ax.set_title(f"{k}-{l}-{m}")
                    if plot_label:
                        np_cluster_layout = np.array(plate)
                        c = 0
                        for i in range(num_clusters):
                            coords = np.argwhere(np_cluster_layout == str(i))
                            if len(coords) > 0:
                                mean = np.mean(coords, axis=0)
                                c += 1
                                # print("mean:", mean)
                                ax.text(mean[1], mean[0], str(i), horizontalalignment='center', verticalalignment='center',
                                        fontsize=13, fontweight='semibold', color='k')
                        ax.set_title(f"{k}-{l}-{m}-c{c}", fontsize=15, fontweight='semibold')
        fig.set_tight_layout(True)
        if show:
            fig.show()
        if save:
            fig.savefig(f"{path_results}/building{rep}.png", dpi=200, facecolor='w', edgecolor='w',)
                                        # orientation='portrait', format=None,
                                        # transparent=False, bbox_inches=None, pad_inches=0.1)

    def plot_each_cluster_layout_t_u_g(self, env, amdp, rep, path_results, plot_label=1, show=1, save=1):
        fig = plt.figure(figsize=(5 * 3, 5 * 4))
        my_cmap = copy.copy(plt.cm.get_cmap('gist_ncar'))
        # my_cmap.set_under('k')
        # my_cmap.set_bad('lime')
        # my_cmap.set_over('dodgerblue')
        # my_cmap = 'gist_ncar'
        asp = 'equal'
        PlotMaker.plates = {}
        if isinstance(amdp.list_of_abstract_states[0], int):
            approach = 'general'
            vmax = len(amdp.list_of_abstract_states)-1
            vmin = -vmax * 0.16
        elif isinstance(amdp.list_of_abstract_states[0], list):
            if amdp.list_of_abstract_states[0][0].isdigit():
                approach = 'topology'
                vmax = (len(amdp.list_of_abstract_states) - 1)/8
                vmin = -vmax * 0.16
            else:
                approach = 'uniform'
                sqrt_=np.sqrt((len(amdp.list_of_abstract_states) - 1) / 8)
                vmax = sqrt_*10 + sqrt_
                vmin = -vmax * 0.16

        print("vmax, vmin:", vmax, vmin)
        # vmin = -vmax*0.16
        # vmin = 0
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    plate = []
                    plate2 = []
                    for i in range(env.size[0]):
                        row = []
                        row2 = []
                        for j in range(env.size[1]):
                            current_state = (i, j, k, l, m)
                            current_coord = (i, j)
                            if current_state in env.valid_states:
                                a_state = amdp.get_abstract_state(current_state)
                                if current_state == env.start_state:
                                    row.append(vmin/2)
                                elif current_coord == env.goal:
                                    row.append(vmin/2)
                                elif approach == 'general':
                                    row.append(a_state)
                                elif approach == 'topology':
                                    row.append(int(a_state[0]))
                                elif approach == 'uniform':
                                    # print(a_state, a_state[0][1], a_state[0][4], type(a_state), type(a_state[0][1]), type(a_state[0][3]))
                                    row.append(int(a_state[0][1])*10+int(a_state[0][4]))
                                row2.append(str(a_state))
                            elif str([i, j]) in env.walls:
                                row.append(vmin/2)
                                row2.append('w')
                            elif env.flags.index((i, j)) == 0 and k == 0:
                                row.append(vmin/2)
                                row2.append('f')
                            elif env.flags.index((i, j)) == 1 and l == 0:
                                row.append(vmin/2)
                                row2.append('f')
                            elif env.flags.index((i, j)) == 2 and m == 0:
                                row.append(vmin/2)
                                row2.append('f')
                        plate.append(row)
                        plate2.append(row2)

                    if k == 0 and l == 0 and m == 0:
                        ax = fig.add_subplot(4, 3, 11)
                        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                        PlotMaker.contour_rect_slow(plate, ax)
                        PlotMaker.plates['000'] = plate
                    elif k == 1 and l == 0 and m == 0:
                        ax = fig.add_subplot(4, 3, 7)
                        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                        PlotMaker.contour_rect_slow(plate, ax)
                        PlotMaker.plates['100'] = plate
                    elif k == 0 and l == 1 and m == 0:
                        ax = fig.add_subplot(4, 3, 8)
                        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                        PlotMaker.contour_rect_slow(plate, ax)
                        PlotMaker.plates['010'] = plate
                    elif k == 0 and l == 0 and m == 1:
                        ax = fig.add_subplot(4, 3, 9)
                        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                        PlotMaker.contour_rect_slow(plate, ax)
                        PlotMaker.plates['001'] = plate
                    elif k == 1 and l == 1 and m == 0:
                        ax = fig.add_subplot(4, 3, 4)
                        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                        PlotMaker.contour_rect_slow(plate, ax)
                        PlotMaker.plates['110'] = plate
                    elif k == 1 and l == 0 and m == 1:
                        ax = fig.add_subplot(4, 3, 5)
                        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                        PlotMaker.contour_rect_slow(plate, ax)
                        PlotMaker.plates['101'] = plate
                    elif k == 0 and l == 1 and m == 1:
                        ax = fig.add_subplot(4, 3, 6)
                        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                        PlotMaker.contour_rect_slow(plate, ax)
                        PlotMaker.plates['011'] = plate
                    elif k == 1 and l == 1 and m == 1:
                        ax = fig.add_subplot(4, 3, 2)
                        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                        PlotMaker.contour_rect_slow(plate, ax)
                        PlotMaker.plates['111'] = plate
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                    ax.tick_params(axis='both', which='both', length=0)
                    if plot_label:
                        np_cluster_layout = np.array(plate2)
                        c = 0
                        for a_state_ in amdp.list_of_abstract_states:
                            coords = np.argwhere(np_cluster_layout == str(a_state_))
                            if len(coords) > 0:
                                if approach == 'general':
                                    a_state_head = a_state_
                                elif approach == 'topology':
                                    a_state_head = a_state_[0]
                                elif approach == 'uniform':
                                    a_state_head = f"{a_state_[0][1]}-{a_state_[0][4]}"
                                mean = np.mean(coords, axis=0)
                                c += 1
                                ax.text(mean[1], mean[0], f"{str(a_state_head)}", horizontalalignment='center', verticalalignment='center',
                                        fontsize=12, fontweight='semibold', color='k')
                    ax.set_title(f"{k}-{l}-{m}-c{c}", fontsize=15, fontweight='semibold')
        # fig.subplots_adjust(right=0.85)
        # cax = fig.add_axes([0.9, 0.23, 0.03, 0.5])
        # fig.colorbar(im, cax=cax)
        if show:
            fig.show()
        if save:
            os.makedirs(f"{path_results}/building_{approach}", exist_ok=True)
            fig.savefig(f"{path_results}/building_{approach}/building_cluster_layout_rep{rep}.png", dpi=200, bbox_inches='tight', transparent=False, pad_inches=0.1)

    def plot_each_amdp_values_t_u_g(self, env, amdp, rep, path_results, approach, plot_label=1, show=1, save=1):
        fig = plt.figure(figsize=(5 * 3, 5 * 4))
        my_cmap = copy.copy(plt.cm.get_cmap('hot'))
        # vmax = np.amax(amdp.values_of_abstract_states)
        # vmin = -0.1 * vmax
        vmax = np.amax(amdp.values_of_abstract_states[:-1])
        values_min = np.min(amdp.values_of_abstract_states[:-1])
        vmin = values_min - 0.07 * (vmax - values_min)
        # vmin = 0
        my_cmap.set_under('grey')
        my_cmap.set_bad('lime')
        my_cmap.set_over('dodgerblue')
        asp = 'equal'
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    plate = []
                    plate2 = []
                    for i in range(env.size[0]):
                        row = []
                        row2 = []
                        for j in range(env.size[1]):
                            current_coord = (i, j)
                            current_state = (i, j, k, l, m)
                            if current_state in env.valid_states:
                                a_state = amdp.get_abstract_state(current_state)
                                if current_state == env.start_state:
                                    row.append(vmax)
                                    row2.append(str(a_state))
                                elif current_coord == env.goal:
                                    row.append(vmax+1)
                                    row2.append(str(a_state))
                                else:
                                    v = amdp.get_value(a_state)
                                    row.append(v)
                                    row2.append(str(a_state))
                            elif str([i, j]) in env.walls:
                                row.append(vmin)
                                row2.append('w')
                            elif env.flags.index((i, j)) == 0 and k == 0:
                                row.append(np.nan)
                                row2.append('f')
                            elif env.flags.index((i, j)) == 1 and l == 0:
                                row.append(np.nan)
                                row2.append('f')
                            elif env.flags.index((i, j)) == 2 and m == 0:
                                row.append(np.nan)
                                row2.append('f')

                        plate.append(row)
                        plate2.append(row2)
                    if k == 0 and l == 0 and m == 0:
                        ax = fig.add_subplot(4, 3, 11)
                        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                        PlotMaker.contour_rect_slow(PlotMaker.plates['000'], ax)
                    elif k == 1 and l == 0 and m == 0:
                        ax = fig.add_subplot(4, 3, 7)
                        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                        PlotMaker.contour_rect_slow(PlotMaker.plates['100'], ax)
                    elif k == 0 and l == 1 and m == 0:
                        ax = fig.add_subplot(4, 3, 8)
                        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                        PlotMaker.contour_rect_slow(PlotMaker.plates['010'], ax)
                    elif k == 0 and l == 0 and m == 1:
                        ax = fig.add_subplot(4, 3, 9)
                        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                        PlotMaker.contour_rect_slow(PlotMaker.plates['001'], ax)
                    elif k == 1 and l == 1 and m == 0:
                        ax = fig.add_subplot(4, 3, 4)
                        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                        PlotMaker.contour_rect_slow(PlotMaker.plates['110'], ax)
                    elif k == 1 and l == 0 and m == 1:
                        ax = fig.add_subplot(4, 3, 5)
                        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                        PlotMaker.contour_rect_slow(PlotMaker.plates['101'], ax)
                    elif k == 0 and l == 1 and m == 1:
                        ax = fig.add_subplot(4, 3, 6)
                        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                        PlotMaker.contour_rect_slow(PlotMaker.plates['011'], ax)
                    elif k == 1 and l == 1 and m == 1:
                        ax = fig.add_subplot(4, 3, 2)
                        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                        PlotMaker.contour_rect_slow(PlotMaker.plates['111'], ax)
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                    ax.tick_params(axis='both', which='both', length=0)
                    if plot_label:
                        np_cluster_layout = np.array(plate2)
                        c = 0
                        for a_state_ in amdp.list_of_abstract_states:
                            coords = np.argwhere(np_cluster_layout == str(a_state_))
                            if len(coords) > 0:
                                if isinstance(a_state_, int):
                                    a_state_head = a_state_
                                elif isinstance(a_state_, list):
                                    if a_state_[0].isdigit():
                                        a_state_head = a_state_[0]
                                    else:
                                        a_state_head = f"{a_state_[0][1]}-{a_state_[0][4]}"
                                mean = np.mean(coords, axis=0)
                                c += 1
                                v_ = round(amdp.get_value(a_state_), 1)
                                # ax.text(mean[1], mean[0], f"{str(a_state_head)}", horizontalalignment='center', verticalalignment='center',
                                #         fontsize=9, fontweight='semibold', color='k')
                                ax.text(mean[1], mean[0], str(v_), horizontalalignment='center', verticalalignment='center',
                                        fontsize=10, fontweight='semibold', color='k')
                    ax.set_title(f"{k}-{l}-{m}", fontsize=15, fontweight='semibold')
        fig.subplots_adjust(right=0.88)
        cax = fig.add_axes([0.9, 0.23, 0.03, 0.5])
        fig.colorbar(im, cax=cax)
        cax.tick_params(axis='both', labelsize=20)
        if show:
            fig.show()
        if save:
            os.makedirs(f"{path_results}/building_{approach}", exist_ok=True)
            fig.savefig(f"{path_results}/building_{approach}/building_solved_values_rep{rep}.png", dpi=200, bbox_inches='tight', transparent=False, pad_inches=0.1)

    def plot_each_flag_reward_movecount(self, flags_episodes, reward_episodes, move_count_episodes, rep, curve_label):
        rolling_window_size = int(len(flags_episodes)/30)
        if curve_label.startswith('t'):
            ls = '-'
        elif curve_label.startswith('U'):
            ls = '--'
        elif curve_label.startswith('T'):
            ls = '-'

        d1 = pd.Series(flags_episodes)
        print("flags_list_episodes.shape:", np.array(flags_episodes).shape)
        rolled_d = pd.Series.rolling(d1, window=rolling_window_size, center=False).mean()
        print('type of movAv:', type(rolled_d))

        self.axs_each_rep[rep, 0].plot(np.arange(len(rolled_d)), rolled_d, linestyle=ls, label=curve_label)
        self.axs_each_rep[rep, 0].set_ylabel("Number of Flags")
        self.axs_each_rep[rep, 0].set_xlabel("Episode No.")
        self.axs_each_rep[rep, 0].set_title(f"flag curve of rep{rep}")
        self.axs_each_rep[rep, 0].legend(loc=4)
        self.axs_each_rep[rep, 0].grid(True)
        # self.axs_each_rep[rep, 0].axvspan(0, num_explore_episodes, facecolor='green', alpha=0.5 / num_of_repetitions)
        # self.axs_each_rep[rep, 0].axvspan(num_explore_episodes, second_evolution, facecolor='blue',alpha=0.5/num_of_repetitions)
        self.axs_each_rep[rep, 0].axis([0, None, 0, 3.5])

        d1 = pd.Series(reward_episodes)
        rolled_d1 = pd.Series.rolling(d1, window=rolling_window_size, center=False).mean()
        self.axs_each_rep[rep, 1].plot(np.arange(len(rolled_d1)), rolled_d1, alpha=1, linestyle=ls, label=curve_label)
        self.axs_each_rep[rep, 1].set_ylabel("reward")
        self.axs_each_rep[rep, 1].set_xlabel("Episode No.")
        self.axs_each_rep[rep, 1].set_title(f"reward curve of rep{rep}")
        self.axs_each_rep[rep, 1].legend(loc=4)
        self.axs_each_rep[rep, 1].grid(True)
        self.axs_each_rep[rep, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        # axs[rep, 1].axvspan(0, num_explore_episodes, facecolor='green', alpha=0.5 / num_of_repetitions)
        # axs[rep, 1].axvspan(num_explore_episodes, second_evolution, facecolor='blue',alpha=0.5/num_of_repetitions)
        self.axs_each_rep[rep, 1].axis([0, None, None, 35000])

        d1 = pd.Series(move_count_episodes)
        rolled_d1 = pd.Series.rolling(d1, window=rolling_window_size, center=False).mean()
        self.axs_each_rep[rep, 2].plot(np.arange(len(rolled_d1)), rolled_d1, alpha=1, linestyle=ls, label=curve_label)
        self.axs_each_rep[rep, 2].set_ylabel("move_count")
        self.axs_each_rep[rep, 2].set_xlabel("Episode No.")
        self.axs_each_rep[rep, 2].set_title(f"move_count curve of rep{rep}")
        self.axs_each_rep[rep, 2].legend(loc=1)
        self.axs_each_rep[rep, 2].grid(True)
        self.axs_each_rep[rep, 2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        # axs[rep, 2].axvspan(0, num_explore_episodes, facecolor='green', alpha=0.5 / num_of_repetitions)
        # axs[rep, 2].axvspan(num_explore_episodes, second_evolution, facecolor='blue', alpha=0.5/num_of_repetitions)
        self.axs_each_rep[rep, 2].axis([0, None, None, None])

        # self.fig_each_rep.show()

    def plot_mean_performance_across_reps(self, flags_episodes_repetitions, reward_episodes_repetitions,
                                          move_count_episodes_repetitions,
                                          curve_label, ax_title=None):
        self.current_approach_mean_performance -= 1
        rolling_window_size = int(len(flags_episodes_repetitions[0])/30)
        if curve_label.startswith('t'):
            ls = '-'
        elif curve_label.startswith('U'):
            ls = '-'
        elif curve_label.startswith('T'):
            ls = '-'
        else:
            ls = '-'

        fs = 17
        fs2 = 13
        print("============Flags plotting============")
        mean_by_rep_flags = np.mean(flags_episodes_repetitions, axis=0)
        std_by_rep_flags = np.std(flags_episodes_repetitions, axis=0)
        print("mean_by_rep_flags.shape", mean_by_rep_flags.shape)
        print("std_by_rep_flags.shape", std_by_rep_flags.shape)
        # plot_errors = std_by_rep_flags / np.sqrt(10)
        # plot_errors = std_by_rep_flags * 2
        confidence_interval = std_by_rep_flags * self.std_factor
        # plt.rcParams['agg.path.chunksize'] = 10000
        d = pd.Series(mean_by_rep_flags)
        s = pd.Series(confidence_interval)
        rolled_d = pd.Series.rolling(d, window=rolling_window_size, center=False).mean()
        rolled_s = pd.Series.rolling(s, window=rolling_window_size, center=False).mean()
        self.axs_mean_performance[0].plot(np.arange(len(rolled_d)), rolled_d, linestyle=ls, label=curve_label)
        self.axs_mean_performance[0].fill_between(np.arange(len(rolled_d)), rolled_d - rolled_s, rolled_d + rolled_s, alpha=0.25)
        if self.current_approach_mean_performance == 0:
            self.axs_mean_performance[0].set_ylabel("Flags", fontsize=fs)
            self.axs_mean_performance[0].set_xlabel("Episode", fontsize=fs)
            # self.axs_mean_performance[0].legend(loc=4)
            self.axs_mean_performance[0].grid(True)
            # self.axs_mean_performance[0].set_title(ax_title)
            # self.axs_mean_performance[0].axvspan(0, num_explore_episodes, facecolor='green', alpha=0.5)
            # axs[0].axvspan(num_explore_episodes,second_evolution, facecolor='blue', alpha=0.5)
            self.axs_mean_performance[0].axis([0, None, 0, 3.5])
            self.axs_mean_performance[0].tick_params(axis='both', labelsize=fs2)
            self.axs_mean_performance[0].xaxis.get_offset_text().set_fontsize('large')
            self.axs_mean_performance[0].yaxis.get_offset_text().set_fontsize('large')

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
        self.axs_mean_performance[1].plot(np.arange(len(rolled_d)), rolled_d, linestyle=ls, label=curve_label)
        self.axs_mean_performance[1].fill_between(np.arange(len(rolled_d)), rolled_d - rolled_s, rolled_d + rolled_s, alpha=0.25)
        if self.current_approach_mean_performance == 0:
            self.axs_mean_performance[1].set_ylabel("Reward", fontsize=fs)
            self.axs_mean_performance[1].set_xlabel("Episode", fontsize=fs)
            # self.axs_mean_performance[1].legend(loc=4)
            self.axs_mean_performance[1].grid(True)
            self.axs_mean_performance[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            # axs[0].set_title(ax_title)
            # axs[0].axvspan(0, num_explore_episodes, facecolor='green', alpha=0.5)
            # axs[1].axvspan(num_explore_episodes, second_evolution, facecolor='blue', alpha=0.5)
            # axs[1].set(xlim=(0, num_total_episodes))
            self.axs_mean_performance[1].axis([0, None, None, 35000])
            self.axs_mean_performance[1].tick_params(axis='both', labelsize=fs2)
            self.axs_mean_performance[1].xaxis.get_offset_text().set_fontsize('large')
            self.axs_mean_performance[1].yaxis.get_offset_text().set_fontsize('large')

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
        # rolled_p = p
        rolled_s = pd.Series.rolling(s, window=rolling_window_size, center=False).mean()
        self.axs_mean_performance[2].plot(rolled_p, rolled_d, linestyle=ls, label=curve_label)
        self.axs_mean_performance[2].fill_between(rolled_p, rolled_d - rolled_s, rolled_d + rolled_s, alpha=0.25)
        if rolled_p.max() > self.max_steps:
            self.max_steps = rolled_p.max()
        if self.current_approach_mean_performance == 0:
            self.axs_mean_performance[2].set_ylabel("Reward", fontsize=fs)
            self.axs_mean_performance[2].set_xlabel("Steps", fontsize=fs)
            self.axs_mean_performance[2].legend(loc=4, fontsize=15)
            self.axs_mean_performance[2].grid(True)
            self.axs_mean_performance[2].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
            # axs[1].set_title(f"reward against steps over {'big' if env.big==1 else 'small'} {env.maze_name}")
            # axs[3].axvspan(0, num_explore_episodes, facecolor='green', alpha=0.5)
            # axs[3].axvspan(num_explore_episodes, second_evolution, facecolor='blue', alpha=0.5)
            # axs[1].set(xlim=(0, num_total_episodes))
            self.axs_mean_performance[2].axis([0, self.max_steps * 1.05, None, 35000])
            self.axs_mean_performance[2].tick_params(axis='both', labelsize=fs2)
            self.axs_mean_performance[2].xaxis.get_offset_text().set_fontsize('large')
            self.axs_mean_performance[2].yaxis.get_offset_text().set_fontsize('large')

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
                             ha='center', va='bottom', fontsize=fontsize, fontweight='semibold')

        mean_by_rep_experiment_time = np.mean(experiment_time_repetitions)
        mean_by_rep_exploration_time = np.mean(exploration_time_repetitions)
        mean_by_rep_word2vec_time = np.mean(solve_word2vec_time_repetitions)
        mean_by_rep_kmeans_time = np.mean(solve_kmeans_time_repetitions)
        mean_by_rep_amdp_time = np.mean(solve_amdp_time_repetitions)
        mean_by_rep_q_time = np.mean(ground_learning_time_repetitions)
        labels = ['Total', 'Explore', 'Skip-gram', 'Cluster', 'Solve AMDP', r'Q($\lambda$)']
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

        # autolabel(rects, fontsize)
        if data[0] > self.highest_bar_height:
            self.highest_bar_height = data[0]
        self.current_approach_time_consumption -= 1

        if self.current_approach_time_consumption == 0:
            self.ax_time_consumption.set_xticks(x)
            self.ax_time_consumption.set_xticklabels(labels)
            self.ax_time_consumption.set_ylabel("Time in sec", fontsize=15)
            # self.ax_time_consumption.set_xlabel(fontsize=16)
            self.ax_time_consumption.legend(loc=9, fontsize=13)
            self.ax_time_consumption.set_ylim(top=self.highest_bar_height * 1.1)
            self.ax_time_consumption.tick_params(axis='y', labelsize=13)
            self.ax_time_consumption.tick_params(axis='x', labelsize=12)


class ExperimentMaker:
    def __init__(self, env_name, big, q_eps, interpreter, print_to_file, plot_maker: PlotMaker, stochasticity=None):
        self.env_name = env_name
        self.big = big
        self.stochasticity = stochasticity

        self.interpreter = interpreter
        self.print_to_file = print_to_file
        self.env = Maze(maze=self.env_name, big=self.big, stochasticity=self.stochasticity)
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

    def plot_maze(self):
        env = self.env
        fontsize = 12 if self.big == 0 else 4.5
        fontweight = 'semibold'
        cmap = ListedColormap(["black", "lightgrey", "yellow", "green", "red"])
        maze_to_plot = np.where(env.room_layout == 'w', 0, 1)
        maze_to_plot[env.state[0], env.state[1]] = 4
        maze_to_plot[env.goal[0], env.goal[1]] = 3
        w, h = figure.figaspect(maze_to_plot)
        print("w,h:", w, h)
        fig1, ax1 = plt.subplots(figsize=(w, h))
        # fig, ax1 = plt.subplots()
        ax1.text(env.state[1] + 0.5, env.state[0] + 0.55, 'S', ha="center", va="center", color="k", fontsize=fontsize,
                 fontweight=fontweight)
        ax1.text(env.goal[1] + 0.5, env.goal[0] + 0.55, 'G', ha="center", va="center", color="k", fontsize=fontsize,
                 fontweight=fontweight)
        for flag in env.flags:
            # print(flag)
            maze_to_plot[flag[0], flag[1]] = 2
            ax1.text(flag[1] + 0.5, flag[0] + 0.55, 'F', ha="center", va="center", color="k", fontsize=fontsize,
                     fontweight=fontweight)
        # print(maze_to_plot)
        ax1.pcolor(maze_to_plot, cmap=cmap, vmin=0, vmax=4, edgecolors='k', linewidth=1)
        ax1.invert_yaxis()
        ax1.axis('off')
        fig1.tight_layout()
        fig1.show()
        fig1.savefig(f"./img_mazes/{env.maze_name}_big{self.big}.png", dpi=600, facecolor='w', edgecolor='w',
                     orientation='portrait', format=None,
                     transparent=False, bbox_inches=None, pad_inches=0.1)

    def _build_and_solve_amdp(self, tiling_size: tuple = None, gensim_opt = None, general=0):
        print("-----Begin build and solve amdp-----")
        assert (tiling_size == None) != (gensim_opt == None), "tiling_size and gensim_opt can't be both None"
        if tiling_size:
            amdp = AMDP_Topology_Uniform(env=self.env, uniform_mode=tiling_size, gensim_opt=None)
        elif gensim_opt and general == 0:
            amdp = AMDP_Topology_Uniform(env=self.env, uniform_mode=None, gensim_opt=gensim_opt)
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
            agent_q.reset_episodic_staff()
            episode_reward = 0
            move_count = 0
            track = [str((env.state[0], env.state[1]))]
            a = agent_q.policy(env.state, env.actions(env.state))
            while not env.isTerminal(env.state):
                agent_q.states_episodic[env.state[0], env.state[1], env.state[2], env.state[3], env.state[4]] += 1
                if ep > self.ground_learning_config['q_eps']-3:  # for visualizing the finial policy
                    agent_q.states_long_life[env.state[0], env.state[1], env.state[2], env.state[3], env.state[4]] += 1
                abstract_state = amdp.get_abstract_state(env.state)
                new_state = env.step(env.state, a)
                move_count += 1
                # if move_count % 10000 == 0:
                #     print(f"move_count in current episode {ep}:", move_count)
                env.update_walls(move_count=sum(self.move_count_episodes))  # to comment when the env is not stochastic
                available_states = env.actions(new_state)
                if len(available_states) > 0:
                    track.append(str((new_state[0], new_state[1])))
                    new_abstract_state = amdp.get_abstract_state(new_state)
                    a_prime = agent_q.policy(new_state, available_states)  ##Greedy next-action selected
                    a_star = agent_q.policyNoRand(new_state, available_states)
                    r = env.reward(env.state, a, new_state)  ## ground level reward
                    episode_reward += r

                    value_new_abstract_state = amdp.get_value(new_abstract_state)
                    value_abstract_state = amdp.get_value(abstract_state)
                    shaping = self.ground_learning_config['gamma'] * value_new_abstract_state * \
                              self.ground_learning_config['omega'] - value_abstract_state * self.ground_learning_config['omega']
                    # shaping = (value_new_abstract_state - value_abstract_state) * self.ground_learning_config['omega']
                    # shaping = 0
                    agent_q.learn(env.state, a, new_state, a_prime, a_star, r + shaping)

                    env.state = new_state
                    a = a_prime
                else:
                    track.append(str(env.state))

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

    def _pickler(self, approach, granu):
        os.makedirs(f"{self.path_results}/performance/{approach}", exist_ok=True)
        pf = f"{self.path_results}/performance/{approach}"
        with open(f"{pf}/k{granu}_flags_eps_reps.pkl", 'wb') as f:
            pickle.dump(self.flags_episodes_repetitions, f)
        with open(f"{pf}/k{granu}_rewards_eps_reps.pkl", 'wb') as f:
            pickle.dump(self.reward_episodes_repetitions, f)
        with open(f"{pf}/k{granu}_mc_eps_reps.pkl", 'wb') as f:
            pickle.dump(self.move_count_episodes_repetitions, f)

        os.makedirs(f"{self.path_results}/times/{approach}", exist_ok=True)
        tm = f"{self.path_results}/times/{approach}"
        with open(f"{tm}/k{granu}_experiment.pkl", 'wb') as f:
            pickle.dump(self.experiment_time_repetitions, f)
        with open(f"{tm}/k{granu}_solve_amdp.pkl", 'wb') as f:
            pickle.dump(self.solve_amdp_time_repetitions, f)
        with open(f"{tm}/k{granu}_ground_learning.pkl", 'wb') as f:
            pickle.dump(self.ground_learning_time_repetitions, f)
        if approach == 'general' or approach == 'topology':
            with open(f"{tm}/k{granu}_exploration.pkl", 'wb') as f:
                pickle.dump(self.exploration_time_repetitions, f)
            with open(f"{tm}/k{granu}_solve_w2v.pkl", 'wb') as f:
                pickle.dump(self.solve_word2vec_time_repetitions, f)
            with open(f"{tm}/k{granu}_solve_kmeans.pkl", 'wb') as f:
                pickle.dump(self.solve_kmeans_time_repetitions, f)

class TopologyExpMaker(ExperimentMaker):
    def __init__(self, env_name: str, big: int, e_mode: str, e_start: str, e_eps: int, mm: int, ds_factor,
                 rep_size: int, win_size: int, sg: int, num_clusters: int, k_means_pkg: str, q_eps: int,
                 repetitions: int, interpreter: str, print_to_file: int, plot_maker: PlotMaker, path_results: str):

        super().__init__(env_name, big, q_eps, interpreter, print_to_file, plot_maker)

        self.explore_config = {
            'e_mode': e_mode,
            'e_start': e_start,
            'e_eps': e_eps,
            'max_move_count': mm,
            'ds_factor': ds_factor,
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
        print("+++++++++++start TopologyExpMaker.run()+++++++++++")
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
        agent_e = ExploreCoordBrain(env=self.env, explore_config=self.explore_config)
        env = self.env
        valid_coords_ = tuple(env.valid_coords)
        for ep in range(self.explore_config['e_eps']):
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
                start_coord = random.choice(valid_coords_)
                start_state = (start_coord[0], start_coord[1], 0, 0, 0)
                env.state = start_state
            elif self.explore_config['e_start'] == 'last':
                start_state = env.state
                # print(start_state)
                env.reset()
                env.state = start_state
            else:
                raise Exception("Invalid self.explore_config['e_start']")

            agent_e.reset_episodic_staff()

            if self.explore_config['e_mode'] == 'sarsa':
                track = [str((env.state[0], env.state[1]))]
                a = agent_e.policy_explore_rl(env.state, env.actions(env.state))
                while move_count < self.explore_config['max_move_count']:
                    agent_e.states_episodic[env.state[0], env.state[1]] += 1
                    agent_e.states_long_life[env.state[0], env.state[1]] += 1
                    new_state = env.step(env.state, a)
                    move_count += 1
                    track.append(str((new_state[0], new_state[1])))

                    r1 = -agent_e.states_long_life[new_state[0], new_state[1]]
                    # r2 = -agent.states_episodic[new_state[0], new_state[1]]
                    # beta = ep / num_explore_episodes
                    # r = (1 - beta) * r1 + (beta) * r2
                    # r = (1 - beta) * r2 + (beta) * r1
                    # r2 = env.reward(env.state, a, new_state)
                    # if r2 < -1:
                    #     r = r1 * 10 + r2
                    #     # print("trap, r:", r)
                    # else:
                    #     r = r1
                    #     r *= 10
                    r = r1
                    r *= 10
                    a_prime = agent_e.policy_explore_rl(new_state, env.actions(new_state))
                    # a_star = agent_e.policyNoRand_explore_rl(new_state, env.actions(new_state))
                    agent_e.learn_explore_sarsa(env.state, a, new_state, a_prime, r)
                    env.state = new_state
                    a = a_prime
            elif self.explore_config['e_mode'] == 'softmax':
                track = [str((env.state[0], env.state[1]))]
                a = agent_e.policy_explore_softmax(env.state, env.actions(env.state))
                while move_count < self.explore_config['max_move_count']:
                    agent_e.state_actions_long_life[env.state[0], env.state[1], a] -= 1
                    new_state = env.step(env.state, a)
                    move_count += 1
                    track.append(str((new_state[0], new_state[1])))
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
            else:
                for _ in range(1):
                    down_sampled = [track[index] for index in sorted(random.sample(range(len(track)),
                                    math.floor(len(track) * self.explore_config['ds_factor'])))]
                    self.sentences_period.append(down_sampled)

        end_exploration = time.time()
        exploration_time = end_exploration - start_exploration
        print("exploration_time:", exploration_time)
        self.exploration_time_repetitions.append(exploration_time)

        # print("np.std(agent.states_episodic):", np.std(agent.states_episodic[agent.states_episodic > 0]))
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

    def _w2v_and_kmeans(self, rep):
        print("-----Begin w2v and k-means-----")
        random.shuffle(self.sentences_collected)
        gensim_opt = GensimOperator_Topology(self.env)
        solve_wor2vec_time, solve_kmeans_time = gensim_opt.get_cluster_layout(sentences=self.sentences_collected,
                                      size=self.w2v_config['rep_size'],
                                      window=self.w2v_config['win_size'],
                                      clusters=self.num_clusters,
                                      skip_gram=self.w2v_config['sg'],
                                      workers=self.w2v_config['workers'],
                                      package=self.k_means_pkg)
        self.solve_word2vec_time_repetitions.append(solve_wor2vec_time)
        self.solve_kmeans_time_repetitions.append(solve_kmeans_time)

        # fpath_cluster_layout = self.path_results + f"/rep{rep}_s{self.w2v_config['rep_size']}_w{self.w2v_config['win_size']}" \
        #                                            f"_kmeans{self.num_clusters}_{self.k_means_pkg}.cluster"
        # gensim_opt.write_cluster_layout(fpath_cluster_layout)

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
        abstraction_mode = 'topology'
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
        print("topology approach uploaded to google sheet")
        print(" FINISHED!")

    def run(self, heatmap=1, cluster_layout=1, time_comparison=0):
        if not os.path.isdir(self.path_results):
            makedirs(self.path_results)
        self._print_before_start()
        curve_label = f"T-{self.num_clusters}"
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
            self.sentences_period = []

            # exploration
            agent_e = self._explore()

            # plot heatmap
            if heatmap:
                ax_title = f"{self.env.maze_name}_big{self.env.big}/emode:{self.explore_config['e_mode']}/estart:{self.explore_config['e_start']}"
                self.plot_maker.plot_each_heatmap(agent_e, rep, ax_title)

            # solve w2v and k-means to get clusters and save cluster file
            gensim_opt = self._w2v_and_kmeans(rep)

            # plot cluster layout
            if cluster_layout:
                ax_title = f"clusters{self.num_clusters}mm{self.explore_config['max_move_count']}s" \
                           f"{self.w2v_config['rep_size']}w{self.w2v_config['win_size']}sg{self.w2v_config['sg']}"
                self.plot_maker.plot_each_cluster_layout(gensim_opt, self.num_clusters, rep, ax_title, plot_label=1)

            # build and solve amdp
            amdp = AMDP_Topology_Uniform(env=self.env, uniform_mode=None, gensim_opt=gensim_opt)
            # self.plot_maker.plot_each_cluster_layout_t_u_g(self.env, amdp, rep, self.path_results, save=0)
            self._solve_amdp(amdp)
            # self.plot_maker.plot_each_amdp_values_t_u_g(self.env, amdp, rep, self.path_results, save=0)

            # ground learning
            self._ground_learning(amdp)

            # experiment timing ends and saved
            end_experiment = time.time()
            experiment_time = end_experiment-start_experiment
            self.experiment_time_repetitions.append(experiment_time)

            # plot flags, reward, move_count for each rep
            self.plot_maker.plot_each_flag_reward_movecount(self.flags_episodes[self.explore_config['e_eps']:],
                                                            self.reward_episodes[self.explore_config['e_eps']:],
                                                            self.move_count_episodes[self.explore_config['e_eps']:],
                                                            rep, curve_label)
            # self.plot_maker.fig_each_rep.show()
            # save performance of each rep
            self.flags_episodes_repetitions.append(self.flags_episodes)
            self.reward_episodes_repetitions.append(self.reward_episodes)
            self.move_count_episodes_repetitions.append(self.move_count_episodes)

        # plot mean performance among all reps
        ### ax_title = f"flags collection in {'big' if self.big==1 else 'small'} {self.env.maze_name}"
        # self._pickler(approach='topology', granu=f"{self.num_clusters}")

        sliced_f_ep_rep = np.array(self.flags_episodes_repetitions)[:, self.explore_config['e_eps']:]
        sliced_r_ep_rep = np.array(self.reward_episodes_repetitions)[:, self.explore_config['e_eps']:]
        sliced_m_ep_rep = np.array(self.move_count_episodes_repetitions)[:, self.explore_config['e_eps']:]
        self.plot_maker.plot_mean_performance_across_reps(sliced_f_ep_rep, sliced_r_ep_rep, sliced_m_ep_rep, curve_label)

        if time_comparison:
            self.plot_maker.plot_mean_time_comparison(self.experiment_time_repetitions, self.solve_amdp_time_repetitions,
                                                      self.ground_learning_time_repetitions, self.exploration_time_repetitions,
                                                      self.solve_word2vec_time_repetitions, self.solve_kmeans_time_repetitions,
                                                      bar_label=curve_label)

        # self._results_upload()

class UniformExpMaker(ExperimentMaker):
    def __init__(self, env_name: str, big: int, stochasticity: dict, tiling_size: tuple, q_eps: int, repetitions: int, interpreter: str,
                 print_to_file: int, plot_maker: PlotMaker, path_results: str):

        super().__init__(env_name, big, q_eps, interpreter, print_to_file, plot_maker, stochasticity)

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

    def run(self, time_comparison=0, final_policy=1):
        if not os.path.isdir(self.path_results):
            makedirs(self.path_results)
        self._print_before_start()
        # curve_label = f"u-{self.tiling_size[0]}x{self.tiling_size[1]}"
        curve_label = f"U-{math.ceil(self.env.size[0]/self.tiling_size[0]) * math.ceil(self.env.size[1]/self.tiling_size[1])}"
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
            amdp = AMDP_Topology_Uniform(env=self.env, uniform_mode=self.tiling_size, gensim_opt=None)
            # self.plot_maker.plot_each_cluster_layout_t_u_g(self.env, amdp, rep, self.path_results, save=1)
            self._solve_amdp(amdp)
            # self.plot_maker.plot_each_amdp_values_t_u_g(self.env, amdp, rep, self.path_results, 'uniform', save=0)

            # ground learning
            agent_g = self._ground_learning(amdp)

            # experiment timing ends and saved
            end_experiment = time.time()
            experiment_time = end_experiment-start_experiment
            self.experiment_time_repetitions.append(experiment_time)

            # To visualize the final policy
            if final_policy:
                value4walls = math.ceil(np.amax(agent_g.states_long_life) / 10)
                for coord in np.argwhere(self.env.room_layout == "w"):
                    agent_g.states_long_life[coord[0], coord[1], :, :, :] = -value4walls
                self.plot_maker.plot_each_heatmap_general(agent_g, rep, self.path_results, show=1, save=0)

            # plot flags, reward, move_count for each rep
            self.plot_maker.plot_each_flag_reward_movecount(self.flags_episodes,
                                                            self.reward_episodes,
                                                            self.move_count_episodes,
                                                            rep, curve_label)

            # save performance of each rep
            self.flags_episodes_repetitions.append(self.flags_episodes)
            self.reward_episodes_repetitions.append(self.reward_episodes)
            self.move_count_episodes_repetitions.append(self.move_count_episodes)

        # plot mean performance among all reps
        # ax_title = f"flags collection in {'big' if self.big == 1 else 'small'} {self.env.maze_name}"
        self._pickler(approach='uniform', granu=f"{self.tiling_size}")

        self.plot_maker.plot_mean_performance_across_reps(self.flags_episodes_repetitions,
                                                          self.reward_episodes_repetitions,
                                                          self.move_count_episodes_repetitions, curve_label)
        if time_comparison:
            self.plot_maker.plot_mean_time_comparison(self.experiment_time_repetitions, self.solve_amdp_time_repetitions,
                                                      self.ground_learning_time_repetitions, bar_label=curve_label)

        # self._results_upload()


class GeneralExpMaker(ExperimentMaker):
    def __init__(self, env_name: str, big: int, stochasticity: dict, e_mode: str, e_start: str, e_eps: int, mm: int, ds_factor,
                 rep_size: int, win_size: int, sg: int, num_clusters: int, k_means_pkg: str, q_eps: int,
                 repetitions: int, interpreter: str, print_to_file: int, plot_maker: PlotMaker, path_results: str):

        super().__init__(env_name, big, q_eps, interpreter, print_to_file, plot_maker, stochasticity)

        self.explore_config = {
            'e_mode': e_mode,
            'e_start': e_start,
            'e_eps': e_eps,
            'max_move_count': mm,
            'ds_factor': ds_factor,
            'lr': 0.1,
            'gamma': 0.999,
            'epsilon_e': 0.01,
        }

        self.w2v_config = {
            'rep_size': rep_size,
            'win_size': win_size,
            'sg': sg,
            'workers': 2056
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
        print(f"=maze=:{self.env_name} | big:{self.big} | repetitions:{self.repetitions} | interpreter:{self.interpreter} |"
              f"print_to_file: {self.print_to_file}")
        print(f"maze_stochasticity:", self.stochasticity)
        print(f"=explore_config=: {self.explore_config}")
        print(f"=w2v_config=: {self.w2v_config}")
        print(f"=ground_learning_config=: {self.ground_learning_config}")
        print(f"num_clusters: {self.num_clusters} | k_means_pkg: {self.k_means_pkg}")

    def _explore(self):
        print("-----Begin Exploration-----")
        start_exploration = time.time()
        agent_e = ExploreStateBrain(env=self.env, explore_config=self.explore_config)
        env = self.env
        valid_states_ = tuple(env.valid_states)
        for ep in range(self.explore_config['e_eps']):
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
                    agent_e.states_episodic[env.state[0], env.state[1], env.state[2], env.state[3], env.state[4]] += 1
                    agent_e.states_long_life[env.state[0], env.state[1], env.state[2], env.state[3], env.state[4]] += 1
                    new_state = env.step(env.state, a)
                    move_count += 1
                    track.append(str(new_state))

                    r1 = -agent_e.states_long_life[new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]]
                    # r2 = -agent.states_episodic[new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]]
                    # beta = ep / num_explore_episodes
                    # r = (1 - beta) * r1 + (beta) * r2
                    # r = (1 - beta) * r2 + (beta) * r1
                    # r2 = env.reward(env.state, a, new_state)
                    # if r2 < -1:
                    #     r = r1*10 + r2
                    #     # print("trap, r:", r)
                    # else:
                    #     r = r1
                    #     r *= 10
                    r = r1
                    r *= 10
                    a_prime = agent_e.policy_explore_rl(new_state, env.actions(new_state))
                    # a_star = agent_e.policyNoRand_explore_rl(new_state, env.actions(new_state))
                    agent_e.learn_explore_sarsa(env.state, a, new_state, a_prime, r)
                    env.state = new_state
                    a = a_prime
            elif self.explore_config['e_mode'] == 'softmax':
                track = [str(env.state)]
                a = agent_e.policy_explore_softmax(env.state, env.actions(env.state))
                while move_count < self.explore_config['max_move_count']:
                    agent_e.state_actions_long_life[env.state[0], env.state[1], env.state[2], env.state[3], env.state[4], a] -= 1
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
                for _ in range(2):
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

    def _explore_stochastic(self):
        print("-----Begin Exploration-----")
        start_exploration = time.time()
        agent_e = ExploreStateBrain(env=self.env, explore_config=self.explore_config)
        env = self.env
        valid_states_ = tuple(env.valid_states)
        # env.choose_stochastic_opens(p1=50)    # this step can be done when initializing env

        for ep in range(self.explore_config['e_eps']):
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
                # start_state = random.choice(valid_states_)
                # start_coord = env.room_layout[start_state[0], start_state[1]]
                bad_start = 1
                while bad_start:
                    start_state = random.choice(valid_states_)
                    start_coord = env.room_layout[start_state[0], start_state[1]]
                    if start_coord != 'z':
                        bad_start = 0
                    else:
                        # print("bad start")
                        pass
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
                    agent_e.states_episodic[env.state[0], env.state[1], env.state[2], env.state[3], env.state[4]] += 1
                    agent_e.states_long_life[env.state[0], env.state[1], env.state[2], env.state[3], env.state[4]] += 1
                    new_state = env.step(env.state, a)
                    move_count += 1
                    env.update_walls(move_count=move_count)
                    # track.append(str(new_state))

                    r1 = -agent_e.states_long_life[new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]]
                    # r2 = -agent.states_episodic[new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]]
                    # beta = ep / num_explore_episodes
                    # r = (1 - beta) * r1 + (beta) * r2
                    # r = (1 - beta) * r2 + (beta) * r1
                    # r2 = env.reward(env.state, a, new_state)
                    # if r2 < -1:
                    #     r = r1*10 + r2
                    #     # print("trap, r:", r)
                    # else:
                    #     r = r1
                    #     r *= 10
                    r = r1
                    r *= 10
                    available_actions = env.actions(new_state)
                    if len(available_actions)>0:
                        a_prime = agent_e.policy_explore_rl(new_state, available_actions)
                        # a_star = agent_e.policyNoRand_explore_rl(new_state, env.actions(new_state))
                        agent_e.learn_explore_sarsa(env.state, a, new_state, a_prime, r)
                        track.append(str(new_state))
                        env.state = new_state
                        a = a_prime
                    else:
                        track.append(str(env.state))
            elif self.explore_config['e_mode'] == 'softmax':
                track = [str(env.state)]
                a = agent_e.policy_explore_softmax(env.state, env.actions(env.state))
                while move_count < self.explore_config['max_move_count']:
                    agent_e.state_actions_long_life[env.state[0], env.state[1], env.state[2], env.state[3], env.state[4], a] -= 1
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
                for _ in range(2):
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

    # def _ground_learning_evo(self, amdp, evo, ds_factor):       # not actually  perfect to use
    #     print("-----Begin Ground Learning EVO-----")
    #     start_ground_learning = time.time()
    #     agent_q = QLambdaBrain(env=self.env, ground_learning_config=self.ground_learning_config)
    #     env = self.env
    #     for evo_ in range(evo+1):
    #         print(f"---start evo {evo_}----")
    #         for ep in range(self.ground_learning_config['q_eps']):
    #             if (ep + 1) % 100 == 0:
    #                 print(f"episode_100: {ep} | avg_move_count: {int(np.mean(self.move_count_episodes[-100:]))} | "
    #                       f"avd_reward: {int(np.mean(self.reward_episodes[-100:]))} | "
    #                       f"env.state: {env.state} | "
    #                       f"env.flagcollected: {env.flags_collected} | "
    #                       f"agent.epsilon: {agent_q.epsilon} | "
    #                       f"agent.lr: {agent_q.lr}")
    #             # set epsilon
    #             epsilon_q_max = self.ground_learning_config['epsilon_q_max']
    #             temp_epsilon = epsilon_q_max/(evo_+1) - (epsilon_q_max/(evo_+1) / self.ground_learning_config['q_eps']) * ep
    #             # temp_epsilon = epsilon_q_max - (epsilon_q_max / self.ground_learning_config['q_eps']) * ep
    #             if temp_epsilon > 0.1:
    #                 agent_q.epsilon = round(temp_epsilon, 5)
    #
    #             env.reset()
    #             agent_q.reset_eligibility()
    #             # agent_q.reset_episodic_staff()
    #             episode_reward = 0
    #             move_count = 0
    #             track = [str(env.state)]
    #             a = agent_q.policy(env.state, env.actions(env.state))
    #             while not env.isTerminal(env.state):
    #                 # agent_q.states_episodic[env.state[0], env.state[1], env.state[2], env.state[3], env.state[4]] += 1
    #                 abstract_state = amdp.get_abstract_state(env.state)
    #                 new_state = env.step(env.state, a)
    #                 move_count += 1
    #                 track.append(str(new_state))
    #                 new_abstract_state = amdp.get_abstract_state(new_state)
    #                 a_prime = agent_q.policy(new_state, env.actions(new_state))  ##Greedy next-action selected
    #                 a_star = agent_q.policyNoRand(new_state, env.actions(new_state))
    #                 r = env.reward(env.state, a, new_state)  ## ground level reward
    #                 episode_reward += r
    #                 # if agent_q.states_episodic[env.state[0], env.state[1], env.state[2], env.state[3], env.state[4]] < 1:
    #                 #     r += 100
    #
    #                 value_new_abstract_state = amdp.get_value(new_abstract_state)
    #                 value_abstract_state = amdp.get_value(abstract_state)
    #                 shaping = self.ground_learning_config['gamma'] * value_new_abstract_state * \
    #                           self.ground_learning_config['omega'] - value_abstract_state * self.ground_learning_config['omega']
    #                 # shaping = 0
    #                 if evo_ == 0:
    #                     agent_q.learn(env.state, a, new_state, a_prime, a_star, r + shaping)
    #                 else:
    #                     agent_q.learn_sarsa(env.state, a, new_state, a_prime, r + shaping)
    #                 env.state = new_state
    #                 a = a_prime
    #
    #             self.reward_episodes.append(episode_reward)
    #             self.flags_episodes.append(env.flags_collected)
    #             self.move_count_episodes.append(move_count)
    #             self.flags_found_order_episodes.append(env.flags_found_order)
    #
    #             self.epsilons_episodes.append(agent_q.epsilon)
    #             self.gamma_episodes.append(agent_q.gamma)
    #             self.lr_episodes.append(agent_q.lr)
    #
    #             if ds_factor == 1:
    #                 self.sentences_period.append(track)
    #                 self.sentences_period_complete.append(track)
    #             else:
    #                 for _ in range(10):
    #                     down_sampled = [track[index] for index in sorted(random.sample(range(len(track)),
    #                                     math.floor(len(track) * self.explore_config['ds_factor'])))]
    #                     for i in range(0, len(down_sampled), 50):
    #                         self.sentences_period.append(down_sampled[i:i+50])
    #                 self.sentences_period_complete.append(track)
    #         last_percentage = int(len(self.sentences_period)/3)
    #         print("len(self.sentences_period):",len(self.sentences_period))
    #         for _ in range(1):
    #             self.sentences_collected.extend(self.sentences_period[-last_percentage:])
    #         self.sentences_period = []
    #         if evo_ < evo:
    #             gensim_opt = self._w2v_and_kmeans()
    #             rep = 0
    #             amdp = AMDP_General(self.sentences_period_complete, env=self.env, gensim_opt=gensim_opt)
    #             self.plot_maker.plot_each_cluster_layout_t_u_g(self.env, amdp, rep, self.path_results, save=0)
    #             self._solve_amdp(amdp)
    #             self.plot_maker.plot_each_amdp_values_t_u_g(self.env, amdp, rep, self.path_results, save=0)
    #
    #     end_ground_learning = time.time()
    #     ground_learning_time = end_ground_learning - start_ground_learning
    #     print("ground_learning_time:", ground_learning_time)
    #     self.ground_learning_time_repetitions.append(ground_learning_time)
    #
    #     print("-----Finish Ground Learning-----")
    #
    #     return agent_q

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

    def run(self, evo=0, heatmap=0, cluster_layout=0, time_comparison=0, final_policy=0):
        if not os.path.isdir(self.path_results):
            makedirs(self.path_results)
        self._print_before_start()
        curve_label = f"T-{int(self.num_clusters/8)}"
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
            # agent_e = self._explore()
            agent_e = self._explore_stochastic()
            print("size of self.sentences_period_complete", np.array(self.sentences_period_complete).shape)
            if heatmap:
                self.plot_maker.plot_each_heatmap_general(agent_e, rep, self.path_results, show=1, save=1)

            # solve w2v and k-means to get clusters and save cluster file
            gensim_opt = self._w2v_and_kmeans()
            if cluster_layout:
                self.plot_maker.plot_each_cluster_layout_general(gensim_opt, self.num_clusters, self.env, self.path_results, rep, save=1, show=1)

            # build and solve amdp
            amdp = AMDP_General(self.sentences_period_complete, env=self.env, gensim_opt=gensim_opt)
            self.plot_maker.plot_each_cluster_layout_t_u_g(self.env, amdp, rep, self.path_results)
            self._solve_amdp(amdp)
            self.plot_maker.plot_each_amdp_values_t_u_g(self.env, amdp, rep, self.path_results, 'general')
            # ground learning
            agent_g = self._ground_learning(amdp)
            # if evo == 0:
            #     self._ground_learning(amdp)
            # elif evo > 0:
            #     self._ground_learning_evo(amdp, evo, 0.5)

            # experiment timing ends and saved
            end_experiment = time.time()
            experiment_time = end_experiment-start_experiment
            self.experiment_time_repetitions.append(experiment_time)

            #To visualize the final policy
            if final_policy:
                value4walls = math.ceil(np.amax(agent_g.states_long_life)/10)
                for coord in np.argwhere(self.env.room_layout == "w"):
                    agent_g.states_long_life[coord[0], coord[1], :, :, :] = -value4walls
                self.plot_maker.plot_each_heatmap_general(agent_g, rep, self.path_results, show=1, save=1)

            # plot flags, reward, move_count for each rep
            self.plot_maker.plot_each_flag_reward_movecount(self.flags_episodes[self.explore_config['e_eps']:],
                                                            self.reward_episodes[self.explore_config['e_eps']:],
                                                            self.move_count_episodes[self.explore_config['e_eps']:],
                                                            rep, curve_label)
            # self.plot_maker.fig_each_rep.show()
            # save performance of each rep
            self.flags_episodes_repetitions.append(self.flags_episodes)
            self.reward_episodes_repetitions.append(self.reward_episodes)
            self.move_count_episodes_repetitions.append(self.move_count_episodes)

        # plot mean performance among all reps
        ### ax_title = f"flags collection in {'big' if self.big==1 else 'small'} {self.env.maze_name}"
        self._pickler(approach='general', granu=f"{self.num_clusters}")

        sliced_f_ep_rep = np.array(self.flags_episodes_repetitions)[:, self.explore_config['e_eps']:]
        sliced_r_ep_rep = np.array(self.reward_episodes_repetitions)[:, self.explore_config['e_eps']:]
        sliced_m_ep_rep = np.array(self.move_count_episodes_repetitions)[:, self.explore_config['e_eps']:]
        self.plot_maker.plot_mean_performance_across_reps(sliced_f_ep_rep, sliced_r_ep_rep, sliced_m_ep_rep, curve_label)

        if time_comparison:
            self.plot_maker.plot_mean_time_comparison(self.experiment_time_repetitions, self.solve_amdp_time_repetitions,
                                                      self.ground_learning_time_repetitions, self.exploration_time_repetitions,
                                                      self.solve_word2vec_time_repetitions, self.solve_kmeans_time_repetitions,
                                                      bar_label=curve_label)

        # self._results_upload()

if __name__ == "__main__":
    maze = 'basic'  # low_connectivity2/external_maze21x21_1/external_maze31x31_2/strips2/spiral/basic/open_space/high_connectivity
    big = 0
    e_mode = 'sarsa'   # 'sarsa' or 'softmax'pwd
    e_start = 'last'   # 'random' or 'last' or 'mix'
    e_eps = 1000
    mm = 100
    ds_factor = 0.5

    q_eps = 500
    repetitions = 2
    rep_size = 128
    win_size = 50
    sg = 1  # 'SG' or 'CBOW'
    # numbers_of_clusters = [9, 16, 25, 36]     # number of abstract states for Uniform will be matched with the number of clusters
    numbers_of_clusters = [16]  # number of abstract states for Uniform will be matched with the number of clusters

    k_means_pkg = 'sklearn'    # 'sklearn' or 'nltk'
    interpreter = 'R'     # L or R
    std_factor = 1 / np.sqrt(10)

    print_to_file = 0
    show = 1
    save = 0
    for i in range(len(numbers_of_clusters)):
        # set directory to store imgs and files
        path_results =f"./cluster_layout/{maze}_big={big}" \
                      f"/topology-vs-uniform{numbers_of_clusters}-oop/rp{repetitions}_{e_start}{e_eps}+{q_eps}_mm{mm}_" \
                      f"ds{ds_factor}_win{win_size}_rep{rep_size}_sg{sg}_{k_means_pkg}_{interpreter}/k[{numbers_of_clusters[i]}]"
        if not os.path.isdir(path_results):
            makedirs(path_results)
        if print_to_file == 1:
            sys.stdout = open(f"{path_results}/output.txt", 'w')
            sys.stderr = sys.stdout

        plot_maker = PlotMaker(repetitions, std_factor, 2)   # third argument should match num of approaches below

        # ===topology approach===
        topology_maker = TopologyExpMaker(env_name=maze, big=big, e_mode=e_mode, e_start=e_start, e_eps=e_eps, mm=mm, ds_factor=ds_factor,
                     rep_size=rep_size, win_size=win_size, sg=sg, num_clusters=numbers_of_clusters[i], k_means_pkg=k_means_pkg, q_eps=q_eps,
                     repetitions=repetitions, interpreter=interpreter, print_to_file=print_to_file, plot_maker=plot_maker, path_results=path_results)
        topology_maker.run()

        # ===uniform approach===
        # ---match number of abstract state same with the one in topology approach, in order to be fair.
        a = math.ceil(topology_maker.env.size[0] / np.sqrt(numbers_of_clusters[i]))
        b = math.ceil(topology_maker.env.size[1] / np.sqrt(numbers_of_clusters[i]))
        print("(a,b): ", (a,b))
        uniform_maker = UniformExpMaker(env_name=maze, big=big, tiling_size=(a, b), q_eps=q_eps, repetitions=repetitions,
                                        interpreter=interpreter, print_to_file=print_to_file, plot_maker=plot_maker,
                                        path_results=path_results)
        uniform_maker.run()

        # ===general approach===
        # general_maker = GeneralExpMaker(env_name=maze, big=big, e_mode=e_mode, e_start='random', e_eps=int(e_eps*6), mm=mm, ds_factor=ds_factor,
        #              rep_size=rep_size, win_size=win_size, sg=sg, num_clusters=int(numbers_of_clusters[i]*8), k_means_pkg=k_means_pkg, q_eps=q_eps,
        #              repetitions=repetitions, interpreter=interpreter, print_to_file=print_to_file, plot_maker=plot_maker, path_results=path_results)
        # general_maker.run()

        # ===plot and save summary===
        print("saving fig_each_rep ...")
        if show:
            plot_maker.fig_each_rep.show()
        if save:
            plot_maker.fig_each_rep.savefig(f"{path_results}/plots_of_each_rep.png", dpi=100, facecolor='w', edgecolor='w',
                                            orientation='portrait', format=None,
                                            transparent=False, bbox_inches=None, pad_inches=0.1)

        print("saving fig_mean_performance ...")
        if show:
            plot_maker.fig_mean_performance.show()
        if save:
            plot_maker.fig_mean_performance.savefig(f"{path_results}/mean_results.png",
                                                    dpi=200, facecolor='w', edgecolor='w', orientation='portrait',
                                                    format=None, transparent=False, bbox_inches=None, pad_inches=0.1)

        print("saving fig_time_consumption ...")
        if show:
            plot_maker.fig_time_consumption.show()
        if save:
            plot_maker.fig_time_consumption.savefig(f"{path_results}/time_consumption.png",
                                                    dpi=500, facecolor='w', edgecolor='w', orientation='portrait',
                                                    format=None, transparent=False, bbox_inches=None, pad_inches=0.1)

        if print_to_file == 1:
            sys.stdout.close()