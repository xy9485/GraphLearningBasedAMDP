import sys
# print(sys.path)
import copy
import math
import statistics
import random
# print(matplotlib.get_backend())
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pandas as pd
from os import makedirs
import numpy as np
import os
import pickle
# from PIL import Image
import time
from abstractions.abstraction_all_approaches import AMDP_Topology_Uniform, AMDP_General
from envs.maze_env_general_all_approaches import Maze
from RL_brains.RL_brain_all_approaches import ExploreStateBrain, ExploreCoordBrain, QLambdaBrain
from gensim_operations.gensim_operation_all_approaches import GensimOperator_Topology, GensimOperator_General
from main_across_all_approaches import TopologyExpMaker, UniformExpMaker, GeneralExpMaker, PlotMaker

def plot_maze(maze, big, version):
    env = Maze(maze=maze, big=big)
    PlotMaker.plot_maze(env=env, version=version, show=1, save=0)

def plot_maze_trap(maze, big, version):
    env = Maze(maze=maze, big=big)
    PlotMaker.plot_maze_trap(env=env, version=version, show=1, save=0)

def plot_room_layout(maze, big, version):
    env = Maze(maze=maze, big=big)
    PlotMaker.plot_manual_rooms(env=env, version=version, show=1, save=1)

def pload(path_results, approach, granu, expmaker):
    expmaker.repetitions = 0

    pf = f"{path_results}/performance/{approach}"
    tm = f"{path_results}/times/{approach}"

    with open(f"{pf}/k{granu}_flags_eps_reps.pkl", 'rb') as f:
        expmaker.flags_episodes_repetitions = pickle.load(f)

    with open(f"{pf}/k{granu}_rewards_eps_reps.pkl", 'rb') as f:
        expmaker.reward_episodes_repetitions = pickle.load(f)

    with open(f"{pf}/k{granu}_mc_eps_reps.pkl", 'rb') as f:
        expmaker.move_count_episodes_repetitions = pickle.load(f)

    with open(f"{tm}/k{granu}_experiment.pkl", 'rb') as f:
        expmaker.experiment_time_repetitions = pickle.load(f)
    with open(f"{tm}/k{granu}_solve_amdp.pkl", 'rb') as f:
        expmaker.solve_amdp_time_repetitions = pickle.load(f)
    with open(f"{tm}/k{granu}_ground_learning.pkl", 'rb') as f:
        expmaker.ground_learning_time_repetitions = pickle.load(f)
    if approach == 'general' or approach == 'topology':
        with open(f"{tm}/k{granu}_exploration.pkl", 'rb') as f:
            expmaker.exploration_time_repetitions = pickle.load(f)
        with open(f"{tm}/k{granu}_solve_w2v.pkl", 'rb') as f:
            expmaker.solve_word2vec_time_repetitions = pickle.load(f)
        with open(f"{tm}/k{granu}_solve_kmeans.pkl", 'rb') as f:
            expmaker.solve_kmeans_time_repetitions = pickle.load(f)

def compare_para_1approach():
    maze = 'low_connectivity2'  # low_connectivity2/external_maze21x21_1/external_maze31x31_2/strips2/spiral/basic/open_space/high_connectivity
    big = 1
    e_mode = 'sarsa'  # 'sarsa' or 'softmax'pwd
    e_start = 'last'  # 'random' or 'last' or 'mix'
    e_eps = 5000
    mm = 100
    ds_factor = 0.5

    q_eps = 500
    repetitions = 25
    rep_size = 128
    win_size = 50
    sg = 1  # 'SG' or 'CBOW'
    numbers_of_clusters = [9, 16, 25, 36]     # number of abstract states for Uniform will be matched with the number of clusters
    # numbers_of_clusters = [9]  # number of abstract states for Uniform will be matched with the number of clusters

    k_means_pkg = 'sklearn'  # 'sklearn' or 'nltk'
    interpreter = 'R'  # L or R
    std_factor = 1 / np.sqrt(10)

    print_to_file = 0
    show = 0
    save = 1
    # path_results = f"./cluster_layout/{maze}_big={big}" \
    #                f"/topo_compare_across_para-oop/gs0_rp{repetitions}_{e_start}{e_eps}+{q_eps}_mm{mm}_" \
    #                f"ds{ds_factor}_c{numbers_of_clusters}_win{win_size}_rep{rep_size}_sg{sg}_{k_means_pkg}_{interpreter}"
    path_results = f"./cluster_layout/{maze}_big={big}" \
                   f"/topo_compare_across_para-oop/gs0_rp{repetitions}_{e_start}{e_eps}+{q_eps}_mm{mm}_" \
                   f"ds{ds_factor}_c{numbers_of_clusters}_win{win_size}_rep{rep_size}_sg{sg}_{k_means_pkg}_{interpreter}"
    if not os.path.isdir(path_results):
        makedirs(path_results)
    if print_to_file == 1:
        sys.stdout = open(f"{path_results}/output.txt", 'w')
        sys.stderr = sys.stdout

    plot_maker = PlotMaker(repetitions, std_factor, len(numbers_of_clusters))

    for i in range(len(numbers_of_clusters)):
        topology_maker = TopologyExpMaker(env_name=maze, big=big, e_mode=e_mode, e_start=e_start, e_eps=e_eps, mm=mm, ds_factor=ds_factor,
                                          rep_size=rep_size, win_size=win_size, sg=sg, num_clusters=numbers_of_clusters[i],
                                          k_means_pkg=k_means_pkg, q_eps=q_eps,
                                          repetitions=repetitions, interpreter=interpreter, print_to_file=print_to_file, plot_maker=plot_maker,
                                          path_results=path_results)
        pload(path_results, 'topology', numbers_of_clusters[i], topology_maker)
        topology_maker.run(heatmap=0, cluster_layout=0, time_comparison=0)

        # env = Maze(maze=maze, big=big)
        # a = math.ceil(env.size[0] / np.sqrt(numbers_of_clusters[i]))
        # b = math.ceil(env.size[1] / np.sqrt(numbers_of_clusters[i]))
        # print("(a,b): ", (a, b))
        # uniform_maker = UniformExpMaker(env_name=maze, big=big, tiling_size=(a, b), q_eps=q_eps, repetitions=repetitions,
        #                                 interpreter=interpreter, print_to_file=print_to_file, plot_maker=plot_maker,
        #                                 path_results=path_results)
        # pload(path_results, 'uniform', (a, b), uniform_maker)
        # uniform_maker.run(time_comparison=1)

        # general_maker = GeneralExpMaker(env_name=maze, big=big, e_mode=e_mode, e_start='random', e_eps=int(e_eps * 6), mm=mm, ds_factor=ds_factor,
        #                                 rep_size=rep_size, win_size=win_size, sg=sg, num_clusters=int(numbers_of_clusters[i] * 8),
        #                                 k_means_pkg=k_means_pkg, q_eps=q_eps,
        #                                 repetitions=repetitions, interpreter=interpreter, print_to_file=print_to_file, plot_maker=plot_maker,
        #                                 path_results=path_results)
        # pload(path_results, 'general', numbers_of_clusters[i] * 8, general_maker)
        # general_maker.run()

    # ===plot and save summary===
    # print("saving fig_each_rep ...")
    # if show:
    #     plot_maker.fig_each_rep.show()
    # if save:
    #     plot_maker.fig_each_rep.savefig(f"{path_results}/plots_of_each_rep.png", dpi=100, facecolor='w', edgecolor='w',
    #                                     orientation='portrait', format=None,
    #                                     transparent=False, bbox_inches=None, pad_inches=0.1)

    print("saving fig_mean_performance ...")
    if show:
        plot_maker.fig_mean_performance.show()
    if save:
        plot_maker.fig_mean_performance.savefig(f"{path_results}/mean_results2.png",
                                                dpi=200, facecolor='w', edgecolor='w', orientation='portrait',
                                                format=None, transparent=False, bbox_inches=None, pad_inches=0.1)

    print("saving fig_time_consumption ...")
    if show:
        plot_maker.fig_time_consumption.show()
    if save:
        plot_maker.fig_time_consumption.savefig(f"{path_results}/time_consumption2.png",
                                                dpi=200, facecolor='w', edgecolor='w', orientation='portrait',
                                                format=None, transparent=False, bbox_inches=None, pad_inches=0.1)

    if print_to_file == 1:
        sys.stdout.close()


def compare_para_2approaches():
    maze = 'open_space'  # low_connectivity2/external_maze21x21_1/external_maze31x31_2/strips2/spiral/basic/open_space/high_connectivity
    big = 0
    e_mode = 'sarsa'  # 'sarsa' or 'softmax'pwd
    e_start = 'random'  # 'random' or 'last' or 'mix'
    e_eps = 500
    mm = 100
    ds_factor = 0.5

    q_eps = 500
    repetitions = 2
    rep_size = 128
    win_size = 50
    sg = 1  # 'SG' or 'CBOW'
    # numbers_of_clusters = [9, 16, 25, 36]     # number of abstract states for Uniform will be matched with the number of clusters
    numbers_of_clusters = [16, 25]  # number of abstract states for Uniform will be matched with the number of clusters

    k_means_pkg = 'sklearn'  # 'sklearn' or 'nltk'
    interpreter = 'R'  # L or R
    std_factor = 1 / np.sqrt(10)

    print_to_file = 0
    show = 1
    save = 1

    path_results = f"./cluster_layout/{maze}_big={big}" \
                   f"/topology-vs-uniform-multi-clusters-oop/rp{repetitions}_{e_start}{e_eps}+{q_eps}_mm{mm}_" \
                   f"ds{ds_factor}_c{numbers_of_clusters}_win{win_size}_rep{rep_size}_sg{sg}_{k_means_pkg}_{interpreter}"
    if not os.path.isdir(path_results):
        makedirs(path_results)
    if print_to_file == 1:
        sys.stdout = open(f"{path_results}/output.txt", 'w')
        sys.stderr = sys.stdout

    plot_maker = PlotMaker(repetitions, std_factor, len(numbers_of_clusters)*2)

    for i in range(len(numbers_of_clusters)):
        # topology_maker = TopologyExpMaker(env_name=maze, big=big, e_mode=e_mode, e_start=e_start, e_eps=e_eps, mm=mm, ds_factor=ds_factor,
        #                                   rep_size=rep_size, win_size=win_size, sg=sg, num_clusters=numbers_of_clusters[i],
        #                                   k_means_pkg=k_means_pkg, q_eps=q_eps,
        #                                   repetitions=repetitions, interpreter=interpreter, print_to_file=print_to_file, plot_maker=plot_maker,
        #                                   path_results=path_results)
        # topology_maker.run(heatmap=0, cluster_layout=0, time_comparison=0)

        general_maker = GeneralExpMaker(env_name=maze, big=big, e_mode=e_mode, e_start='random', e_eps=int(e_eps * 6), mm=mm, ds_factor=ds_factor,
                                        rep_size=rep_size, win_size=win_size, sg=sg, num_clusters=int(numbers_of_clusters[i] * 8),
                                        k_means_pkg=k_means_pkg, q_eps=q_eps,
                                        repetitions=repetitions, interpreter=interpreter, print_to_file=print_to_file, plot_maker=plot_maker,
                                        path_results=path_results)
        general_maker.run(time_comparison=1)

        # ===uniform approach===
        # ---match number of abstract state same with the one in topology approach, in order to be fair.
        env = Maze(maze=maze, big=big)
        a = math.ceil(env.size[0] / np.sqrt(numbers_of_clusters[i]))
        b = math.ceil(env.size[1] / np.sqrt(numbers_of_clusters[i]))
        print("(a,b): ", (a,b))
        uniform_maker = UniformExpMaker(env_name=maze, big=big, tiling_size=(a, b), q_eps=q_eps, repetitions=repetitions,
                                        interpreter=interpreter, print_to_file=print_to_file, plot_maker=plot_maker,
                                        path_results=path_results)
        uniform_maker.run(time_comparison=1)



    # ===plot and save summary===
    # print("saving fig_each_rep ...")
    # if show:
    #     plot_maker.fig_each_rep.show()
    # if save:
    #     plot_maker.fig_each_rep.savefig(f"{path_results}/plots_of_each_rep.png", dpi=100, facecolor='w', edgecolor='w',
    #                                     orientation='portrait', format=None,
    #                                     transparent=False, bbox_inches=None, pad_inches=0.1)

    print("saving fig_mean_performance ...")
    if show:
        plot_maker.fig_mean_performance.show()
    if save:
        plot_maker.fig_mean_performance.savefig(f"{path_results}/mean_results_errorbar_yerror.png",
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

def compare_approaches():
    maze = 'low_connectivity2'  # low_connectivity2/external_maze21x21_1/external_maze31x31_2/strips2/spiral/basic/open_space/high_connectivity
    big = 0
    e_mode = 'sarsa'  # 'sarsa' or 'softmax'
    e_start = 'random'  # 'random' or 'last' or 'semi_random'
    e_eps = 500
    mm = 100
    ds_factor = 0.5

    q_eps = 500
    repetitions = 2
    rep_size = 128
    win_size = 50
    sg = 1  # 'SG' or 'CBOW'
    # numbers_of_clusters = [9, 16, 25, 36]     # number of abstract states for Uniform will be matched with the number of clusters
    numbers_of_clusters = [16, 25]  # number of abstract states for Uniform will be matched with the number of clusters

    k_means_pkg = 'sklearn'  # 'sklearn' or 'nltk'
    interpreter = 'R'  # L or R
    std_factor = 1 / np.sqrt(10)

    print_to_file = 0
    show = 1
    save = 0
    # plot_maze(maze, big=0, version=1)
    for i in range(len(numbers_of_clusters)):
        # set directory to store imgs and files
        # path_results = f"./cluster_layout/{maze}_big={big}" \
        #                f"/topology-vs-uniform{numbers_of_clusters}-oop/v4_rp{repetitions}_{e_start}{e_eps}+{q_eps}_mm{mm}_" \
        #                f"ds{ds_factor}_win{win_size}_rep{rep_size}_sg{sg}_{k_means_pkg}_{interpreter}/k[{numbers_of_clusters[i]}]"
        path_results = f"./cluster_layout/{maze}_big={big}" \
                       f"/general-vs-uniform{numbers_of_clusters}-oop/rp{repetitions}_{e_start}{e_eps}+{q_eps}_mm{mm}_" \
                       f"ds{ds_factor}_win{win_size}_rep{rep_size}_sg{sg}_{k_means_pkg}_{interpreter}/k[{numbers_of_clusters[i]}]"
        # path_results = f"./cluster_layout/{maze}_big={big}" \
        #                f"/uniform{numbers_of_clusters}-oop/rp{repetitions}_{e_start}{e_eps}+{q_eps}_mm{mm}_" \
        #                f"ds{ds_factor}_win{win_size}_rep{rep_size}_sg{sg}_{k_means_pkg}_{interpreter}/k[{numbers_of_clusters[i]}]0--"
        if not os.path.isdir(path_results):
            makedirs(path_results)
        if print_to_file == 1:
            sys.stdout = open(f"{path_results}/output.txt", 'w')
            sys.stderr = sys.stdout
        plot_maker = PlotMaker(repetitions, std_factor, 2)  # third argument should match num of approaches below

        # ===topology approach===
        # topology_maker = TopologyExpMaker(env_name=maze, big=big, e_mode=e_mode, e_start=e_start, e_eps=e_eps, mm=mm, ds_factor=ds_factor,
        #                                   rep_size=rep_size, win_size=win_size, sg=sg, num_clusters=numbers_of_clusters[i], k_means_pkg=k_means_pkg,
        #                                   q_eps=q_eps, repetitions=repetitions, interpreter=interpreter, print_to_file=print_to_file,
        #                                   plot_maker=plot_maker, path_results=path_results)
        # pload(path_results, 'topology', numbers_of_clusters[i], topology_maker)
        # topology_maker.run(time_comparison=1)


        # ===general approach===
        general_maker = GeneralExpMaker(env_name=maze, big=big, e_mode=e_mode, e_start='random', e_eps=int(e_eps*6), mm=mm, ds_factor=ds_factor,
                     rep_size=rep_size, win_size=win_size, sg=sg, num_clusters=int(numbers_of_clusters[i]*8), k_means_pkg=k_means_pkg, q_eps=q_eps,
                     repetitions=repetitions, interpreter=interpreter, print_to_file=print_to_file, plot_maker=plot_maker, path_results=path_results)
        # pload(path_results, 'general', numbers_of_clusters[i]*8, general_maker)
        general_maker.run(heatmap=0, time_comparison=1)

        # ===uniform approach===
        # ---match number of abstract state same with the one in topology approach, in order to be fair.
        env = Maze(maze=maze, big=big)
        a = math.ceil(env.size[0] / np.sqrt(numbers_of_clusters[i]))
        b = math.ceil(env.size[1] / np.sqrt(numbers_of_clusters[i]))
        print("(a,b): ", (a, b))
        uniform_maker = UniformExpMaker(env_name=maze, big=big, tiling_size=(a, b), q_eps=q_eps, repetitions=repetitions,
                                        interpreter=interpreter, print_to_file=print_to_file, plot_maker=plot_maker,
                                        path_results=path_results)
        # pload(path_results, 'uniform', (a, b), uniform_maker)
        uniform_maker.run(time_comparison=1)

        # ===plot and save summary===
        print("saving fig_each_rep ...")
        if show:
            plot_maker.fig_each_rep.show()
        if save:
            plot_maker.fig_each_rep.savefig(f"{path_results}/plots_of_each_rep2.png",
                                            dpi=100, transparent=False, bbox_inches='tight', pad_inches=0.1)

        print("saving fig_mean_performance ...")
        if show:
            plot_maker.fig_mean_performance.show()
        if save:
            plot_maker.fig_mean_performance.savefig(f"{path_results}/mean_results2.png",
                                                    dpi=300, transparent=False, bbox_inches='tight', pad_inches=0.1)

        print("saving fig_time_consumption ...")
        if show:
            plot_maker.fig_time_consumption.show()
        if save:
            plot_maker.fig_time_consumption.savefig(f"{path_results}/time_consumption2.png",
                                                    dpi=300, transparent=False, bbox_inches='tight', pad_inches=0.1)

        if print_to_file == 1:
            sys.stdout.close()

if __name__ == "__main__":
    # compare_para_1approach()
    # compare_para_2approaches()
    compare_approaches()
    # plot_maze('spiral', big=0, version=1)     # low_connectivity2/external_maze21x21_1/external_maze31x31_2/strips2/spiral/basic/open_space/high_connectivity
    # plot_room_layout('simple', big=0, version=0)
    # plot_maze_trap('low_connectivity', big=0, version=1)
