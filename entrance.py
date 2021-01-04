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
# from PIL import Image
import time
from abstractions.abstraction_all_approaches import AMDP_Topology_Uniform, AMDP_General
from envs.maze_env_general_all_approaches import Maze
from RL_brains.RL_brain_all_approaches import ExploreStateBrain, ExploreCoordBrain, QLambdaBrain
from gensim_operations.gensim_operation_all_approaches import GensimOperator_Topology, GensimOperator_General
from main_across_all_approaches import TopologyExpMaker, UniformExpMaker, GeneralExpMaker, PlotMaker

def compare_para_1approach():
    maze = 'strips2'  # low_connectivity2/external_maze21x21_1/external_maze31x31_2/strips2/spiral/basic/open_space
    big = 0
    e_mode = 'sarsa'  # 'sarsa' or 'softmax'pwd
    e_start = 'last'  # 'random' or 'last' or 'mix'
    e_eps = 1000
    mm = 100
    ds_factor = 0.5

    q_eps = 500
    repetitions = 2
    rep_size = 128
    win_size = 50
    sg = 1  # 'SG' or 'CBOW'
    # numbers_of_clusters = [9, 16, 25, 36]     # number of abstract states for Uniform will be matched with the number of clusters
    numbers_of_clusters = [9]  # number of abstract states for Uniform will be matched with the number of clusters

    k_means_pkg = 'sklearn'  # 'sklearn' or 'nltk'
    interpreter = 'R'  # L or R
    std_factor = 1 / np.sqrt(10)

    print_to_file = 0
    show = 1
    save = 0

    path_results = f"./cluster_layout/{maze}_big={big}" \
                   f"/topo_compare_across_para-oop/rp{repetitions}_{e_start}{e_eps}+{q_eps}_mm{mm}_" \
                   f"ds{ds_factor}_c{numbers_of_clusters}_win{win_size}_rep{rep_size}_sg{sg}_{k_means_pkg}_{interpreter}"
    if not os.path.isdir(path_results):
        makedirs(path_results)
    if print_to_file == 1:
        sys.stdout = open(f"{path_results}/output.txt", 'w')
        sys.stderr = sys.stdout

    plot_maker = PlotMaker(repetitions, std_factor, len(numbers_of_clusters))

    for i in range(len(numbers_of_clusters)):
        # topology_maker = TopologyExpMaker(env_name=maze, big=big, e_mode=e_mode, e_start=e_start, e_eps=e_eps, mm=mm, ds_factor=ds_factor,
        #                                   rep_size=rep_size, win_size=win_size, sg=sg, num_clusters=numbers_of_clusters[i],
        #                                   k_means_pkg=k_means_pkg, q_eps=q_eps,
        #                                   repetitions=repetitions, interpreter=interpreter, print_to_file=print_to_file, plot_maker=plot_maker,
        #                                   path_results=path_results)
        # topology_maker.run()

        general_maker = GeneralExpMaker(env_name=maze, big=big, e_mode=e_mode, e_start='random', e_eps=int(e_eps * 6), mm=mm, ds_factor=ds_factor,
                                        rep_size=rep_size, win_size=win_size, sg=sg, num_clusters=int(numbers_of_clusters[i] * 8),
                                        k_means_pkg=k_means_pkg, q_eps=q_eps,
                                        repetitions=repetitions, interpreter=interpreter, print_to_file=print_to_file, plot_maker=plot_maker,
                                        path_results=path_results)
        general_maker.run()

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


def compare_para_2approaches():
    maze = 'high_connectivity'  # low_connectivity2/external_maze21x21_1/external_maze31x31_2/strips2/spiral/basic/open_space/high_connectivity
    big = 1
    e_mode = 'sarsa'  # 'sarsa' or 'softmax'pwd
    e_start = 'last'  # 'random' or 'last' or 'mix'
    e_eps = 5000
    mm = 100
    ds_factor = 0.5

    q_eps = 500
    repetitions = 2
    rep_size = 128
    win_size = 50
    sg = 1  # 'SG' or 'CBOW'
    # numbers_of_clusters = [9, 16, 25, 36]     # number of abstract states for Uniform will be matched with the number of clusters
    numbers_of_clusters = [9, 25]  # number of abstract states for Uniform will be matched with the number of clusters

    k_means_pkg = 'sklearn'  # 'sklearn' or 'nltk'
    interpreter = 'R'  # L or R
    std_factor = 1 / np.sqrt(10)

    print_to_file = 1
    show = 0
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
        topology_maker = TopologyExpMaker(env_name=maze, big=big, e_mode=e_mode, e_start=e_start, e_eps=e_eps, mm=mm, ds_factor=ds_factor,
                                          rep_size=rep_size, win_size=win_size, sg=sg, num_clusters=numbers_of_clusters[i],
                                          k_means_pkg=k_means_pkg, q_eps=q_eps,
                                          repetitions=repetitions, interpreter=interpreter, print_to_file=print_to_file, plot_maker=plot_maker,
                                          path_results=path_results)
        topology_maker.run(heatmap=0, cluster_layout=0)

        # ===uniform approach===
        # ---match number of abstract state same with the one in topology approach, in order to be fair.
        a = math.ceil(topology_maker.env.size[0] / np.sqrt(numbers_of_clusters[i]))
        b = math.ceil(topology_maker.env.size[1] / np.sqrt(numbers_of_clusters[i]))
        print("(a,b): ", (a,b))
        uniform_maker = UniformExpMaker(env_name=maze, big=big, tiling_size=(a, b), q_eps=q_eps, repetitions=repetitions,
                                        interpreter=interpreter, print_to_file=print_to_file, plot_maker=plot_maker,
                                        path_results=path_results)
        uniform_maker.run()

        # general_maker = GeneralExpMaker(env_name=maze, big=big, e_mode=e_mode, e_start='random', e_eps=int(e_eps * 6), mm=mm, ds_factor=ds_factor,
        #                                 rep_size=rep_size, win_size=win_size, sg=sg, num_clusters=int(numbers_of_clusters[i] * 8),
        #                                 k_means_pkg=k_means_pkg, q_eps=q_eps,
        #                                 repetitions=repetitions, interpreter=interpreter, print_to_file=print_to_file, plot_maker=plot_maker,
        #                                 path_results=path_results)
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
        plot_maker.fig_mean_performance.savefig(f"{path_results}/mean_results_errorbar_yerror.png",
                                                dpi=200, facecolor='w', edgecolor='w', orientation='portrait',
                                                format=None, transparent=False, bbox_inches=None, pad_inches=0.1)

    # print("saving fig_time_consumption ...")
    # if show:
    #     plot_maker.fig_time_consumption.show()
    # if save:
    #     plot_maker.fig_time_consumption.savefig(f"{path_results}/time_consumption.png",
    #                                             dpi=500, facecolor='w', edgecolor='w', orientation='portrait',
    #                                             format=None, transparent=False, bbox_inches=None, pad_inches=0.1)

    if print_to_file == 1:
        sys.stdout.close()

def compare_approaches():
    maze = 'basic'  # low_connectivity2/external_maze21x21_1/external_maze31x31_2/strips2/spiral/basic/open_space/high_connectivity
    big = 0
    e_mode = 'sarsa'  # 'sarsa' or 'softmax'pwd
    e_start = 'last'  # 'random' or 'last' or 'mix'
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

    k_means_pkg = 'sklearn'  # 'sklearn' or 'nltk'
    interpreter = 'R'  # L or R
    std_factor = 1 / np.sqrt(10)

    print_to_file = 0
    show = 1
    save = 0
    for i in range(len(numbers_of_clusters)):
        # set directory to store imgs and files
        path_results = f"./cluster_layout/{maze}_big={big}" \
                       f"/topology-vs-uniform{numbers_of_clusters}-oop/rp{repetitions}_{e_start}{e_eps}+{q_eps}_mm{mm}_" \
                       f"ds{ds_factor}_win{win_size}_rep{rep_size}_sg{sg}_{k_means_pkg}_{interpreter}/k[{numbers_of_clusters[i]}]"
        if not os.path.isdir(path_results):
            makedirs(path_results)
        if print_to_file == 1:
            sys.stdout = open(f"{path_results}/output.txt", 'w')
            sys.stderr = sys.stdout

        plot_maker = PlotMaker(repetitions, std_factor, 2)  # third argument should match num of approaches below

        # ===topology approach===
        topology_maker = TopologyExpMaker(env_name=maze, big=big, e_mode=e_mode, e_start=e_start, e_eps=e_eps, mm=mm, ds_factor=ds_factor,
                                          rep_size=rep_size, win_size=win_size, sg=sg, num_clusters=numbers_of_clusters[i], k_means_pkg=k_means_pkg,
                                          q_eps=q_eps,
                                          repetitions=repetitions, interpreter=interpreter, print_to_file=print_to_file, plot_maker=plot_maker,
                                          path_results=path_results)
        topology_maker.run()

        # ===uniform approach===
        # ---match number of abstract state same with the one in topology approach, in order to be fair.
        a = math.ceil(topology_maker.env.size[0] / np.sqrt(numbers_of_clusters[i]))
        b = math.ceil(topology_maker.env.size[1] / np.sqrt(numbers_of_clusters[i]))
        print("(a,b): ", (a, b))
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

if __name__ == "__main__":
    # compare_para_1approach
    compare_para_2approaches()
    # compare_approaches()