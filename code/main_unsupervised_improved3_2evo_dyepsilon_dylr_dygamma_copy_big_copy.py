import sys
print(sys.path)
import copy
import math
import random
import pickle
import matplotlib
# print(matplotlib.get_backend())
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import font
import tkinter as tk
import pandas as pd
from errno import EEXIST
from os import makedirs, path
import numpy as np
import os
from stat import S_IREAD, S_IRGRP, S_IROTH
# from PIL import Image
import time
from statistics import mean
from hello import yahoo
from abstraction import AMDP
from maze_env_general import Maze
from RL_brain_fast_explore import WatkinsQLambda
from gensim_operation_online import GensimOperator

# from sympy.core.symbol import symbols
# from sympy.solvers.solveset import nonlinsolve
# from sympy import exp, solve

# abstraction_mode = [None, (3, 3), (4, 4), (5, 5), (7, 7), (9, 9), None]   # 可修改
abstraction_mode = [None]  # 可修改
env = Maze(maze='external_maze1_61x61')  # initialize env 可修改
print("env.name:",env.maze_name)
print("env.flags:", env.flags, env.room_layout[env.flags[0][0],env.flags[0][1]], env.room_layout[env.flags[1][0],env.flags[1][1]], env.room_layout[env.flags[2][0],env.flags[2][1]])
print("env.goal:", env.goal, env.room_layout[env.goal[0],env.goal[1]])
print("env.state:",env.state)

num_of_actions = 4
num_of_experiments = len(abstraction_mode)
lr = 0.1
lam = 0.9
gamma = 0.999
omega = 100
epsilon = 1  # probability for choosing random action  #可修改
epsilon_max = 1
epsilon_max1 = 1
print(f"lr={lr} / lam={lam} / gamma={gamma} / omega={omega} / epsilon_max={epsilon_max} / epsilon_max1={epsilon_max1}")
num_randomwalk_episodes = 500
second_evolution = num_randomwalk_episodes + 1000
# third_evolution = 500 + 1500
# fourth_evolution = 500 + 1500
num_saved_from_p1 = 1
# num_saved_from_p2 = 1500
num_of_episodes = num_randomwalk_episodes + 4004        # 可修改
num_of_repetitions = 2 # 可修改
max_move_count = 5000
num_overflowed_eps = 0
min_length_to_save_as_path = 400
length_of_phase1 = second_evolution-num_randomwalk_episodes
length_of_phase2 = num_of_episodes-second_evolution

explore_mode = 'rl diff starts'
# lr_max = lr
# lr_min = 0.01
# a, b= symbols('a, b', real=True)
# solution = solve([exp(-(0+b)/a)-lr_max, exp(-(length_of_phase1+b)/a)-lr_min], [a, b])
# lr_func_a = solution[a]
# lr_func_b = solution[b]

# gamma_max = gamma
# gamma_min = 0.97
# a, b= symbols('a, b', real=True)
# solution = solve([exp(-(0+b)/a)-gamma_max, exp(-(length_of_phase1+b)/a)-gamma_min], [a, b])
# gamma_func_a = solution[a]
# gamma_func_b = solution[b]


config = {
    'maze': env.maze_name,
    'mode': 'random+biased_paths2',
    'ep': num_of_episodes,
    'rp': num_of_repetitions,
    'max_move_count': max_move_count,
    'min_length_to_save': min_length_to_save_as_path,
    'representation_size': 300,
    'window': 300,
    'kmeans_clusters': [15, 25, 30],
    'package': 'sklearn',
    'word2vec': 'SG'
}

# folder_cluster_layout = f"cluster_layout/{config['maze']}/{config['mode']}/rp{config['rp']}_ep{config['ep']}" \
#                         f"_c1_{num_randomwalk_episodes}" \
#                         f"_c2_{second_evolution}({num_saved_from_p1})_rw q updated and epsilon1.0--1.0--" \

folder_cluster_layout = f"/home/xue/projects/masterthesis/cluster_layout/{config['maze']}/{config['mode']}/explore by {explore_mode}/rp{config['rp']}_ep{config['ep']}_mm{max_move_count}" \
                        f"_evo1_{num_randomwalk_episodes}(q_update)" \
                        f"_evo2_{second_evolution}({num_saved_from_p1})_eps(1.0--0.1)x2_lr0.1_gamma0.999_fr10000_gr1000*flags_nr-1" \

# /home/xue/projects/masterthesis/cluster_layout

if not os.path.isdir(folder_cluster_layout):
    makedirs(folder_cluster_layout)

# for ploting
solve_amdp_time_experiments_repetitions = []
simulation_time_experiments_repetitions = []
reward_list_episodes_experiments_repetitions = []
flags_list_episodes_experiments_repetitions = []
move_count_episodes_experiments_repetitions = []
flags_found_order_experiments_repetitions = []
path_episodes_experiments_repetitions = []
epsilon_changing_written = False

fig, axs = plt.subplots(num_of_repetitions, 6, figsize=(6 * 4, num_of_repetitions * 3))
# st = fig.suptitle("curves of each repetition",fontsize=14)
for rep in range(0, num_of_repetitions):

    solve_amdp_time_experiments = []
    simulation_time_experiments = []
    reward_list_episodes_experiments = []
    flags_list_episodes_experiments = []
    move_count_episodes_experiments = []
    path_episodes_experiments = []
    flags_found_order_experiments = []


    # move_count = 0
    # totalMoveCount = 0
    # maxflag = 0
    # maxIndex = 0
    # flagCount = 0
    # reward_list_episodes = []
    # flags_list_episodes = []

    for e in range(0, num_of_experiments):
        print('===================num_of_experiments:', e, "Repetition:", rep)
        start2 = time.time()
        move_count = 0
        totalMoveCount = 0
        maxflag = 0
        maxIndex = 0
        flagCount = 0
        reward_list_episodes = []
        flags_list_episodes = []
        move_count_episodes = []
        flags_found_order_episodes = []
        path_episodes = []
        solve_amdp_time_phases = []
        all_path_lengths = []
        paths_period = []

        epsilons_one_experiment = []
        lr_one_experiment = []
        gamma_one_experiment = []

        agent = WatkinsQLambda(env.size, num_of_actions, env, epsilon, lr, gamma, lam)  ## resets the agent
        gensim_opt = GensimOperator(path_episodes, env)

        print("Begin Training:")
        print("agent.lr:",agent.lr)
        for ep in range(0, num_of_episodes):
            if (ep + 1) % 100 == 0:
                print(f"episode_100: {ep} | avg_move_count: {int(np.mean(move_count_episodes[-100:]))} | "
                      f"avd_reward: {int(np.mean(reward_list_episodes[-100:]))} | "
                      f"env.state: {env.state} | "
                      f"env.flagcollected: {env.flags_collected} | "
                      f"agent.epsilon: {agent.epsilon} | "
                      f"agent.lr: {agent.lr}")

            if ep == num_randomwalk_episodes:

                # print("path_episodes:",path_episodes)
                # min_length_to_save_as_path -= 150
                print("num_overflowed_eps:",num_overflowed_eps)
                print("len of paths_period:", len(paths_period))
                # epsilon_at_first_evo = 1
                agent.epsilon = epsilon_max
                # agent.lr = lr_max
                # agent.gamma =gamma_max
                # saved_paths_randomwalk = sorted(paths_period, key=lambda l: len(l))[:int(0.99*len(paths_period))]
                saved_paths_randomwalk = paths_period
                path_episodes.extend(saved_paths_randomwalk)
                paths_period = []
                # get embedding from gensim and built cluster-layout
                random.shuffle(path_episodes)
                gensim_opt.sentences = path_episodes
                gensim_opt.get_clusterlayout_from_paths(size=config['representation_size'], window=config['window'], clusters=config['kmeans_clusters'][0],
                                                        package=config['package'])
                fpath_cluster_layout = folder_cluster_layout + f"/rep{rep}_s{config['representation_size']}_w{config['window']}" \
                                                               f"_kmeans{config['kmeans_clusters'][0]}_{config['package']}.cluster"
                gensim_opt.write_cluster_layout(fpath_cluster_layout)
                # plot cluster layout
                copy_cluster_layout = copy.deepcopy(gensim_opt.cluster_layout)
                for row in copy_cluster_layout:
                    for index, item in enumerate(row):
                        if row[index].isdigit():
                            row[index] = (int(row[index])+1)*1000
                        else:
                            row[index] = 0
                axs[rep, 3].imshow(np.array(copy_cluster_layout), aspect='auto', cmap=plt.get_cmap("gist_ncar"))
                axs[rep, 3].set_title(f"clusters{config['kmeans_clusters'][0]}s{config['representation_size']}w{config['window']}{config['word2vec']}")

                im=axs[rep, 5].imshow(agent.states_long_life, cmap='hot')
                axs[rep, 5].set_title(f"{env.maze_name}/{explore_mode}")
                fig.colorbar(im,ax=axs[rep, 5])
                fig.show()
                fig.savefig(f"{folder_cluster_layout}/visitation.png", dpi=600, facecolor='w',
                            edgecolor='w',
                            orientation='portrait', format=None,
                            transparent=False, bbox_inches=None, pad_inches=0.1)

                # build and solve AMDP
                amdp = AMDP(env=env, tiling_mode=None, dw_clt_layout=gensim_opt.cluster_layout)
                start1 = time.time()
                amdp.solveAbstraction()
                end1 = time.time()
                solve_amdp_time_phases.append(end1 - start1)

            elif ep == second_evolution:
                # min_length_to_save_as_path -= 200
                # agent.resetQ()

                print("len of paths_period:", len(paths_period))
                # epsilon_at_second_evo = 1
                agent.epsilon = epsilon_max1
                # agent.lr = lr_max
                # agent.gamma =gamma_max
                # saved_paths_period1 = sorted(paths_period, key=lambda l: len(l))[:num_saved_from_p1]
                # saved_paths_period1 = sorted(paths_period, key=lambda l: len(l))[:int(num_saved_from_p1 * len(paths_period))]
                saved_paths_period1 = paths_period
                # path_episodes.extend(saved_paths_period1)
                paths_period = []
                # get embedding from gensim and built cluster-layout
                random.shuffle(path_episodes)
                gensim_opt.sentences = path_episodes
                gensim_opt.get_clusterlayout_from_paths(size=config['representation_size'], window=config['window'], clusters=config['kmeans_clusters'][1],
                                                        package=config['package'])
                fpath_cluster_layout = folder_cluster_layout + f"/rep{rep}_s{config['representation_size']}_w{config['window']}" \
                                                               f"_kmeans{config['kmeans_clusters'][1]}_{config['package']}.cluster"
                gensim_opt.write_cluster_layout(fpath_cluster_layout)
                # plot cluster layout
                copy_cluster_layout = copy.deepcopy(gensim_opt.cluster_layout)
                for row in copy_cluster_layout:
                    for index, item in enumerate(row):
                        if row[index].isdigit():
                            row[index] = (int(row[index]) + 1) * 1000
                        else:
                            row[index] = 0
                axs[rep, 4].imshow(np.array(copy_cluster_layout), aspect='auto', cmap=plt.get_cmap("gist_ncar"))
                axs[rep, 4].set_title(f"clusters{config['kmeans_clusters'][1]}s{config['representation_size']}w{config['window']}{config['word2vec']}")
                # build and solve AMDP
                amdp = AMDP(env=env, tiling_mode=None, dw_clt_layout=np.array(gensim_opt.cluster_layout))
                start1 = time.time()
                amdp.solveAbstraction()
                end1 = time.time()
                solve_amdp_time_phases.append(end1 - start1)

            # Third EVO
            # elif ep == third_evolution:
            #     # min_length_to_save_as_path -= 100
            #
            #     print("len of paths_period:",len(paths_period))
            #     saved_paths_period2 = sorted(paths_period,key=lambda l:len(l))[:int(0.99*len(paths_period))]
            #     path_episodes.extend(saved_paths_period2)
            #     paths_period = []
            #
            #     gensim_opt.sentences = path_episodes
            #     gensim_opt.get_clusterlayout_from_paths(size=64, window=20, clusters=config['kmeans_clusters'][2], package=config['package'])
            #     fpath_cluster_layout = folder_cluster_layout + f"/rep{rep}_s{config['representation_size']}_w{config['window']}" \
            #             f"_kmeans{config['kmeans_clusters'][2]}_{config['package']}.cluster"
            #     gensim_opt.write_cluster_layout(fpath_cluster_layout)
            #
            #     amdp = AMDP(env=env, tiling_mode=None, dw_clt_layout=np.array(gensim_opt.cluster_layout))
            #     start1 = time.time()
            #     amdp.solveAbstraction()
            #     end1 = time.time()
            #     solve_amdp_time_phases.append(end1 - start1)
            last_final = env.state
            env.reset()
            if ep < num_randomwalk_episodes:
                env.state = last_final
            agent.resetEligibility()  # 可以修改


            #=========Here to modify epsilon value:====================
            #$$$scheme1: prefer exploitation a little more$$$
            #~~~for 2 times of evo~~~
            if num_randomwalk_episodes > ep:
                temp_eps = epsilon - (epsilon / num_randomwalk_episodes) * (ep)
                if temp_eps > 0.1:
                    agent.epsilon = round(temp_eps, 5)
                    # agent.epsilon -= epsilon_at_first_evo/(second_evolution-num_randomwalk_episodes)
            if num_randomwalk_episodes <= ep < second_evolution:
                temp_eps = epsilon_max - (epsilon_max / length_of_phase1) * (ep - num_randomwalk_episodes)
                if temp_eps > 0.1:
                    agent.epsilon = round(temp_eps, 5)
                    # agent.epsilon -= epsilon_at_first_evo/(second_evolution-num_randomwalk_episodes)
            if second_evolution <= ep:
                temp_eps = epsilon_max1 - (epsilon_max1 / length_of_phase2) * (ep - second_evolution)
                if temp_eps > 0.1:
                    agent.epsilon = round(temp_eps, 5)
                    # agent.epsilon -= epsilon_at_second_evo / (num_of_episodes - second_evolution)
            #~~~for 1 time of evo~~~
            # if num_randomwalk_episodes <= ep < num_of_episodes:
            #     temp_eps = epsilon_max - (epsilon_max / (num_randomwalk_episodes-num_randomwalk_episodes)) * (ep - num_randomwalk_episodes)
            #     if temp_eps > 0.1:
            #         agent.epsilon = round(temp_eps, 5)

            #$$$scheme2: prefer exploration a little more$$$
            # if num_randomwalk_episodes+(second_evolution-num_randomwalk_episodes)/10 < ep < second_evolution:
            #     agent.epsilon -= epsilon_max/(second_evolution-num_randomwalk_episodes)
            #
            # if ep > second_evolution+(num_of_episodes-second_evolution)/10:
            #     agent.epsilon -= epsilon_max1/(num_of_episodes-second_evolution)

            #=========agent.lr changing=========
            # if num_randomwalk_episodes <= ep < second_evolution:
            #     agent.lr = math.exp(-(ep - num_randomwalk_episodes + lr_func_b)/lr_func_a)
            # if second_evolution <= ep < num_of_episodes:
            #     agent.lr = math.exp(-(ep - second_evolution + lr_func_b)/lr_func_a)

            # =========agent.gamma changing=========
            ## scheme1 : exp curve
            # if num_randomwalk_episodes <= ep < second_evolution:
            #     agent.gamma = math.exp(-(ep - num_randomwalk_episodes + gamma_func_b)/gamma_func_a)
            # if second_evolution <= ep < num_of_episodes:
            #     agent.gamma = math.exp(-(ep - second_evolution + gamma_func_b)/gamma_func_a)
            ## schme2 : exp linear
            # if num_randomwalk_episodes <= ep < second_evolution:
            #     agent.gamma = gamma_max-((gamma_max-gamma_min)/length_of_phase1)*(ep-num_randomwalk_episodes)
            # if second_evolution <= ep:
            #     agent.gamma = gamma_max-((gamma_max-gamma_min)/length_of_phase2)*(ep-second_evolution)

            epsilons_one_experiment.append(agent.epsilon)
            lr_one_experiment.append(agent.lr)
            gamma_one_experiment.append(agent.gamma)


            episode_reward = 0
            move_count = 0
            a = agent.policy_explore_rl(env.state, env.actions(env.state))
            path = [str((env.state[0], env.state[1]))]

            while (not env.isTerminal(env.state) or ep < num_randomwalk_episodes):
            # while not env.isTerminal(env.state):
                # print("env.isTerminal(env.state):",env.isTerminal(env.state))
                move_count += 1

                if ep < num_randomwalk_episodes:
                    if move_count > max_move_count:
                        num_overflowed_eps += 1
                        break

                    # raveled = agent.states_long_life.ravel()
                    # # raveled = raveled[raveled>0]
                    # if len(raveled) > 0:
                    #     samples = np.random.choice(raveled, int((env.size[0] * env.size[1]) / 20), replace=True)
                    #     visits_mean = np.mean(samples)
                    #     if visits_mean > 500:
                    #         print("visits_mean:",visits_mean)
                    #         break

                    # else:
                    new_state = env.step(env.state, a)
                    agent.states_episodic[new_state[0], new_state[1]] += 1
                    agent.states_long_life[new_state[0], new_state[1]] += 1
                    # r = env.reward(env.state, a, new_state)
                    # r1 = math.sqrt(1/agent.states_episodic[env.state[0],env.state[1]])
                    # r2 = math.sqrt(1/agent.states_long_life[env.state[0],env.state[1]])
                    r1 = -math.sqrt(agent.states_long_life[new_state[0], new_state[1]])
                    # r1 = -math.log(agent.states_long_life[new_state[0], new_state[1]])
                    r2 = -agent.states_episodic[new_state[0], new_state[1]]
                    beta = ep/num_randomwalk_episodes
                    r = (1-beta)*r1 + (beta)*r2
                    # r = r1
                    episode_reward += r
                    a_prime = agent.policy_explore_rl(new_state, env.actions(new_state))
                    a_star = agent.policyNoRand_explore_rl(new_state, env.actions(new_state))
                    agent.learn_explore(env.state, a, new_state, a_prime, a_star, r)
                    path.append(str((new_state[0], new_state[1])))

                else:
                    ##Select action using policy
                    abstract_state = amdp.getAbstractState(env.state)
                    new_state = env.step(env.state, a)
                    new_abstract_state = amdp.getAbstractState(new_state)
                    # print(new_state, new_abstract_state)

                    a_prime = agent.policy(new_state, env.actions(new_state))  ##Greedy next-action selected
                    a_star = agent.policyNoRand(new_state, env.actions(new_state))
                    ## Optimal next action ---- comparison of the two required for Watkins Q-lambda

                    r = env.reward(env.state, a, new_state)  ## ground level reward
                    episode_reward += r
                    # if r > 0:
                    #     print("r>0:",r)
                    # if r % 1000 == 0:
                    #     print("hit goal:",r)

                    value_new_abstract_state = amdp.getValue(new_abstract_state)
                    value_abstract_state = amdp.getValue(abstract_state)
                    # print("type(value_abstract_state):",type(value_abstract_state))
                    shaping = gamma * value_new_abstract_state * omega - value_abstract_state * omega
                    agent.learn(env.state, a, new_state, a_prime, a_star, r + shaping)  # 可以修改
                    path.append(str((new_state[0], new_state[1])))

                env.state = new_state
                a = a_prime
            # next steps actions and states set.

            ############# Keep Track of Stuff for each ep ################
            # flagCount += env.flags_collected
            # totalMoveCount += move_count
            reward_list_episodes.append(episode_reward)
            flags_list_episodes.append(env.flags_collected)
            move_count_episodes.append(move_count)
            flags_found_order_episodes.append(env.flags_found_order)
            # if np.random.rand() < 0.5:
            #     path.reverse()
            paths_period.append(path)
            all_path_lengths.append(len(path))
            # if len(path) > min_length_to_save_as_path:
            #     path_episodes.append(path)
        # =====================
        solve_amdp_time_experiments.append(solve_amdp_time_phases)
        reward_list_episodes_experiments.append(reward_list_episodes)
        flags_list_episodes_experiments.append(flags_list_episodes)
        move_count_episodes_experiments.append(move_count_episodes)
        flags_found_order_experiments.append(flags_found_order_episodes)

        path_episodes_experiments.append(path_episodes)

        end2 = time.time()
        simulation_time_experiments.append(end2 - start2)
        print("last state:", env.state)
        print("all_path_lengths:", all_path_lengths)
        print("len of all_path_lengths:", len(all_path_lengths))

        # plot flag collection in one experiment
        plt.rcParams['agg.path.chunksize'] = 10000
        plt.rcParams['lines.linewidth'] = 1
        d = pd.Series(flags_list_episodes)
        print("flags_list_episodes.shape:",np.array(flags_list_episodes).shape)
        movAv = pd.Series.rolling(d, window=int(num_of_episodes / 30), center=False).mean()
        print('type of movAv:', type(movAv))
        axs[rep, 0].plot(np.arange(len(movAv)), movAv, 'k', label=f"learning_rolled")
        axs[rep, 0].set_ylabel("Number of Flags")
        axs[rep, 0].set_xlabel("Episode No.")
        axs[rep, 0].set_title(f"flag curve of exp{e}_rep{rep}")
        axs[rep, 0].grid(True)
        axs[rep, 0].axvspan(0, num_randomwalk_episodes, facecolor='green', alpha=0.5)
        axs[rep, 0].axvspan(num_randomwalk_episodes, second_evolution, facecolor='blue',alpha=0.5)
        axs[rep, 0].axis([0, None, 0, 3.5])
        # axs[rep, 0].legend(loc=2)

        d1 = pd.Series(reward_list_episodes)
        rolled_d1 = pd.Series.rolling(d1, window=int(num_of_episodes/30), center=False).mean()
        # d2 = pd.Series(reward_list_episodes_eva)
        # rolled_d2 = pd.Series.rolling(d2, window=int(num_of_episodes/30), center=False).mean()
        # d1 = np.array(reward_list_episodes)
        # d2 = np.array(reward_list_episodes_eva)
        # axs[rep, 1].plot(np.arange(len(d1)), d1, color='black', alpha=0.25, label=f"learning")
        # axs[rep, 1].plot(np.arange(len(d2)), d2, color='red', alpha=0.25, label=f"evaluation")
        axs[rep, 1].plot(np.arange(len(rolled_d1)), rolled_d1, color='black', alpha=1, label=f"learning_rolled")
        # axs[rep, 1].plot(np.arange(len(rolled_d2)), rolled_d2, color='red', alpha=1, label=f"evaluation_rolled")
        axs[rep, 1].set_ylabel("reward")
        axs[rep, 1].set_xlabel("Episode No.")
        axs[rep, 1].set_title(f"reward curve of exp{e}_rep{rep}")
        axs[rep, 1].grid(True)
        axs[rep, 1].axvspan(0, num_randomwalk_episodes, facecolor='green', alpha=0.5)
        axs[rep, 1].axvspan(num_randomwalk_episodes, second_evolution, facecolor='blue',alpha=0.5)
        axs[rep, 1].axis([0, None, None, None])
        # axs[rep, 1].legend(loc=2)

        d1 = pd.Series(move_count_episodes)
        rolled_d1 = pd.Series.rolling(d1, window=int(num_of_episodes / 30), center=False).mean()
        # d2 = pd.Series(move_count_episodes_eva)
        # rolled_d2 = pd.Series.rolling(d2, window=int(num_of_episodes / 30), center=False).mean()
        # d1 = np.array(move_count_episodes)
        # d2 = np.array(move_count_episodes_eva)
        # axs[rep, 2].plot(np.arange(len(d1)), d1, color='black', alpha=0.25, label=f"learning")
        # axs[rep, 2].plot(np.arange(len(d2)), d2, color='red', alpha=0.25, label=f"evaluation")
        axs[rep, 2].plot(np.arange(len(rolled_d1)), rolled_d1, color='black', alpha=1, label=f"learning_rolled")
        # axs[rep, 2].plot(np.arange(len(rolled_d2)), rolled_d2, color='red', alpha=1, label=f"evaluation_rolled")
        axs[rep, 2].set_ylabel("move_count")
        axs[rep, 2].set_xlabel("Episode No.")
        axs[rep, 2].set_title(f"move_count curve of exp{e}_rep{rep}")
        axs[rep, 2].grid(True)
        axs[rep, 2].axvspan(0, num_randomwalk_episodes, facecolor='green', alpha=0.5)
        axs[rep, 2].axvspan(num_randomwalk_episodes, second_evolution, facecolor='blue', alpha=0.5)
        axs[rep, 2].axis([0, None, None, None])
        # axs[rep, 2].legend(loc=1)


        # len_list0 = [len(x) for x in saved_paths_randomwalk]
        # print("avg and len of random walk period0:", mean(len_list0),len(len_list0))
        # print("max_length, min_length and median of period0:", max(len_list0), min(len_list0), np.median(len_list0))
        # axs[rep, 1].set_title(f"episodes lengths distribution of phase0")
        # axs[rep, 1].hist(len_list0, bins=50, facecolor='green',density=True, alpha=0.5)
        # axs[rep, 1].set_ylabel("Proportion")
        # axs[rep, 1].set_xlabel("Length of episodes")

        # len_list1 = [len(x) for x in saved_paths_period1]
        # print("avg_length and len of period1:", mean(len_list1), len(saved_paths_period1))
        # print("max_length, min_length and median of period1:", max(len_list1), min(len_list1), np.median(len_list1))
        # axs[rep, 2].set_title(f"episodes lengths distribution of phase1")
        # axs[rep, 2].hist(len_list1, bins=50, facecolor='blue',density=True, alpha=0.5)
        # axs[rep, 2].set_ylabel("Proportion")
        # axs[rep, 2].set_xlabel("Length of episodes")
        # =====================
        print("agent.epsilon:", agent.epsilon)

        # if not epsilon_changing_written:
        #     fig1, axs[-1,1] = plt.subplots()
        #     axs[-1,1].plot(np.arange(len(epsilons_one_experiment)),np.array(epsilons_one_experiment), 'k')
        #     axs[-1,1].set_ylabel("epsilon")
        #     axs[-1,1].set_xlabel("epsilons")
        #     axs[-1,1].set_title("agent.epsilon changing")
        #     axs[-1,1].axvspan(0, num_randomwalk_episodes, facecolor='green', alpha=0.5)
        #     axs[-1,1].axvspan(num_randomwalk_episodes, second_evolution, facecolor='blue', alpha=0.5)
        #     epsilon_changing_written = True
        #     fig1.savefig(f"{folder_cluster_layout}/epsilon_changing.png", dpi=1200, facecolor='w', edgecolor='w',
        #                  orientation='portrait', papertype=None, format=None,
        #                  transparent=False, bbox_inches=None, pad_inches=0.1)



    solve_amdp_time_experiments_repetitions.append(solve_amdp_time_experiments)
    simulation_time_experiments_repetitions.append(simulation_time_experiments)
    flags_list_episodes_experiments_repetitions.append(flags_list_episodes_experiments)
    reward_list_episodes_experiments_repetitions.append(reward_list_episodes_experiments)
    move_count_episodes_experiments_repetitions.append(move_count_episodes_experiments)
    path_episodes_experiments_repetitions.append(path_episodes_experiments)
    flags_found_order_experiments_repetitions.append(flags_found_order_experiments)

fig.tight_layout()
fig.show()
fig.savefig(f"{folder_cluster_layout}/flags_collection_of_each_rep.png", dpi=600, facecolor='w', edgecolor='w',
            orientation='portrait', format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1)

print("flags collected in last 200ep of each exp:")
for i in range(len(flags_list_episodes_experiments_repetitions)):
    for j in range(num_of_experiments):
        print(flags_list_episodes_experiments_repetitions[i][j][-200:])
print("flags_list_episodes_experiments_repetitions.shape:", np.array(flags_list_episodes_experiments_repetitions).shape)

print("move count in last 200ep of each exp:")
for i in range(len(move_count_episodes_experiments_repetitions)):
    for j in range(num_of_experiments):
        print(move_count_episodes_experiments_repetitions[i][j][-200:])
print("move_count_episodes_experiments_repetitions.shape:", np.array(move_count_episodes_experiments_repetitions).shape)

print("total move count of :", np.sum(np.array(move_count_episodes_experiments_repetitions)))
# print("order of flags collection:",flags_found_order_experiments_repetitions)


# if not os.path.isfile(fpath_paths):
#     with open(fpath_paths, "w") as f:
#         for repetition in path_episodes_experiments_repetitions:
#             for exp in repetition:
#                 for episode in exp:
#                     if len(episode) > 200:
#                         for coord in episode:
#                             f.write(str(coord).replace(' ', '') + ' ')
#                         f.write('\n')
#     print("file: " + fpath_paths + " is saved")
# else:
#     print("file: " + fpath_paths + " is already there")
# os.chmod(fpath_paths, S_IREAD)
# # print(np.array(path_episodes_experiments_repetitions))
# print("path_episodes_experiments_repetitions.shape: ", np.array(path_episodes_experiments_repetitions).shape)


######################################################
################## Make Directory ####################

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise


######################################################
################## Draw Graph ########################


# labs = ["True", "3x3", "4x4", "5x5", "7x7", "9x9", "10x10", "None"]     # 可修改
labs = ["biased"]  # 可修改
output_dir = folder_cluster_layout

fig, axs = plt.subplots(1, 4, figsize=(5 * 4, 4))

## Reward
plotRewards = np.mean(flags_list_episodes_experiments_repetitions, axis=0)
plotSDs = np.std(flags_list_episodes_experiments_repetitions, axis=0)
print("plotRewards.shape", plotRewards.shape)
print("plotSDs.shape", plotSDs.shape)
plotErrors = plotSDs / np.sqrt(10)
plt.rcParams['agg.path.chunksize'] = 10000
for i in range(0, len(plotRewards)):
    d = pd.Series(plotRewards[i])
    print("d.shape:",d.shape)
    # s = pd.Series(plotErrors[i])
    movAv = pd.Series.rolling(d, window=int(num_of_episodes / 30), center=False).mean()
    print("movAv.shape:",movAv.shape)
    l, caps, c = axs[0].errorbar(np.arange(len(movAv)), movAv, yerr=plotErrors[i], color='black',label=labs[i], capsize=5,
                              errorevery=int(num_of_episodes / 30))
    for cap in caps:
        cap.set_marker("_")
axs[0].set_ylabel("No. Of Flags Collected")
axs[0].set_xlabel("Episode No.")
axs[0].legend(loc=4)
axs[0].grid(True)
axs[0].set_title("flags collection with errorbar")
axs[0].axvspan(0,num_randomwalk_episodes,facecolor='green', alpha=0.5)
axs[0].axvspan(num_randomwalk_episodes,second_evolution, facecolor='blue', alpha=0.5)
axs[0].axis([0, None, 0, 3])
# print(whenConverged)

# with open("{}/resultsListPickle".format(output_dir), 'wb') as p:
#     pickle.dump(toPickle, p)

##plt.title("Number of Episodes: " + str(num_of_episodes) + " Alpha: " + str(lr) + " Gamma: " + str(gamma) + " Lambda: " +str(lam) + " Epsilon: "+str(agent.epsilon))

## move_counts changing
mean_by_rep_move_count = np.mean(move_count_episodes_experiments_repetitions, axis=0)
std_by_rep_move_count = np.std(move_count_episodes_experiments_repetitions, axis=0)
print("mean_by_rep_move_count.shape:", mean_by_rep_move_count.shape)
print("std_by_rep_move_count.shape:", std_by_rep_move_count.shape)
plotErrors = std_by_rep_move_count / np.sqrt(10)
plt.rcParams['agg.path.chunksize'] = 10000
for i in range(0, len(mean_by_rep_move_count)):
    d = pd.Series(mean_by_rep_move_count[i])
    s = pd.Series(plotErrors[i])
    movAv = pd.Series.rolling(d, window=int(num_of_episodes / 30), center=False).mean()
    l, caps, c = axs[1].errorbar(np.arange(len(movAv)), movAv, yerr=plotErrors[i], color='black', capsize=5,
                              errorevery=int(num_of_episodes / 30))
    for cap in caps:
        cap.set_marker("_")
axs[1].set_ylabel("move_count")
axs[1].set_xlabel("Episode No.")
axs[1].grid(True)
axs[1].set_title("move_count with errorbar")
axs[1].axvspan(0,num_randomwalk_episodes,facecolor='green', alpha=0.5)
axs[1].axvspan(num_randomwalk_episodes,second_evolution, facecolor='blue', alpha=0.5)
# axs[1].set(xlim=(0, num_of_episodes))
axs[1].axis([0, None, None, None])


## epsilon changing
axs[2].plot(np.arange(len(epsilons_one_experiment)),np.array(epsilons_one_experiment), 'k', label='epsilon')
axs[2].plot(np.arange(len(lr_one_experiment)),np.array(lr_one_experiment), 'r', label='lr')
axs[2].set_ylabel("epsilon/lr")
axs[2].set_xlabel("episodes")
axs[2].legend(loc=1)
axs[2].set_title("agent.epsilon and lr changing")
axs[2].grid(True)
axs[2].axvspan(0, num_randomwalk_episodes, facecolor='green', alpha=0.5)
axs[2].axvspan(num_randomwalk_episodes, second_evolution, facecolor='blue', alpha=0.5)
# axs[2].set(xlim=(0, num_of_episodes))
axs[2].axis([0, None, 0, None])

# gamma changingabel='epsilon')
axs[3].plot(np.arange(len(gamma_one_experiment)),np.array(gamma_one_experiment), 'r', label='gamma')
axs[3].set_ylabel("gamma")
axs[3].set_xlabel("episodes")
axs[3].legend(loc=1)
axs[3].set_title("agent.gamma changing")
axs[3].grid(True)
axs[3].axvspan(0, num_randomwalk_episodes, facecolor='green', alpha=0.5)
axs[3].axvspan(num_randomwalk_episodes, second_evolution, facecolor='blue', alpha=0.5)
# axs[3].set(xlim=(0, num_of_episodes))
axs[3].axis([0, None, None, None])

fig.tight_layout()
fig.show()
fig.savefig("{}/flagcollection_errorbar_and_epsilon_changing.png".format(output_dir), dpi=600, facecolor='w', edgecolor='w',
            orientation='portrait', format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1)


# ## Flags Collected
# plt.figure(2)
# plotFlags = np.mean(flags_list_episodes_experiments_repetitions, axis=0)
# print("plotFlags.shape:",plotFlags.shape)
# plt.rcParams['agg.path.chunksize'] = 10000
# for i in range(0, len(plotFlags)):
#     d = pd.Series(plotFlags[i])
#     movAv = pd.Series.rolling(d, window=int(num_of_episodes / 30), center=False).mean()
#     plt.plot(np.arange(len(movAv)), movAv, label=labs[i])
# plt.ylabel("Number of Flags")
# plt.xlabel("Episde No.")
# plt.legend(loc=4)
#
# plt.savefig("{}/rewardGraph_noerrorbar.png".format(output_dir), dpi=1200, facecolor='w', edgecolor='w',
#             orientation='portrait', papertype=None, format=None,
#             transparent=False, bbox_inches=None, pad_inches=0.1)

# plt.figure(3)
# plotAbsTimings = np.mean(solve_amdp_time_experiments_repetitions, axis=0)
# for i, v in enumerate(plotAbsTimings):
#     plt.text(i - 0.25, v + 1.5, str(round(np.sum(v), 1)), color='blue', fontweight='bold')
# plt.bar(np.arange(len(plotAbsTimings)), plotAbsTimings)
# plt.xticks(np.arange(len(plotAbsTimings)), labs)
# plt.xlabel("Abstraction Used")
# plt.ylabel("Time Taken")
# plt.title("Time Taken to Solve Each Abstraction")
# # plt.savefig("{}/AbstractionTime.png".format(output_dir), dpi=1200, facecolor='w', edgecolor='w',
# #             orientation='portrait', papertype=None, format=None,
# #             transparent=False, bbox_inches=None, pad_inches=0.1)

# plt.figure(4)
# plotSimTimings = np.mean(simulation_time_experiments_repetitions, axis=0)
# for i, v in enumerate(plotSimTimings):
#     plt.text(i - 0.30, v + 1.5, str(round(v, 1)), color='blue', fontweight='bold')
# plt.bar(np.arange(len(plotSimTimings)), plotSimTimings)
# plt.xticks(np.arange(len(plotSimTimings)), labs)
# plt.xlabel("Experiments")
# plt.ylabel("Time Taken In Seconds")
# plt.title("Time Taken To Simulate each experiment with episodes" + str(num_of_episodes))
# # plt.savefig("{}/SimulationTime.png".format(output_dir), dpi=1200, facecolor='w', edgecolor='w',
# #             orientation='portrait', papertype=None, format=None,
# #             transparent=False, bbox_inches=None, pad_inches=0.1)

