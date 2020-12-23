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
from abstractions.abstraction_new_building import AMDP
from envs.maze_env_general_new_plus_stable import Maze
from RL_brains.RL_brain_fast_explore_new import WatkinsQLambda
from gensim_operations.gensim_operation_online_new import GensimOperator


def function1(maze, big, e_mode, e_start, e_eps, subsample_factor, q_eps, repetitions, mm, rep_size, win_size, w2v, k,
              num_clusters, k_means_pkg, interpreter, output_file):
    # relative path
    folder_cluster_layout = f"./cluster_layout/{maze}_big={big}" \
                            f"/topology{num_clusters} against uniform/rp{repetitions}_{e_start}{e_eps}+{q_eps}_mm{mm}_" \
                            f"ds{subsample_factor}_win{win_size}_rep{rep_size}_{w2v}_{k_means_pkg}_{interpreter}/k[{k}]"
    if not os.path.isdir(folder_cluster_layout):
        makedirs(folder_cluster_layout)
    if output_file == 1:
        sys.stdout = open(f"{folder_cluster_layout}/output.txt", 'w')
        sys.stderr = sys.stdout
    print("PID: ", os.getpid())
    print("folder_cluster_layout:", folder_cluster_layout)

    # for local python interpreter
    # folder_cluster_layout = f"/Users/yuan/PycharmProjects/Masterthesis/cluster_layout/{config['maze']}" \
    #                         f"/{config['mode']}/k{str(config['kmeans_clusters'][:1])}/rp{num_of_repetitions}_ep{num_total_episodes}_mm{max_move_count}" \
    #                         f"_evo1_{num_explore_episodes}_eps(1.0--0.1)_lr0.1_gamma0.999_fr10000_gr1000*flags_nr-1"
    # for remote server interpreter
    # folder_cluster_layout = f"/home/xue/projects/masterthesis/cluster_layout/{config['maze']}" \
    #                         f"/{config['mode']}/k{str(config['kmeans_clusters'][:1])}/rp{num_of_repetitions}_ep{num_total_episodes}_mm{max_move_count}" \
    #                         f"_evo1_{num_explore_episodes}_eps(1.0--0.1)_lr0.1_gamma0.999_fr10000_gr1000*flags_nr-1"

    # from sympy.core.symbol import symbols
    # from sympy.solvers.solveset import nonlinsolve
    # from sympy import exp, solve
    print(f"maze:{maze}, big:{big}, e_mode:{e_mode}, e_start:{e_start}, e_eps:{e_eps}, "
          f"subsample_factor:{subsample_factor}, q_eps:{q_eps}, repetitions:{repetitions}, \n"
          f"mm={mm}, rep_size={rep_size}, win_size={win_size}, w2v = {w2v}, clusters={num_clusters}, k_means_pkg ={k_means_pkg}")
    print("======================================================")
    env = Maze(maze=maze, big=big)  # initialize env 可修改
    print("env.name:", env.maze_name)
    print("env.big:", env.big)
    print("env.flags:", env.flags, env.room_layout[env.flags[0][0], env.flags[0][1]],
          env.room_layout[env.flags[1][0], env.flags[1][1]], env.room_layout[env.flags[2][0], env.flags[2][1]])
    print("env.goal:", env.goal, env.room_layout[env.goal[0], env.goal[1]])
    print("env.state:", env.state)
    # print("env.walls:", env.walls)

    # ==============plotting mazes==============
    # fontsize = 12 if big == 0 else 4.5
    # fontweight = 'semibold'
    # cmap = ListedColormap(["black", "lightgrey", "yellow", "green", "red"])
    # maze_to_plot = np.where(env.room_layout == 'w', 0, 1)
    # maze_to_plot[env.state[0], env.state[1]] = 4
    # maze_to_plot[env.goal[0], env.goal[1]] = 3
    # w, h = figure.figaspect(maze_to_plot)
    # print("w,h:",w,h)
    # fig1, ax1 = plt.subplots(figsize=(w, h))
    # # fig, ax1 = plt.subplots()
    # ax1.text(env.state[1] + 0.5, env.state[0] + 0.55, 'S', ha="center", va="center", color="k", fontsize=fontsize, fontweight=fontweight)
    # ax1.text(env.goal[1] + 0.5, env.goal[0] + 0.55, 'G', ha="center", va="center", color="k", fontsize=fontsize, fontweight=fontweight)
    # for flag in env.flags:
    #     # print(flag)
    #     maze_to_plot[flag[0], flag[1]] = 2
    #     ax1.text(flag[1] + 0.5, flag[0] + 0.55, 'F', ha="center", va="center", color="k", fontsize=fontsize, fontweight=fontweight)
    # # print(maze_to_plot)
    # ax1.pcolor(maze_to_plot, cmap=cmap, vmin=0, vmax=4, edgecolors='k', linewidth=1)
    # ax1.invert_yaxis()
    # ax1.axis('off')
    # fig1.tight_layout()
    # fig1.show()
    # fig1.savefig(f"./img_mazes/{env.maze_name}_big{big}.png", dpi=600, facecolor='w', edgecolor='w',
    #             orientation='portrait', format=None,
    #             transparent=False, bbox_inches=None, pad_inches=0.1)
    # ==============plotting mazes end==============

    # abstraction_mode = [None, ((5,5),(4,4))]   # 可修改
    # abstraction_mode = [None,((19,19),(16,16))]  # 可修改
    a = math.ceil(env.size[0] / np.sqrt(k))
    b = math.ceil(env.size[1] / np.sqrt(k))
    abstraction_mode = [k, (b, a)]  # 可修改
    # abstraction_mode = [None,((21,21),(16,16))]  # 可修改
    # abstraction_mode = [None, ((16,16),(12,12))]   # 可修改
    print("abstraction_mode:", abstraction_mode)
    print("e_mode:", e_mode)
    num_of_actions = 4
    num_of_experiments = len(abstraction_mode)
    lr = 0.1
    lam = 0.9
    gamma = 0.999
    omega = 100
    epsilon_e = 0.01  # probability for choosing random action  #可修改
    epsilon_qmax = 1
    epsilon_qmax2 = 1
    print(f"lr={lr} / lam={lam} / gamma={gamma} / omega={omega} / epsilon_e={epsilon_e} / epsilon_max={epsilon_qmax}"
          f"/ epsilon_max1={epsilon_qmax2}")
    num_explore_episodes = e_eps
    num_total_episodes = num_explore_episodes + q_eps
    # third_evolution = 500 + 1500
    # fourth_evolution = 500 + 1500
    num_saved_from_p1 = 1
    # num_saved_from_p2 = 1500
    # num_total_episodes = num_explore_episodes + 2001        # 可修改
    num_of_repetitions = repetitions  # 可修改
    max_move_count = mm
    num_overflowed_eps = 0
    min_length_to_save_as_path = 500
    length_of_phase1 = num_total_episodes - num_explore_episodes
    # length_of_phase2 = num_total_episodes-second_evolution
    rolling_window_size = int(q_eps / 30)

    config = {
        'maze': maze + "_big=" + str(big),
        'mode': f'topology(RL probe) {num_clusters} against tiling (hard goal)',
        'representation_size': rep_size,
        'window_size': win_size,
        'kmeans_clusters': [k, None, None],
        'package': k_means_pkg,
        'word2vec': w2v,
        'errorbar_yerror_factor': 0.3162  # or 1 or 2 or 1/np.sqrt(10) as 0.3162
    }
    print("config:", config)
    explore_mode = 'rl probe diff starts'
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

    # for ploting
    simulation_time_experiments_repetitions = []
    exploration_time_experiments_repetitions = []
    solve_word2vec_time_experiments_repetitions = []
    solve_amdp_time_experiments_repetitions = []
    solve_q_time_experiments_repetitions = []

    reward_list_episodes_experiments_repetitions = []
    flags_list_episodes_experiments_repetitions = []
    move_count_episodes_experiments_repetitions = []
    flags_found_order_experiments_repetitions = []
    path_episodes_experiments_repetitions = []
    epsilon_changing_written = False
    longlife_exploration_std = []
    longlife_exploration_mean = []
    fig, axs = plt.subplots(num_of_repetitions, 5, figsize=(5 * 5, num_of_repetitions * 4))
    # fig.set_tight_layout(True)
    # st = fig.suptitle("curves of each repetition",fontsize=14)
    for rep in range(0, num_of_repetitions):
        simulation_time_experiments = []
        exploration_time_experiments = []
        solve_word2vec_time_experiments = []
        solve_amdp_time_experiments = []
        solve_q_time_experiments = []

        reward_list_episodes_experiments = []
        reward_list_steps_experiments = []
        flags_list_episodes_experiments = []
        move_count_episodes_experiments = []
        path_episodes_experiments = []
        flags_found_order_experiments = []

        for e in range(0, num_of_experiments):
            print('===================num_of_experiments:', e, "Repetition:", rep)
            start_experiment = time.time()
            start_exploration = time.time()
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
            states_explored = []
            # solve_amdp_time_phases = []
            all_path_lengths = []
            paths_period = []

            epsilons_one_experiment = []
            lr_one_experiment = []
            gamma_one_experiment = []

            agent = WatkinsQLambda(env.size, num_of_actions, env, epsilon_e, lr, gamma, lam)  ## resets the agent
            gensim_opt = GensimOperator(path_episodes, env)

            print("Begin Training:")
            print("agent.lr:", agent.lr)

            env.reset()
            for ep in range(0, num_total_episodes):
                if (ep + 1) % 100 == 0:
                    print(f"episode_100: {ep} | avg_move_count: {int(np.mean(move_count_episodes[-100:]))} | "
                          f"avd_reward: {int(np.mean(reward_list_episodes[-100:]))} | "
                          f"env.state: {env.state} | "
                          f"env.flagcollected: {env.flags_collected} | "
                          f"agent.epsilon: {agent.epsilon} | "
                          f"agent.lr: {agent.lr}")
                # if ep == num_explore_episodes//2:
                #     e_mode = 'SM'
                if ep == num_explore_episodes:
                    end_exploration = time.time()
                    exploration_time = end_exploration - start_exploration
                    print("exploration_time:", exploration_time)
                    # print("path_episodes:",path_episodes)
                    # min_length_to_save_as_path -= 150
                    print("num_overflowed_eps:", num_overflowed_eps)
                    print("len of paths_period:", len(paths_period))
                    # epsilon_at_first_evo = 1
                    agent.epsilon = epsilon_qmax
                    # agent.lr = lr_max
                    # agent.gamma =gamma_max
                    # saved_paths_randomwalk = sorted(paths_period, key=lambda l: len(l))[:int(0.99*len(paths_period))]
                    if isinstance(abstraction_mode[e], int):
                        print("plot heatmap")
                        im = axs[rep, 4].imshow(agent.states_long_life, cmap='hot')
                        axs[rep, 4].set_title(f"{env.maze_name}_big{env.big}/emode:{e_mode}_{e_start}")
                        fig.colorbar(im, ax=axs[rep, 4])
                        # fig.show()
                        print("np.std(agent.states_episodic):",
                              np.std(agent.states_episodic[agent.states_episodic > 0]))
                        longlife_exploration_std.append(np.std(agent.states_long_life[agent.states_long_life > 0]))
                        print("longlife_exploration_std:", longlife_exploration_std)
                        longlife_exploration_mean.append(np.mean(agent.states_long_life[agent.states_long_life > 0]))
                        print("longlife_exploration_mean:", longlife_exploration_mean)
                        print("longlife_exploration_sum:", np.sum(agent.states_long_life[agent.states_long_life > 0]))

                        saved_paths_randomwalk = paths_period
                        path_episodes.extend(saved_paths_randomwalk)
                        paths_period = []
                        # get embedding from gensim and built cluster-layout
                        random.shuffle(path_episodes)
                        gensim_opt.sentences = path_episodes
                        start_word2vec = time.time()
                        gensim_opt.get_clusterlayout_from_paths(size=config['representation_size'],
                                                                window=config['window_size'],
                                                                clusters=config['kmeans_clusters'][0],
                                                                skip_gram=int(config['word2vec'] == 'SG'),
                                                                package=config['package'])
                        end_word2vec = time.time()
                        solve_wor2vec_time = end_word2vec - start_word2vec
                        print(f"solve_wor2vec_time {config['word2vec']}: {solve_wor2vec_time}")
                        fpath_cluster_layout = folder_cluster_layout + f"/rep{rep}_s{config['representation_size']}_w{config['window_size']}" \
                                                                       f"_kmeans{config['kmeans_clusters'][0]}_{config['package']}.cluster"
                        gensim_opt.write_cluster_layout(fpath_cluster_layout)
                        # plot cluster layout
                        copy_cluster_layout = copy.deepcopy(gensim_opt.cluster_layout)
                        for row in copy_cluster_layout:
                            for index, item in enumerate(row):
                                if row[index].isdigit():
                                    row[index] = (int(row[index]) + 1) * 10
                                else:
                                    row[index] = 0
                        axs[rep, 3].imshow(np.array(copy_cluster_layout), aspect='auto', cmap=plt.get_cmap("gist_ncar"))

                        indice_center_clusters = []
                        np_cluster_layout = np.array(gensim_opt.cluster_layout)
                        for i in range(k):
                            coords = np.argwhere(np_cluster_layout == str(i))
                            indice_center_clusters.append(np.mean(coords, axis=0))
                        for index, cen in enumerate(indice_center_clusters):
                            print(index,cen)
                            axs[rep, 3].text(cen[1], cen[0], str(index), horizontalalignment='center', verticalalignment='center', fontsize=13, fontweight='semibold')

                        axs[rep, 3].set_title(
                            f"clusters{config['kmeans_clusters'][0]}mm{max_move_count}s{config['representation_size']}w{config['window_size']}_{config['word2vec']}")
                        fig.show()
                        # fig.savefig(f"{folder_cluster_layout}/visitation.png", dpi=600, facecolor='w',
                        #             edgecolor='w',
                        #             orientation='portrait', format=None,
                        #             transparent=False, bbox_inches=None, pad_inches=0.1)

                        amdp = AMDP(env=env, tiling_mode=None, dw_clt_layout=None, gensim_opt=gensim_opt)
                        start_amdp = time.time()
                        amdp.solve_amdp()
                        end_amdp = time.time()
                        solve_amdp_time = end_amdp - start_amdp
                        print("solve_amdp_time:", solve_amdp_time)
                    else:
                        solve_wor2vec_time = 0
                        amdp = AMDP(env=env, tiling_mode=abstraction_mode[e], dw_clt_layout=None, gensim_opt=None)
                        start_amdp = time.time()
                        amdp.solveAbstraction()
                        end_amdp = time.time()
                        solve_amdp_time = end_amdp - start_amdp
                        print("solve_amdp_time:",solve_amdp_time)
                    start_q_learning = time.time()

                # =========Here to modify epsilon value:====================
                # $$$scheme1: prefer exploitation a little more$$$
                # ~~~for 2 times of evo~~~
                if num_explore_episodes > ep:
                    # temp_eps = epsilon - (epsilon / num_explore_episodes) * (ep)
                    # if temp_eps > 0.01:
                    #     agent.epsilon = round(temp_eps, 5)
                    agent.epsilon = epsilon_e
                if num_explore_episodes <= ep < num_total_episodes:
                    temp_eps = epsilon_qmax - (epsilon_qmax / length_of_phase1) * (ep - num_explore_episodes)
                    if temp_eps > 0.1:
                        agent.epsilon = round(temp_eps, 5)
                # if second_evolution <= ep:
                #     temp_eps = epsilon_max1 - (epsilon_max1 / length_of_phase2) * (ep - second_evolution)
                #     if temp_eps > 0.1:
                #         agent.epsilon = round(temp_eps, 5)
                # agent.epsilon -= epsilon_at_second_evo / (num_total_episodes - second_evolution)
                # ~~~for 1 time of evo~~~
                # if num_explore_episodes <= ep < num_total_episodes:
                #     temp_eps = epsilon_max - (epsilon_max / (num_explore_episodes-num_explore_episodes)) * (ep - num_explore_episodes)
                #     if temp_eps > 0.1:
                #         agent.epsilon = round(temp_eps, 5)

                # $$$scheme2: prefer exploration a little more$$$
                # if num_explore_episodes+(second_evolution-num_explore_episodes)/10 < ep < second_evolution:
                #     agent.epsilon -= epsilon_max/(second_evolution-num_explore_episodes)
                #
                # if ep > second_evolution+(num_total_episodes-second_evolution)/10:
                #     agent.epsilon -= epsilon_max1/(num_total_episodes-second_evolution)

                # =========agent.lr changing=========
                # if num_explore_episodes <= ep < second_evolution:
                #     agent.lr = math.exp(-(ep - num_explore_episodes + lr_func_b)/lr_func_a)
                # if second_evolution <= ep < num_total_episodes:
                #     agent.lr = math.exp(-(ep - second_evolution + lr_func_b)/lr_func_a)

                # =========agent.gamma changing=========
                ## scheme1 : exp curve
                # if num_explore_episodes <= ep < second_evolution:
                #     agent.gamma = math.exp(-(ep - num_explore_episodes + gamma_func_b)/gamma_func_a)
                # if second_evolution <= ep < num_total_episodes:
                #     agent.gamma = math.exp(-(ep - second_evolution + gamma_func_b)/gamma_func_a)
                ## schme2 : exp linear
                # if num_explore_episodes <= ep < second_evolution:
                #     agent.gamma = gamma_max-((gamma_max-gamma_min)/length_of_phase1)*(ep-num_explore_episodes)
                # if second_evolution <= ep:
                #     agent.gamma = gamma_max-((gamma_max-gamma_min)/length_of_phase2)*(ep-second_evolution)

                epsilons_one_experiment.append(agent.epsilon)
                lr_one_experiment.append(agent.lr)
                gamma_one_experiment.append(agent.gamma)

                # states_explored.append(env.state)
                # last_final = random.choice(states_explored)

                if ep < num_explore_episodes:
                    if e_start == 'random':
                        env.reset()
                        start_coord = random.choice(env.valid_nodes)
                        start_state = (start_coord[0], start_coord[1], 0, 0, 0)
                        env.state = start_state
                    elif e_start == 'last':
                        start_state = env.state
                        # print(start_state)
                        env.reset()
                        env.state = start_state
                    elif e_start == 'mix':
                        if ep % 20 == 0:
                            env.reset()
                            start_coord = random.choice(env.valid_nodes)
                            start_state = (start_coord[0], start_coord[1], 0, 0, 0)
                            env.state = start_state
                        else:
                            start_state = env.state
                            # print(start_state)
                            env.reset()
                            env.state = start_state
                    else:
                        raise Exception("Ops, invalide e_start mode")
                else:
                    env.reset()
                # if (ep + 1) % 10 == 0 and ep < num_explore_episodes:
                #     print("np.std(agent.states_episodic):", np.std(agent.states_episodic[agent.states_episodic>0]))
                #     print("np.std(agent.states_long_life):", np.std(agent.states_long_life[agent.states_long_life>0]))
                agent.resetEligibility()  # 可以修改

                episode_reward = 0
                move_count = 0
                if ep < num_explore_episodes:
                    if e_mode == 'RL':
                        a = agent.policy_explore_rl(env.state, env.actions(env.state))
                    if e_mode == 'SM':
                        a = agent.policy_explore2(env.state, env.actions(env.state))
                else:
                    a = agent.policy(env.state, env.actions(env.state))
                path = [str((env.state[0], env.state[1]))]

                while (not env.isTerminal(env.state) or ep < num_explore_episodes):
                    # while not env.isTerminal(env.state):
                    # print("env.isTerminal(env.state):",env.isTerminal(env.state))
                    move_count += 1
                    if ep < num_explore_episodes:
                        if not isinstance(abstraction_mode[e], int):
                            break
                        if move_count > max_move_count:
                            num_overflowed_eps += 1
                            break
                        agent.states_episodic[env.state[0], env.state[1]] += 1
                        agent.states_long_life[env.state[0], env.state[1]] += 1
                        agent.state_actions_long_life[env.state[0], env.state[1], a] -= 1
                        if e_mode == 'RL':
                            new_state = env.step(env.state, a)
                            # agent.states_episodic[new_state[0], new_state[1]] += 1
                            # agent.states_long_life[new_state[0], new_state[1]] += 1
                            # r = env.reward(env.state, a, new_state)
                            # r1 = math.sqrt(1/agent.states_episodic[env.state[0],env.state[1]])
                            # r2 = math.sqrt(1/agent.states_long_life[env.state[0],env.state[1]])
                            r1 = -agent.states_long_life[new_state[0], new_state[1]]
                            # r1 = -math.sqrt(agent.states_long_life[new_state[0], new_state[1]])
                            # r1 = -math.log(agent.states_long_life[new_state[0], new_state[1]])

                            # r2 = -agent.states_episodic[new_state[0], new_state[1]]
                            # beta = ep / num_explore_episodes
                            # r = (1 - beta) * r1 + (beta) * r2
                            # r = (1 - beta) * r2 + (beta) * r1
                            r = r1
                            r *= 10
                            # episode_reward += r
                            a_prime = agent.policy_explore_rl(new_state, env.actions(new_state))
                            a_star = agent.policyNoRand_explore_rl(new_state, env.actions(new_state))
                            # agent.learn_explore(env.state, a, new_state, a_prime, a_star, r)
                            agent.learn_explore_sarsa(env.state, a, new_state, a_prime, a_star, r)
                            # agent.learn_explore_boltz(env.state, a, new_state, a_prime, a_star, r, env.actions(new_state), ep)
                            path.append(str(new_state))
                            env.flags_collected = 0
                        if e_mode == 'SM':
                            # ===softmax exploration policy===
                            new_state = env.step(env.state, a)
                            # agent.states_long_life[new_state[0], new_state[1]] += 1
                            a_prime = agent.policy_explore2(new_state, env.actions(new_state))
                            path.append(str(new_state))
                            env.flags_collected = 0
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
                move_count_episodes.append(move_count-1)
                flags_found_order_episodes.append(env.flags_found_order)
                # if np.random.rand() < 0.5:
                #     path.reverse()
                if subsample_factor == 1:
                    paths_period.append(path)
                else:
                    subsampled = [path[index] for index in sorted(random.sample(range(len(path)), math.floor(len(path) * subsample_factor)))]
                    paths_period.append(subsampled)

                all_path_lengths.append(len(path))
                # if len(path) > min_length_to_save_as_path:
                #     path_episodes.append(path)
            # =====================
            end_experiment = time.time()
            simulation_time_experiments.append(end_experiment - start_experiment)
            solve_amdp_time_experiments.append(solve_amdp_time)
            solve_word2vec_time_experiments.append(solve_wor2vec_time)
            exploration_time_experiments.append(exploration_time)
            solve_q_time_experiments.append(end_experiment - start_q_learning)

            reward_list_episodes_experiments.append(reward_list_episodes)
            flags_list_episodes_experiments.append(flags_list_episodes)
            move_count_episodes_experiments.append(move_count_episodes)
            flags_found_order_experiments.append(flags_found_order_episodes)

            path_episodes_experiments.append(path_episodes)

            print("last state:", env.state)
            print("all_path_lengths:", all_path_lengths)
            print("len of all_path_lengths:", len(all_path_lengths))

            # plot flag collection in one experiment
            plt.rcParams['agg.path.chunksize'] = 10000
            plt.rcParams['lines.linewidth'] = 1
            d = pd.Series(flags_list_episodes[num_explore_episodes:])
            print("flags_list_episodes.shape:", np.array(flags_list_episodes).shape)
            movAv = pd.Series.rolling(d, window=rolling_window_size, center=False).mean()
            print('type of movAv:', type(movAv))
            axs[rep, 0].plot(np.arange(len(movAv)), movAv, label=f"learning_rolled_{e}")
            axs[rep, 0].set_ylabel("Number of Flags")
            axs[rep, 0].set_xlabel("Episode No.")
            axs[rep, 0].set_title(f"flag curve of exp{e}_rep{rep}")
            axs[rep, 0].legend(loc=4)
            axs[rep, 0].grid(True)
            # axs[rep, 0].axvspan(0, num_explore_episodes, facecolor='green', alpha=0.5 / num_of_repetitions)
            # axs[rep, 0].axvspan(num_explore_episodes, second_evolution, facecolor='blue',alpha=0.5/num_of_repetitions)
            axs[rep, 0].axis([0, None, 0, 3.5])
            # axs[rep, 0].legend(loc=2)

            d1 = pd.Series(reward_list_episodes[num_explore_episodes:])
            rolled_d1 = pd.Series.rolling(d1, window=rolling_window_size, center=False).mean()
            # d2 = pd.Series(reward_list_episodes_eva)
            # rolled_d2 = pd.Series.rolling(d2, window=int(num_total_episodes/30), center=False).mean()
            # d1 = np.array(reward_list_episodes)
            # d2 = np.array(reward_list_episodes_eva)
            # axs[rep, 1].plot(np.arange(len(d1)), d1, color='black', alpha=0.25, label=f"learning")
            # axs[rep, 1].plot(np.arange(len(d2)), d2, color='red', alpha=0.25, label=f"evaluation")
            axs[rep, 1].plot(np.arange(len(rolled_d1)), rolled_d1, alpha=1, label=f"learning_rolled_{e}")
            # axs[rep, 1].plot(np.arange(len(rolled_d2)), rolled_d2, color='red', alpha=1, label=f"evaluation_rolled")
            axs[rep, 1].set_ylabel("reward")
            axs[rep, 1].set_xlabel("Episode No.")
            axs[rep, 1].set_title(f"reward curve of exp{e}_rep{rep}")
            axs[rep, 1].legend(loc=4)
            axs[rep, 1].grid(True)
            # axs[rep, 1].axvspan(0, num_explore_episodes, facecolor='green', alpha=0.5 / num_of_repetitions)
            # axs[rep, 1].axvspan(num_explore_episodes, second_evolution, facecolor='blue',alpha=0.5/num_of_repetitions)
            axs[rep, 1].axis([0, None, None, 35000])
            # axs[rep, 1].legend(loc=2)

            d1 = pd.Series(move_count_episodes[num_explore_episodes:])
            rolled_d1 = pd.Series.rolling(d1, window=rolling_window_size, center=False).mean()
            # d2 = pd.Series(move_count_episodes_eva)
            # rolled_d2 = pd.Series.rolling(d2, window=int(num_total_episodes / 30), center=False).mean()
            # d1 = np.array(move_count_episodes)
            # d2 = np.array(move_count_episodes_eva)
            # axs[rep, 2].plot(np.arange(len(d1)), d1, color='black', alpha=0.25, label=f"learning")
            # axs[rep, 2].plot(np.arange(len(d2)), d2, color='red', alpha=0.25, label=f"evaluation")
            axs[rep, 2].plot(np.arange(len(rolled_d1)), rolled_d1, alpha=1, label=f"learning_rolled_{e}")
            # axs[rep, 2].plot(np.arange(len(rolled_d2)), rolled_d2, color='red', alpha=1, label=f"evaluation_rolled")
            axs[rep, 2].set_ylabel("move_count")
            axs[rep, 2].set_xlabel("Episode No.")
            axs[rep, 2].set_title(f"move_count curve of exp{e}_rep{rep}")
            axs[rep, 2].legend(loc=1)
            axs[rep, 2].grid(True)
            # axs[rep, 2].axvspan(0, num_explore_episodes, facecolor='green', alpha=0.5 / num_of_repetitions)
            # axs[rep, 2].axvspan(num_explore_episodes, second_evolution, facecolor='blue', alpha=0.5/num_of_repetitions)
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

        simulation_time_experiments_repetitions.append(simulation_time_experiments)
        exploration_time_experiments_repetitions.append(exploration_time_experiments)
        solve_word2vec_time_experiments_repetitions.append(solve_word2vec_time_experiments)
        solve_amdp_time_experiments_repetitions.append(solve_amdp_time_experiments)
        solve_q_time_experiments_repetitions.append(solve_q_time_experiments)

        flags_list_episodes_experiments_repetitions.append(flags_list_episodes_experiments)
        reward_list_episodes_experiments_repetitions.append(reward_list_episodes_experiments)
        move_count_episodes_experiments_repetitions.append(move_count_episodes_experiments)
        path_episodes_experiments_repetitions.append(path_episodes_experiments)
        flags_found_order_experiments_repetitions.append(flags_found_order_experiments)

    # fig.tight_layout()
    fig.set_tight_layout(True)
    fig.show()
    fig.savefig(f"{folder_cluster_layout}/plots_of_each_rep.png", dpi=600, facecolor='w', edgecolor='w',
                orientation='portrait', format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1)

    print("flags collected in last 200ep of each exp:")
    for i in range(len(flags_list_episodes_experiments_repetitions)):
        for j in range(num_of_experiments):
            print(flags_list_episodes_experiments_repetitions[i][j][-200:])
    print("flags_list_episodes_experiments_repetitions.shape:",
          np.array(flags_list_episodes_experiments_repetitions).shape)

    print("move count in last 200ep of each exp:")
    for i in range(len(move_count_episodes_experiments_repetitions)):
        for j in range(num_of_experiments):
            print(move_count_episodes_experiments_repetitions[i][j][-200:])
    print("move_count_episodes_experiments_repetitions.shape:",
          np.array(move_count_episodes_experiments_repetitions).shape)

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
    # labs = ["16/25", "15x15/12x12"]  # 可修改
    labs = [f"topology-{config['kmeans_clusters'][0]}",
            f"uniform-{abstraction_mode[1][0]}x{abstraction_mode[1][1]}",
            # f"{config['kmeans_clusters'][1]}",
            # f"{abstraction_mode[3][0]}x{abstraction_mode[3][1]}",
            ]

    fmts = ['b-', 'b:', 'r-', 'r:']

    output_dir = folder_cluster_layout

    fig, axs = plt.subplots(1, 2, figsize=(5 * 2, 4 * 1))
    fig.set_tight_layout(True)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    ## Flags
    print("============Flags plotting============")
    # mean_by_rep_flags = np.mean(flags_list_episodes_experiments_repetitions, axis=0)
    # std_by_rep_flags = np.std(flags_list_episodes_experiments_repetitions, axis=0)
    # print("mean_by_rep_flags.shape", mean_by_rep_flags.shape)
    # print("std_by_rep_flags.shape", std_by_rep_flags.shape)
    # # plot_errors = std_by_rep_flags / np.sqrt(10)
    # # plot_errors = std_by_rep_flags * 2
    # plot_errors = std_by_rep_flags * config['errorbar_yerror_factor']
    # plt.rcParams['agg.path.chunksize'] = 10000
    # for i in range(0, len(mean_by_rep_flags)):
    #     d = pd.Series(mean_by_rep_flags[i])
    #     print("d.shape:",d.shape)
    #     s = pd.Series(plot_errors[i])
    #     rolled_d = pd.Series.rolling(d, window=window_size, center=False).mean()
    #     rolled_s = pd.Series.rolling(s, window=window_size, center=False).mean()
    #     print("movAv.shape:",rolled_d.shape)
    #     # l, caps, c = axs[0].errorbar(np.arange(len(rolled_d)), rolled_d, yerr=plot_errors[i], label=labs[i], capsize=5,
    #     #                           errorevery=int(num_total_episodes / 30))
    #     # l, caps, c = axs[0].errorbar(np.arange(len(rolled_d)), rolled_d, yerr=rolled_s, fmt=fmts[i], label=labs[i], capsize=5,
    #     #                              errorevery=int(num_total_episodes / 30))
    #     # for cap in caps:
    #     #     cap.set_marker("_")
    #     axs[0].plot(np.arange(len(rolled_d)), rolled_d, label=labs[i])
    #     axs[0].fill_between(np.arange(len(rolled_d)), rolled_d - rolled_s, rolled_d + rolled_s, alpha=0.25)
    # axs[0].set_ylabel("No. Of Flags Collected")
    # axs[0].set_xlabel("Episode No.")
    # axs[0].legend(loc=4)
    # axs[0].grid(True)
    # axs[0].set_title(f"flags collection over {'big' if env.big==1 else 'small'} {env.maze_name}")
    # axs[0].axvspan(0,num_explore_episodes,facecolor='green', alpha=0.5)
    # # axs[0].axvspan(num_explore_episodes,second_evolution, facecolor='blue', alpha=0.5)
    # axs[0].axis([0, None, 0, 3])

    ## Reward
    print("============Reward plotting============")
    mean_by_rep_reward = np.mean(reward_list_episodes_experiments_repetitions, axis=0)[:, num_explore_episodes:]
    std_by_rep_reward = np.std(reward_list_episodes_experiments_repetitions, axis=0)[:, num_explore_episodes:]
    print("mean_by_rep_reward.shape:", mean_by_rep_reward.shape)
    print("std_by_rep_reward.shape", std_by_rep_reward.shape)
    # print("part of mean_by_rep_reward: \n", pd.DataFrame(mean_by_rep_reward).iloc[:, 500:1000])
    # print("part of std_by_rep_reward: \n", pd.DataFrame(std_by_rep_reward).iloc[:, 500:1000])
    # plot_errors = std_by_rep_reward / np.sqrt(10)
    # plot_errors = std_by_rep_reward * 2
    plot_errors = std_by_rep_reward * config['errorbar_yerror_factor']
    plt.rcParams['agg.path.chunksize'] = 10000
    for i in range(0, len(mean_by_rep_reward)):
        d = pd.Series(mean_by_rep_reward[i])
        s = pd.Series(plot_errors[i])
        rolled_d = pd.Series.rolling(d, window=rolling_window_size, center=False).mean()
        rolled_s = pd.Series.rolling(s, window=rolling_window_size, center=False).mean()
        # print("part of rolled_s: \n", rolled_s[500:1000])
        # l, caps, c = axs[1].errorbar(np.arange(len(rolled_d)), rolled_d, yerr=plot_errors[i], label=labs[i], capsize=5,
        #                              errorevery=int(num_total_episodes / 30))
        # l, caps, c = axs[1].errorbar(np.arange(len(rolled_d)), rolled_d, yerr=rolled_s, fmt=fmts[i], label=labs[i], capsize=5,
        #                              errorevery=int(num_total_episodes / 30))
        # for cap in caps:
        #     cap.set_marker("_")
        axs[0].plot(np.arange(len(rolled_d)), rolled_d, label=labs[i])
        axs[0].fill_between(np.arange(len(rolled_d)), rolled_d - rolled_s, rolled_d + rolled_s, alpha=0.25)
    axs[0].set_ylabel("reward")
    axs[0].set_xlabel("Episode No.")
    axs[0].legend(loc=4)
    axs[0].grid(True)
    # axs[0].set_title(f"reward over {'big' if env.big==1 else 'small'} {env.maze_name}")
    # axs[0].axvspan(0, num_explore_episodes, facecolor='green', alpha=0.5)
    # axs[1].axvspan(num_explore_episodes, second_evolution, facecolor='blue', alpha=0.5)
    # axs[1].set(xlim=(0, num_total_episodes))
    axs[0].axis([0, None, None, 35000])

    ## move_counts changing
    print("============Move_counts plotting============")
    # mean_by_rep_move_count = np.mean(move_count_episodes_experiments_repetitions, axis=0)
    # std_by_rep_move_count = np.std(move_count_episodes_experiments_repetitions, axis=0)
    # print("mean_by_rep_move_count.shape:", mean_by_rep_move_count.shape)
    # print("std_by_rep_move_count.shape:", std_by_rep_move_count.shape)
    # print("part of mean_by_rep_move_count: \n", pd.DataFrame(mean_by_rep_move_count).iloc[:, 500:1000])
    # print("part of std_by_rep_move_count: \n", pd.DataFrame(std_by_rep_move_count).iloc[:, 500:1000])
    # # plot_errors = std_by_rep_move_count / np.sqrt(10)
    # # plot_errors = std_by_rep_move_count * 2
    # plot_errors = std_by_rep_move_count * config['errorbar_yerror_factor']
    # # print("plot_errors 500-1000:", plot_errors[:, 400:800])
    # plt.rcParams['agg.path.chunksize'] = 10000
    # for i in range(0, len(mean_by_rep_move_count)):
    #     d = pd.Series(mean_by_rep_move_count[i])
    #     s = pd.Series(plot_errors[i])
    #     rolled_d = pd.Series.rolling(d, window=window_size, center=False).mean()
    #     rolled_s = pd.Series.rolling(s, window=window_size, center=False).mean()
    #     print("part of rolled_s: \n", rolled_s[500:1000])
    #     # l, caps, c = axs[2].errorbar(np.arange(len(rolled_d)), rolled_d, yerr=plot_errors[i], label=labs[i], capsize=5,
    #     #                           errorevery=int(num_total_episodes / 30))
    #     # l, caps, c = axs[2].errorbar(np.arange(len(rolled_d)), rolled_d, yerr=rolled_s, fmt=fmts[i], label=labs[i], capsize=5,
    #     #                              errorevery=int(num_total_episodes / 30))
    #     # for cap in caps:
    #     #     cap.set_marker("_")
    #
    #     # trying shaded confidence interval
    #     # axs[2].plot(np.arange(len(rolled_d)), rolled_d, label=labs[i])
    #     # axs[2].fill_between(np.arange(len(rolled_d)), rolled_d-rolled_s, rolled_d+rolled_s, alpha=0.5)
    #
    #     axs[2].plot(np.arange(len(rolled_d)), rolled_d, label=labs[i])
    #     axs[2].fill_between(np.arange(len(rolled_d)), rolled_d - rolled_s, rolled_d + rolled_s, alpha=0.25)
    # axs[2].set_ylabel("move_count")
    # axs[2].set_xlabel("Episode No.")
    # axs[2].legend(loc=1)
    # axs[2].grid(True)
    # axs[2].set_title("move_count with errorbar")
    # axs[2].axvspan(0,num_explore_episodes,facecolor='green', alpha=0.5)
    # # axs[2].axvspan(num_explore_episodes,second_evolution, facecolor='blue', alpha=0.5)
    # # axs[2].set(xlim=(0, num_total_episodes))
    # axs[2].axis([0, None, None, None])

    ### reward against time steps
    print("============Reward against time steps plotting============")
    mean_by_rep_reward = np.mean(reward_list_episodes_experiments_repetitions, axis=0)[:, num_explore_episodes:]
    std_by_rep_reward = np.std(reward_list_episodes_experiments_repetitions, axis=0)[:, num_explore_episodes:]
    mean_by_rep_move_count = np.mean(move_count_episodes_experiments_repetitions, axis=0)[:, num_explore_episodes:]
    mean_by_rep_move_count = np.cumsum(mean_by_rep_move_count, axis=1)
    # max_mean_by_rep_move_count = np.amax(mean_by_rep_move_count)
    # print("max_mean_by_rep_move_count:",max_mean_by_rep_move_count)
    print("mean_by_rep_reward.shape:", mean_by_rep_reward.shape)
    print("std_by_rep_reward.shape:", std_by_rep_reward.shape)
    print("mean_by_rep_move_count:", mean_by_rep_move_count.shape)
    # plot_errors = std_by_rep_reward / np.sqrt(10)
    # plot_errors = std_by_rep_reward * 2
    plot_errors = std_by_rep_reward * config['errorbar_yerror_factor']
    plt.rcParams['agg.path.chunksize'] = 10000
    max_steps = 0
    for i in range(0, len(mean_by_rep_reward)):
        d = pd.Series(mean_by_rep_reward[i])
        p = pd.Series(mean_by_rep_move_count[i])
        s = pd.Series(plot_errors[i])
        rolled_d = pd.Series.rolling(d, window=rolling_window_size, center=False).mean()
        rolled_p = pd.Series.rolling(p, window=rolling_window_size, center=False).mean()
        rolled_s = pd.Series.rolling(s, window=rolling_window_size, center=False).mean()
        if rolled_p.max() > max_steps:
            max_steps = rolled_p.max()
        # l, caps, c = axs[3].errorbar(rolled_p, rolled_d, yerr=plot_errors[i], label=labs[i], capsize=5,
        #                              errorevery=int(num_total_episodes / 30))
        # l, caps, c = axs[3].errorbar(rolled_p, rolled_d, yerr=rolled_s, fmt=fmts[i], label=labs[i], capsize=5,
        #                              errorevery=int(num_total_episodes / 30))
        # for cap in caps:
        #     cap.set_marker("_")
        axs[1].plot(rolled_p, rolled_d, label=labs[i])
        axs[1].fill_between(rolled_p, rolled_d - rolled_s, rolled_d + rolled_s, alpha=0.25)
    axs[1].set_ylabel("reward")
    axs[1].set_xlabel("steps")
    axs[1].legend(loc=4)
    axs[1].grid(True)
    axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    # axs[1].set_title(f"reward against steps over {'big' if env.big==1 else 'small'} {env.maze_name}")
    # axs[3].axvspan(0, num_explore_episodes, facecolor='green', alpha=0.5)
    # axs[3].axvspan(num_explore_episodes, second_evolution, facecolor='blue', alpha=0.5)
    # axs[1].set(xlim=(0, num_total_episodes))
    axs[1].axis([0, max_steps, None, 35000])

    # plt.tight_layout()
    fig.show()
    fig.savefig(f"{output_dir}/mean_results_errorbar_yerror*{config['errorbar_yerror_factor']}.png", dpi=600,
                facecolor='w',
                edgecolor='w',
                orientation='portrait', format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1)
    #
    # ### epsilon and lr changing
    # print("============Epsilon and lr changing plotting============")
    # fig, ax1 = plt.subplots()
    # ax1.plot(np.arange(len(epsilons_one_experiment)),np.array(epsilons_one_experiment), 'k', label='epsilon')
    # ax1.plot(np.arange(len(lr_one_experiment)),np.array(lr_one_experiment), 'r', label='lr')
    # ax1.set_ylabel("epsilon/lr")
    # ax1.set_xlabel("episodes")
    # ax1.legend(loc=1)
    # # ax1.set_title("agent's epsilon and lr")
    # ax1.grid(True)
    # ax1.text(0.5, 0.75, f'flags:{str(env.flags)}', horizontalalignment='center', verticalalignment='center', transform=axs[4].transAxes, fontsize=13)
    # ax1.text(0.5, 0.6, f'goal:{str(env.goal)}', horizontalalignment='center', verticalalignment='center', transform=axs[4].transAxes, fontsize=13)
    # ax1.axvspan(0, num_explore_episodes, facecolor='green', alpha=0.5)
    # # axs[4].axvspan(num_explore_episodes, second_evolution, facecolor='blue', alpha=0.5)
    # # axs[4].set(xlim=(0, num_total_episodes))
    # ax1.axis([0, None, 0, None])

    ### time consuming comparison
    mean_by_rep_simulation_time = np.mean(simulation_time_experiments_repetitions, axis=0)
    mean_by_rep_exploration_time = np.mean(exploration_time_experiments_repetitions, axis=0)
    mean_by_rep_word2vec_time = np.mean(solve_word2vec_time_experiments_repetitions, axis=0)
    mean_by_rep_amdp_time = np.mean(solve_amdp_time_experiments_repetitions, axis=0)
    mean_by_rep_q_time = np.mean(solve_q_time_experiments_repetitions, axis=0)
    labels = ['Total', 'Exploration', 'Word2vec', 'AMDP', 'Q-learning']
    data1 = [mean_by_rep_simulation_time[0],
             mean_by_rep_exploration_time[0],
             mean_by_rep_word2vec_time[0],
             mean_by_rep_amdp_time[0],
             mean_by_rep_q_time[0],
             ]
    data1 = [round(item) for item in data1]
    print("data1:",data1)
    data2 = [mean_by_rep_simulation_time[1],
             0,
             0,
             mean_by_rep_amdp_time[1],
             mean_by_rep_q_time[1]
             ]
    data2 = [round(item) for item in data2]
    print("data2:",data2)
    # explode = (0, 0.1, 0, 0)
    # ax1.pie(data1, explode=explode, labels=labels, shadow=True)
    # ax1.axis('equal')
    width = 0.35
    x = np.arange(len(data1))
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax1.annotate('{}'.format(height),
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')

    fig, ax1 = plt.subplots(figsize=(5, 4))
    fig.set_tight_layout(True)
    rects1 = ax1.bar(x - width / 2, data1, width, label='topology')
    rects2 = ax1.bar(x + width / 2, data2, width, label='uniform')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("time taken in sec")
    ax1.legend(loc=9)
    ax1.set_ylim(top=max([data1[0], data2[0]]) * 1.1)
    autolabel(rects1)
    autolabel(rects2)
    fig.show()
    fig.savefig(f"{output_dir}/running_time_comparison.png", dpi=600, facecolor='w',
                edgecolor='w',
                orientation='portrait', format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1)

    width = 0.5
    fig, ax1 = plt.subplots(figsize=(5, 4))
    fig.set_tight_layout(True)
    rects1 = ax1.bar(x, data1, width, label='topology')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("time taken in sec")
    ax1.legend(loc=9)
    ax1.set_ylim(top=data1[0] * 1.1)
    autolabel(rects1)
    fig.show()
    fig.savefig(f"{output_dir}/running_time.png", dpi=600, facecolor='w',
                edgecolor='w',
                orientation='portrait', format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1)
    # upload experiments details to google sheets
    print("============upload experiments details to google sheets============")
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials

    scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

    creds = ServiceAccountCredentials.from_json_keyfile_name("google_creds.json", scope)
    client = gspread.authorize(creds)

    sheet = client.open("experiments_result").sheet1  # Open the spreadhseet
    final_reward_0 = statistics.mean(mean_by_rep_reward[0][-rolling_window_size:])
    total_steps_0 = mean_by_rep_move_count[0][-1]
    final_reward_1 = statistics.mean(mean_by_rep_reward[1][-rolling_window_size:])
    total_steps_1 = mean_by_rep_move_count[1][-1]

    insert_row_0 = [env.maze_name, big, abstraction_mode[0], e_mode, e_start, e_eps, mm, subsample_factor, data1[1],
                   round(np.mean(longlife_exploration_mean), 2), round(np.mean(longlife_exploration_std), 2),
                   rep_size, win_size, w2v, 5, data1[2], k_means_pkg, k, data1[3], q_eps, data1[4], repetitions,
                    interpreter, data1[0], round(final_reward_0,2), total_steps_0, lr, gamma, lam, omega, epsilon_e,
                   "1-0.1", config["errorbar_yerror_factor"], folder_cluster_layout]

    insert_row_1 = [env.maze_name, big, str(abstraction_mode[1]), "None", "None", "None", "None", "None", data2[1],
                    "None" ,"None",
                    "None", "None", "None", 5, data2[2], "None", "None", data2[3], q_eps, data2[4], repetitions,
                    interpreter, data2[0], round(final_reward_1,2), total_steps_1, lr, gamma, lam, omega, "None",
                    "1-0.1", config["errorbar_yerror_factor"], folder_cluster_layout]
    sheet.append_row(insert_row_0)
    sheet.append_row(insert_row_1)
    print("uploaded to google sheet")
    print(" FINISHED!")
    if output_file == 1:
        sys.stdout.close()


if __name__ == "__main__":
    maze = 'low_connectivity2'  # low_connectivity2/external_maze21x21_1/external_maze31x31_2/strips2/spiral/basic
    big = 1
    e_mode = 'RL'   # 'RL' or 'SM'
    e_start = 'last'   # 'random' or 'last' or 'mix'
    e_eps = 5000
    mm = 100
    subsample_factor = 0.5

    q_eps = 1000
    repetitions = 5
    rep_size = 64
    win_size = 40
    w2v = 'SG'  # 'SG' or 'CBOW'
    # clusters = [9, 16, 25, 36]     # number of abstract states for Uniform will be matched with the number of clusters
    clusters = [9]  # number of abstract states for Uniform will be matched with the number of clusters
    k_means_pkg = 'sklearn'    # 'sklearn' or 'nltk'
    interpreter = 'R'     # L or R
    output_file = 1

    for i in range(len(clusters)):
        function1(maze=maze, big=big, e_mode=e_mode, e_start=e_start, e_eps=e_eps, subsample_factor=subsample_factor,
                  q_eps=q_eps, repetitions=repetitions, mm=mm, rep_size=rep_size, win_size=win_size, w2v=w2v,
                  k=clusters[i], num_clusters=clusters, k_means_pkg=k_means_pkg, interpreter=interpreter, output_file=output_file)
