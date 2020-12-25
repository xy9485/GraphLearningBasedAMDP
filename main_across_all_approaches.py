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

        self.fig_time_consumption, self.ax_time_consumption = plt.subplots(figsize=(5, 4))
        self.fig_time_consumption.set_tight_layout(True)
        self.current_approach_time_consumption = self.num_approaches
        self.highest_bar_height = 0

    # def _initialize_plot_for_each_rep(self, num_of_repetitions):
    #     self.fig_each_rep, self.axs_each_rep = plt.subplots(num_of_repetitions, 5,
    #                                                         figsize=(5 * 5, num_of_repetitions * 4))
    #     self.fig_each_rep.set_tight_layout(True)

    def plot_each_heatmap(self, agent_e, rep, ax_title):
        # plot heatmap
        im = self.axs_each_rep[rep, 4].imshow(agent_e.states_long_life, cmap='hot')
        self.fig_each_rep.colorbar(im, ax=self.axs_each_rep[rep, 4])
        self.axs_each_rep[rep, 4].set_title(ax_title)
        self.fig_each_rep.show()

    def plot_each_cluster_layout(self, gensim_opt, num_clusters, rep, ax_title, plot_label=1):
        copy_cluster_layout = copy.deepcopy(gensim_opt.cluster_layout)
        for row in copy_cluster_layout:
            for index, item in enumerate(row):
                if row[index].isdigit():
                    row[index] = (int(row[index]) + 1) * 10
                else:
                    row[index] = 0
        self.axs_each_rep[rep, 3].imshow(np.array(copy_cluster_layout), aspect='auto', cmap=plt.get_cmap("gist_ncar"))

        if plot_label == 1:
            indice_center_clusters = []
            np_cluster_layout = np.array(gensim_opt.cluster_layout)
            for i in range(num_clusters):
                coords = np.argwhere(np_cluster_layout == str(i))
                indice_center_clusters.append(np.mean(coords, axis=0))
            for index, cen in enumerate(indice_center_clusters):
                print(index, cen)
                self.axs_each_rep[rep, 3].text(cen[1], cen[0], str(index), horizontalalignment='center', verticalalignment='center',
                                 fontsize=13, fontweight='semibold')

        self.axs_each_rep[rep, 3].set_title(ax_title)
        self.fig_each_rep.show()

    def plot_each_flag_reward_movecount(self, flags_episodes, reward_episodes, move_count_episodes, rep, plot_label):
        rolling_window_size = int(len(flags_episodes)/30)

        d1 = pd.Series(flags_episodes)
        print("flags_list_episodes.shape:", np.array(flags_episodes).shape)
        rolled_d = pd.Series.rolling(d1, window=rolling_window_size, center=False).mean()
        print('type of movAv:', type(rolled_d))

        self.axs_each_rep[rep, 0].plot(np.arange(len(rolled_d)), rolled_d, label=f"learning_rolled_{plot_label}")
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
        self.axs_each_rep[rep, 1].plot(np.arange(len(rolled_d1)), rolled_d1, alpha=1, label=f"learning_rolled_{plot_label}")
        self.axs_each_rep[rep, 1].set_ylabel("reward")
        self.axs_each_rep[rep, 1].set_xlabel("Episode No.")
        self.axs_each_rep[rep, 1].set_title(f"reward curve of rep{rep}")
        self.axs_each_rep[rep, 1].legend(loc=4)
        self.axs_each_rep[rep, 1].grid(True)
        # axs[rep, 1].axvspan(0, num_explore_episodes, facecolor='green', alpha=0.5 / num_of_repetitions)
        # axs[rep, 1].axvspan(num_explore_episodes, second_evolution, facecolor='blue',alpha=0.5/num_of_repetitions)
        self.axs_each_rep[rep, 1].axis([0, None, None, 35000])

        d1 = pd.Series(move_count_episodes)
        rolled_d1 = pd.Series.rolling(d1, window=rolling_window_size, center=False).mean()
        self.axs_each_rep[rep, 2].plot(np.arange(len(rolled_d1)), rolled_d1, alpha=1, label=f"learning_rolled_{plot_label}")
        self.axs_each_rep[rep, 2].set_ylabel("move_count")
        self.axs_each_rep[rep, 2].set_xlabel("Episode No.")
        self.axs_each_rep[rep, 2].set_title(f"move_count curve of rep{rep}")
        self.axs_each_rep[rep, 2].legend(loc=1)
        self.axs_each_rep[rep, 2].grid(True)
        # axs[rep, 2].axvspan(0, num_explore_episodes, facecolor='green', alpha=0.5 / num_of_repetitions)
        # axs[rep, 2].axvspan(num_explore_episodes, second_evolution, facecolor='blue', alpha=0.5/num_of_repetitions)
        self.axs_each_rep[rep, 2].axis([0, None, None, None])

        self.fig_each_rep.show()

    def plot_mean_performance_across_reps(self, flags_episodes_repetitions, reward_episodes_repetitions,
                                          move_count_episodes_repetitions,
                                          curve_label, ax_title=None):
        self.current_approach_mean_performance -= 1
        rolling_window_size = int(len(flags_episodes_repetitions[0])/30)

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
        self.axs_mean_performance[0].plot(np.arange(len(rolled_d)), rolled_d, label=curve_label)
        self.axs_mean_performance[0].fill_between(np.arange(len(rolled_d)), rolled_d - rolled_s, rolled_d + rolled_s, alpha=0.25)
        if self.current_approach_mean_performance == 0:
            self.axs_mean_performance[0].set_ylabel("No. Of Flags Collected")
            self.axs_mean_performance[0].set_xlabel("Episode No.")
            self.axs_mean_performance[0].legend(loc=4)
            self.axs_mean_performance[0].grid(True)
            # self.axs_mean_performance[0].set_title(ax_title)
            # self.axs_mean_performance[0].axvspan(0, num_explore_episodes, facecolor='green', alpha=0.5)
            # axs[0].axvspan(num_explore_episodes,second_evolution, facecolor='blue', alpha=0.5)
            self.axs_mean_performance[0].axis([0, None, 0, 3.5])


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
        self.axs_mean_performance[1].plot(np.arange(len(rolled_d)), rolled_d, label=curve_label)
        self.axs_mean_performance[1].fill_between(np.arange(len(rolled_d)), rolled_d - rolled_s, rolled_d + rolled_s, alpha=0.25)
        if self.current_approach_mean_performance == 0:
            self.axs_mean_performance[1].set_ylabel("reward")
            self.axs_mean_performance[1].set_xlabel("Episode No.")
            self.axs_mean_performance[1].legend(loc=4)
            self.axs_mean_performance[1].grid(True)
            # axs[0].set_title(ax_title)
            # axs[0].axvspan(0, num_explore_episodes, facecolor='green', alpha=0.5)
            # axs[1].axvspan(num_explore_episodes, second_evolution, facecolor='blue', alpha=0.5)
            # axs[1].set(xlim=(0, num_total_episodes))
            self.axs_mean_performance[1].axis([0, None, None, 35000])

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
        self.axs_mean_performance[2].plot(rolled_p, rolled_d, label=curve_label)
        self.axs_mean_performance[2].fill_between(rolled_p, rolled_d - rolled_s, rolled_d + rolled_s, alpha=0.25)
        if rolled_p.max() > self.max_steps:
            self.max_steps = rolled_p.max()
        if self.current_approach_mean_performance == 0:
            self.axs_mean_performance[2].set_ylabel("reward")
            self.axs_mean_performance[2].set_xlabel("steps")
            self.axs_mean_performance[2].legend(loc=4)
            self.axs_mean_performance[2].grid(True)
            self.axs_mean_performance[2].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            # axs[1].set_title(f"reward against steps over {'big' if env.big==1 else 'small'} {env.maze_name}")
            # axs[3].axvspan(0, num_explore_episodes, facecolor='green', alpha=0.5)
            # axs[3].axvspan(num_explore_episodes, second_evolution, facecolor='blue', alpha=0.5)
            # axs[1].set(xlim=(0, num_total_episodes))
            self.axs_mean_performance[2].axis([0, self.max_steps * 1.05, None, 35000])

    def plot_mean_time_comparison(self, experiment_time_repetitions, solve_amdp_time_repetitions,
                                  ground_learning_time_repetitions, exploration_time_repetitions=[0],
                                  solve_word2vec_time_repetitions=[0], bar_label=None):
        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                self.ax_time_consumption.annotate('{}'.format(height),
                             xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom')

        mean_by_rep_experiment_time = np.mean(experiment_time_repetitions)
        mean_by_rep_exploration_time = np.mean(exploration_time_repetitions)
        mean_by_rep_word2vec_time = np.mean(solve_word2vec_time_repetitions)
        mean_by_rep_amdp_time = np.mean(solve_amdp_time_repetitions)
        mean_by_rep_q_time = np.mean(ground_learning_time_repetitions)
        labels = ['Total', 'Exploration', 'Word2vec', 'AMDP', 'Q-learning']
        data = [mean_by_rep_experiment_time,
                mean_by_rep_exploration_time,
                mean_by_rep_word2vec_time,
                mean_by_rep_amdp_time,
                mean_by_rep_q_time]
        data = [round(item) for item in data]
        print("data:", data)
        width = 0.35
        x = np.arange(len(data))
        if self.num_approaches == 3:
            if self.current_approach_time_consumption == 3:
                rects = self.ax_time_consumption.bar(x - width, data, width, label=bar_label)
            elif self.current_approach_time_consumption == 2:
                rects = self.ax_time_consumption.bar(x, data, width, label=bar_label)
            elif self.current_approach_time_consumption == 1:
                rects = self.ax_time_consumption.bar(x + width, data, width, label=bar_label)
        elif self.num_approaches == 2:
            if self.current_approach_time_consumption == 2:
                rects = self.ax_time_consumption.bar(x - width/2, data, width, label=bar_label)
            elif self.current_approach_time_consumption == 1:
                rects = self.ax_time_consumption.bar(x + width / 2, data, width, label=bar_label)
        elif self.num_approaches == 1:
            rects = self.ax_time_consumption.bar(x, data, width, label=bar_label)

        autolabel(rects)
        if data[0] > self.highest_bar_height:
            self.highest_bar_height = data[0]
        self.current_approach_time_consumption -= 1

        if self.current_approach_time_consumption == 0:
            self.ax_time_consumption.set_xticks(x)
            self.ax_time_consumption.set_xticklabels(labels)
            self.ax_time_consumption.set_ylabel("time taken in sec")
            self.ax_time_consumption.legend(loc=9)
            self.ax_time_consumption.set_ylim(top=self.highest_bar_height * 1.1)

        self.fig_time_consumption.show()

class ExperimentMaker:

    def __init__(self, env_name, big, q_eps, interpreter, print_to_file, plot_maker: PlotMaker):
        self.env_name = env_name
        self.big = big

        self.interpreter = interpreter
        self.print_to_file = print_to_file
        self.env = Maze(maze=self.env_name, big=self.big)
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

    def _build_and_solve_amdp(self, tiling_size: tuple = None, gensim_opt: GensimOperator_Topology = None, general=0):
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
        start_amdp = time.time()
        amdp.solve_amdp()
        end_amdp = time.time()
        solve_amdp_time = end_amdp - start_amdp
        print("solve_amdp_time:", solve_amdp_time)
        self.solve_amdp_time_repetitions.append(solve_amdp_time)
        print("-----Finish build and solve amdp-----")
        return amdp

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

    def _results_upload(self):
        print("============upload experiments details to google sheets============")
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials

        scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
                 "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

        creds = ServiceAccountCredentials.from_json_keyfile_name("google_creds.json", scope)
        client = gspread.authorize(creds)
        # sheet = client.open("experiments_result").sheet1  # Open the spreadhseet
        sheet = client.open("experiments_result").worksheet("Sheet1")
        # gspread api to get worksheet
        ### worksheet = sh.get_worksheet(0)
        ### worksheet = sh.worksheet("January")

        mean_by_rep_experiment_time = np.mean(self.experiment_time_repetitions)
        if isinstance(self, UniformExpMaker):
            mean_by_rep_exploration_time = 0
            mean_by_rep_word2vec_time = 0
        else:
            mean_by_rep_exploration_time = np.mean(self.exploration_time_repetitions)
            mean_by_rep_word2vec_time = np.mean(self.solve_word2vec_time_repetitions)
        mean_by_rep_amdp_time = np.mean(self.solve_amdp_time_repetitions)
        mean_by_rep_q_time = np.mean(self.ground_learning_time_repetitions)
        data = [mean_by_rep_experiment_time,
                mean_by_rep_exploration_time,
                mean_by_rep_word2vec_time,
                mean_by_rep_amdp_time,
                mean_by_rep_q_time]
        data = [round(item, 1) for item in data]

        rolling_window_size = int(self.ground_learning_config['q_eps'] / 50)
        final_reward = statistics.mean(np.mean(self.reward_episodes_repetitions, axis=0)[-rolling_window_size:])
        total_steps = np.cumsum(np.mean(self.move_count_episodes_repetitions, axis=0))[-1]

        if isinstance(self, UniformExpMaker):
            abstraction_mode = 'uniform'
            insert_row = [self.env.maze_name, big, abstraction_mode, '--', '--', '--', '--', '--', data[1],
                          '--', '--',
                          '--', '--', '--', '--', data[2], '--', '--',
                          data[3], self.ground_learning_config['q_eps'], data[4], self.repetitions, self.interpreter, data[0],
                          round(final_reward, 2), total_steps, self.ground_learning_config['lr'], self.ground_learning_config['gamma'],
                          self.ground_learning_config['lambda'], self.ground_learning_config['omega'], '--',
                          "1-0.1", self.plot_maker.std_factor, self.path_results]
        else:
            w2v = 'SG' if self.w2v_config['sg'] == 1 else 'CBOW'
            negative = 5
            if isinstance(self, TopologyExpMaker):
                abstraction_mode = 'topology'
            elif isinstance(self, GeneralExpMaker):
                abstraction_mode = 'general'
            insert_row = [self.env.maze_name, big, abstraction_mode, e_mode, e_start, e_eps, mm,self.explore_config['ds_factor'], data[1],
                          round(np.mean(self.longlife_exploration_mean_repetitions), 2),round(np.mean(self.longlife_exploration_std_repetitions), 2),
                          self.w2v_config['rep_size'], self.w2v_config['win_size'], w2v, negative, data[2], self.k_means_pkg, self.num_clusters,
                          data[3], self.ground_learning_config['q_eps'], data[4], self.repetitions, self.interpreter, data[0],
                          round(final_reward, 2), total_steps, self.ground_learning_config['lr'], self.ground_learning_config['gamma'],
                          self.ground_learning_config['lambda'], self.ground_learning_config['omega'], self.explore_config['epsilon_e'],
                          "1-0.1", self.plot_maker.std_factor, self.path_results]
        sheet.append_row(insert_row)
        print("uploaded to google sheet")
        print(" FINISHED!")

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
                start_coord = random.choice(env.valid_coords)
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
                    r = r1
                    r *= 10
                    a_prime = agent_e.policy_explore_rl(new_state, env.actions(new_state))
                    a_star = agent_e.policyNoRand_explore_rl(new_state, env.actions(new_state))
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
        print("len of self.sentences_period:", len(self.sentences_period))

        print("-----Finish Exploration-----")
        return agent_e

    def _w2v_and_kmeans(self, rep):
        print("-----Begin w2v and k-means-----")
        gensim_opt = GensimOperator_Topology(self.env)
        start_word2vec = time.time()
        gensim_opt.get_cluster_layout(sentences=self.sentences_collected,
                                      size=self.w2v_config['rep_size'],
                                      window=self.w2v_config['win_size'],
                                      clusters=self.num_clusters,
                                      skip_gram=self.w2v_config['sg'],
                                      workers=self.w2v_config['workers'],
                                      package=self.k_means_pkg)
        end_word2vec = time.time()
        solve_wor2vec_time = end_word2vec - start_word2vec
        print(f"solve_wor2vec_time: {solve_wor2vec_time}")
        self.solve_word2vec_time_repetitions.append(solve_wor2vec_time)

        fpath_cluster_layout = self.path_results + f"/rep{rep}_s{self.w2v_config['rep_size']}_w{self.w2v_config['win_size']}" \
                                                   f"_kmeans{self.num_clusters}_{self.k_means_pkg}.cluster"
        gensim_opt.write_cluster_layout(fpath_cluster_layout)

        print("-----Finish w2v and k-means-----")
        return gensim_opt

    # def _build_and_solve_amdp(self, gensim_opt: GensimOperator_Topology):
    #     print("-----Begin build and solve amdp-----")
    #     amdp = AMDP_Topology_Uniform(env=self.env, uniform_mode=None, gensim_opt=gensim_opt)
    #     start_amdp = time.time()
    #     amdp.solve_amdp()
    #     end_amdp = time.time()
    #     solve_amdp_time = end_amdp - start_amdp
    #     print("solve_amdp_time:", solve_amdp_time)
    #     self.solve_amdp_time_repetitions.append(solve_amdp_time)
    #     print("-----Finish build and solve amdp-----")
    #     return amdp

    def run(self):
        if not os.path.isdir(self.path_results):
            makedirs(self.path_results)
        if self.print_to_file == 1:
            sys.stdout = open(f"{self.path_results}/output.txt", 'w')
            sys.stderr = sys.stdout

        self._print_before_start()

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
            ax_title = f"{self.env.maze_name}_big{self.env.big}/emode:{self.explore_config['e_mode']}/estart:{self.explore_config['e_start']}"
            self.plot_maker.plot_each_heatmap(agent_e, rep, ax_title)

            # solve w2v and k-means to get clusters and save cluster file
            self.sentences_collected.extend(self.sentences_period)
            self.sentences_period = []
            random.shuffle(self.sentences_collected)
            gensim_opt = self._w2v_and_kmeans(rep)

            # plot cluster layout
            ax_title = f"clusters{self.num_clusters}mm{self.explore_config['max_move_count']}s" \
                       f"{self.w2v_config['rep_size']}w{self.w2v_config['win_size']}_sg{self.w2v_config['sg']}"
            self.plot_maker.plot_each_cluster_layout(gensim_opt, self.num_clusters, rep, ax_title, plot_label=1)

            # build and solve amdp
            amdp = self._build_and_solve_amdp(gensim_opt=gensim_opt)

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
                                                            rep, 'topology')

            # save performance of each rep
            self.flags_episodes_repetitions.append(self.flags_episodes)
            self.reward_episodes_repetitions.append(self.reward_episodes)
            self.move_count_episodes_repetitions.append(self.move_count_episodes)

        # plot mean performance among all reps
        ### ax_title = f"flags collection in {'big' if self.big==1 else 'small'} {self.env.maze_name}"
        curve_label = f"topology-{self.num_clusters}"
        sliced_f_ep_rep = np.array(self.flags_episodes_repetitions)[:, self.explore_config['e_eps']:]
        sliced_r_ep_rep = np.array(self.reward_episodes_repetitions)[:, self.explore_config['e_eps']:]
        sliced_m_ep_rep = np.array(self.move_count_episodes_repetitions)[:, self.explore_config['e_eps']:]
        plot_maker.plot_mean_performance_across_reps(sliced_f_ep_rep, sliced_r_ep_rep, sliced_m_ep_rep, curve_label)

        plot_maker.plot_mean_time_comparison(self.experiment_time_repetitions, self.solve_amdp_time_repetitions,
                                             self.ground_learning_time_repetitions, self.exploration_time_repetitions,
                                             self.solve_word2vec_time_repetitions, bar_label='topology')

        self._results_upload()
        if self.print_to_file == 1:
            sys.stdout.close()

class UniformExpMaker(ExperimentMaker):
    def __init__(self, env_name: str, big: int, tiling_size: tuple, q_eps: int, repetitions: int, interpreter: str,
                 print_to_file: int, plot_maker: PlotMaker, path_results: str):

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
        print("+++++++++++start TopologyExpMaker.run()+++++++++++")
        print("PID: ", os.getpid())
        print("=path_results=:", self.path_results)
        print(f"maze:{self.env_name} | big:{self.big} | repetitions:{self.repetitions} | interpreter:{self.interpreter} |"
              f"print_to_file: {self.print_to_file}")
        print(f"=explore_config=: Nothing to show")
        print(f"=w2v_config=: Nothing to show")
        print(f"=ground_learning_config=: {self.ground_learning_config}")


    # def _build_and_solve_amdp(self, tiling_size: tuple):
    #     print("-----Begin build and solve amdp-----")
    #     amdp = AMDP_Topology_Uniform(env=self.env, uniform_mode=tiling_size, gensim_opt=None)
    #     start_amdp = time.time()
    #     amdp.solve_amdp()
    #     end_amdp = time.time()
    #     solve_amdp_time = end_amdp - start_amdp
    #     print("solve_amdp_time:", solve_amdp_time)
    #     self.solve_amdp_time_repetitions.append(solve_amdp_time)
    #     print("-----Finish build and solve amdp-----")
    #     return amdp

    def run(self):
        if not os.path.isdir(self.path_results):
            makedirs(self.path_results)
        if self.print_to_file == 1:
            sys.stdout = open(f"{self.path_results}/output.txt", 'w')
            sys.stderr = sys.stdout
        self._print_before_start()

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
            amdp = self._build_and_solve_amdp(tiling_size=self.tiling_size)

            # ground learning
            self._ground_learning(amdp)

            # experiment timing ends and saved
            end_experiment = time.time()
            experiment_time = end_experiment-start_experiment
            self.experiment_time_repetitions.append(experiment_time)

            # plot flags, reward, move_count for each rep
            self.plot_maker.plot_each_flag_reward_movecount(self.flags_episodes,
                                                            self.reward_episodes,
                                                            self.move_count_episodes,
                                                            rep, 'uniform')

            # save performance of each rep
            self.flags_episodes_repetitions.append(self.flags_episodes)
            self.reward_episodes_repetitions.append(self.reward_episodes)
            self.move_count_episodes_repetitions.append(self.move_count_episodes)

        # plot mean performance among all reps
        # ax_title = f"flags collection in {'big' if self.big == 1 else 'small'} {self.env.maze_name}"
        curve_label = f"uniform-{self.tiling_size[0]}x{self.tiling_size[1]}"
        plot_maker.plot_mean_performance_across_reps(self.flags_episodes_repetitions,
                                                     self.reward_episodes_repetitions,
                                                     self.move_count_episodes_repetitions, curve_label)

        plot_maker.plot_mean_time_comparison(self.experiment_time_repetitions, self.solve_amdp_time_repetitions,
                                             self.ground_learning_time_repetitions, bar_label='uniform')

        self._results_upload()
        if self.print_to_file == 1:
            sys.stdout.close()

class GeneralExpMaker(ExperimentMaker):
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
        agent_e = ExploreStateBrain(env=self.env, explore_config=self.explore_config)
        env = self.env
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
                start_state = random.choice(env.valid_states)
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
            else:
                down_sampled = [track[index] for index in sorted(random.sample(range(len(track)),
                                math.floor(len(track) * self.explore_config['ds_factor'])))]
                self.sentences_period.append(down_sampled)

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
        print("len of self.sentences_period:", len(self.sentences_period))

        print("-----Finish Exploration-----")
        return agent_e

    def _w2v_and_kmeans(self):
        print("-----Begin w2v and k-means-----")
        gensim_opt = GensimOperator_General(self.env)
        start_word2vec = time.time()
        gensim_opt.get_cluster_labels(sentences=self.sentences_collected,
                                                 size=self.w2v_config['rep_size'],
                                                 window=self.w2v_config['win_size'],
                                                 clusters=self.num_clusters,
                                                 skip_gram=self.w2v_config['sg'],
                                                 workers=self.w2v_config['workers'],
                                                 package=self.k_means_pkg)
        end_word2vec = time.time()
        solve_wor2vec_time = end_word2vec - start_word2vec
        print(f"solve_wor2vec_time: {solve_wor2vec_time}")
        self.solve_word2vec_time_repetitions.append(solve_wor2vec_time)

        print("-----Finish w2v and k-means-----")
        return gensim_opt

    # def _build_and_solve_amdp(self, gensim_opt: GensimOperator_Topology):
    #     print("-----Begin build and solve amdp-----")
    #     amdp = AMDP_Topology_Uniform(env=self.env, uniform_mode=None, gensim_opt=gensim_opt)
    #     start_amdp = time.time()
    #     amdp.solve_amdp()
    #     end_amdp = time.time()
    #     solve_amdp_time = end_amdp - start_amdp
    #     print("solve_amdp_time:", solve_amdp_time)
    #     self.solve_amdp_time_repetitions.append(solve_amdp_time)
    #     print("-----Finish build and solve amdp-----")
    #     return amdp

    def run(self):
        if not os.path.isdir(self.path_results):
            makedirs(self.path_results)
        if self.print_to_file == 1:
            sys.stdout = open(f"{self.path_results}/output.txt", 'w')
            sys.stderr = sys.stdout

        self._print_before_start()

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

            # solve w2v and k-means to get clusters and save cluster file
            self.sentences_collected.extend(self.sentences_period)
            self.sentences_period = []
            random.shuffle(self.sentences_collected)
            gensim_opt = self._w2v_and_kmeans()

            # build and solve amdp
            amdp = self._build_and_solve_amdp(gensim_opt=gensim_opt, general=1)

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
                                                            rep, 'general')

            # save performance of each rep
            self.flags_episodes_repetitions.append(self.flags_episodes)
            self.reward_episodes_repetitions.append(self.reward_episodes)
            self.move_count_episodes_repetitions.append(self.move_count_episodes)

        # plot mean performance among all reps
        ### ax_title = f"flags collection in {'big' if self.big==1 else 'small'} {self.env.maze_name}"
        curve_label = f"general-{self.num_clusters}"
        sliced_f_ep_rep = np.array(self.flags_episodes_repetitions)[:, self.explore_config['e_eps']:]
        sliced_r_ep_rep = np.array(self.reward_episodes_repetitions)[:, self.explore_config['e_eps']:]
        sliced_m_ep_rep = np.array(self.move_count_episodes_repetitions)[:, self.explore_config['e_eps']:]
        plot_maker.plot_mean_performance_across_reps(sliced_f_ep_rep, sliced_r_ep_rep, sliced_m_ep_rep, curve_label)

        plot_maker.plot_mean_time_comparison(self.experiment_time_repetitions, self.solve_amdp_time_repetitions,
                                             self.ground_learning_time_repetitions, self.exploration_time_repetitions,
                                             self.solve_word2vec_time_repetitions, bar_label='general')

        self._results_upload()
        if self.print_to_file == 1:
            sys.stdout.close()


if __name__ == "__main__":
    maze = 'low_connectivity2'  # low_connectivity2/external_maze21x21_1/external_maze31x31_2/strips2/spiral/basic
    big = 0
    e_mode = 'sarsa'   # 'sarsa' or 'softmax'
    e_start = 'last'   # 'random' or 'last' or 'mix'
    e_eps = 2000
    mm = 100
    ds_factor = 0.5

    q_eps = 500
    repetitions = 2
    rep_size = 128
    win_size = 40
    sg = 1  # 'SG' or 'CBOW'
    # clusters = [9, 16, 25, 36]     # number of abstract states for Uniform will be matched with the number of clusters
    numbers_of_clusters = [16]  # number of abstract states for Uniform will be matched with the number of clusters
    k_means_pkg = 'sklearn'    # 'sklearn' or 'nltk'
    interpreter = 'R'     # L or R
    print_to_file = 0

    std_factor = 1 / np.sqrt(10)

    for i in range(len(numbers_of_clusters)):
        plot_maker = PlotMaker(repetitions, std_factor, 3)
        # set directory to store imgs and files
        path_results =f"./cluster_layout/{maze}_big={big}" \
                      f"/topology{numbers_of_clusters} test/rp{repetitions}_{e_start}{e_eps}+{q_eps}_mm{mm}_" \
                      f"ds{ds_factor}_win{win_size}_rep{rep_size}_sg{sg}_{k_means_pkg}_{interpreter}/k[{numbers_of_clusters[i]}]"

        # ===topology approach===
        topology_maker = TopologyExpMaker(env_name=maze, big=big, e_mode=e_mode, e_start=e_start, e_eps=e_eps, mm=mm, ds_factor=ds_factor,
                     rep_size=rep_size, win_size=win_size, sg=sg, num_clusters=numbers_of_clusters[i], k_means_pkg=k_means_pkg, q_eps=q_eps,
                     repetitions=repetitions, interpreter=interpreter, print_to_file=print_to_file, plot_maker=plot_maker, path_results=path_results)
        topology_maker.run()

        # ===uniform approach===
        # match number of abstract state same with the one in topology approach, in order to be fair.
        a = math.ceil(topology_maker.env.size[0] / np.sqrt(numbers_of_clusters[i]))
        b = math.ceil(topology_maker.env.size[1] / np.sqrt(numbers_of_clusters[i]))
        print("(a,b): ", (a,b))
        uniform_maker = UniformExpMaker(env_name=maze, big=big, tiling_size=(a, b), q_eps=q_eps, repetitions=repetitions,
                                        interpreter=interpreter, print_to_file=print_to_file, plot_maker=plot_maker,
                                        path_results=path_results)
        uniform_maker.run()

        # ===general approach===
        general_maker = GeneralExpMaker(env_name=maze, big=big, e_mode=e_mode, e_start='random', e_eps=int(e_eps*2), mm=mm, ds_factor=ds_factor,
                     rep_size=rep_size, win_size=win_size, sg=sg, num_clusters=int(numbers_of_clusters[i]*8+1), k_means_pkg=k_means_pkg, q_eps=q_eps,
                     repetitions=repetitions, interpreter=interpreter, print_to_file=print_to_file, plot_maker=plot_maker, path_results=path_results)
        general_maker.run()

        # ===plot and save mean performance===
        print("saving fig_each_rep ...")
        plot_maker.fig_each_rep.savefig(f"{path_results}/plots_of_each_rep.png", dpi=600, facecolor='w', edgecolor='w',
                    orientation='portrait', format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1)

        plot_maker.fig_mean_performance.show()
        # plot_maker.fig_mean_performance.savefig(f"{path_results}/mean_results_errorbar_yerror*{plot_maker.std_factor}.png",
        #                                         dpi=600, facecolor='w', edgecolor='w', orientation='portrait',
        #                                         format=None, transparent=False, bbox_inches=None, pad_inches=0.1)
