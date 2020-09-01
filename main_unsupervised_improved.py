import copy
import random
import pickle
import matplotlib.pyplot as plt
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
from abstraction import AMDP
from maze_env_general import Maze
from RL_brain_fast import WatkinsQLambda
from gensim_operation_online import GensimOperator

# abstraction_mode = [None, (3, 3), (4, 4), (5, 5), (7, 7), (9, 9), None]   # 可修改
abstraction_mode = [None]  # 可修改
env = Maze(maze='low_connectivity')  # initialize env 可修改


num_of_actions = 4
num_of_experiments = len(abstraction_mode)
lr = 0.1
lam = 0.9
gamma = 0.99
omega = 20

epsilon = 1   # probability for choosing random action  #可修改
num_randomwalk_episodes = 400
second_evolution  = 400 + 999
third_evolution = 400 + 999 + 1999
num_saved_from_p1 = 750
num_saved_from_p2 = 1500
num_of_episodes = num_randomwalk_episodes + 6000 # 可修改
num_of_repetitions = 2      # 可修改
max_move_count = 10000
min_length_to_save_as_path = 400

config = {
    'maze': env.maze_name,
    'mode': 'random+biased_paths',
    'ep': num_of_episodes,
    'rp': num_of_repetitions,
    'max_move_count': max_move_count,
    'min_length_to_save': min_length_to_save_as_path,
    'representation_size': 64,
    'window': 20,
    'kmeans_clusters': [10,20,25],
    'package': 'sklearn'
}


folder_cluster_layout = f"cluster_layout/{config['maze']}/{config['mode']}/rp{config['rp']}_ep{config['ep']}" \
                        f"_c1_{num_randomwalk_episodes}" \
                        f"_c2_{second_evolution}" \
                        # f"_c3_{third_evolution}"

# folder_cluster_layout = f"cluster_layout/{config['maze']}/{config['mode']}/rp{config['rp']}_ep{config['ep']}" \
#                         f"_c1_{num_randomwalk_episodes}" \
#                         f"_c2_{second_evolution}({num_saved_from_p1})" \

if not os.path.isdir(folder_cluster_layout):
    makedirs(folder_cluster_layout)

# for ploting
solve_amdp_time_experiments_repetitions = []
simulation_time_experiments_repetitions = []
reward_list_episodes_experiments_repetitions = []
flags_list_episodes_experiments_repetitions = []
move_count_episodes_experiments_repetitions = []
path_episodes_experiments_repetitions = []
flags_found_order_experiments_repetitions = []


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
        path_episodes = []
        solve_amdp_time_phases = []
        all_path_lengths = []
        paths_period = []

        agent = WatkinsQLambda(env.size, num_of_actions, env, epsilon, lr, gamma, lam)  ## resets the agent
        gensim_opt = GensimOperator(path_episodes, env)

        print("Begin Training:")
        for ep in range(0, num_of_episodes):
            if (ep + 1) % 100 == 0:
                print("episode_100:", ep)

            if ep == num_randomwalk_episodes:
                # print("path_episodes:",path_episodes)
                # min_length_to_save_as_path -= 150

                print("len of paths_period:",len(paths_period))
                agent.epsilon = 0.5
                saved_paths_randomwalk = paths_period
                path_episodes.extend(saved_paths_randomwalk)
                paths_period = []

                gensim_opt.sentences = path_episodes
                gensim_opt.get_clusterlayout_from_paths(size=64, window=20, clusters=config['kmeans_clusters'][0], package=config['package'])
                fpath_cluster_layout = folder_cluster_layout + f"/rep{rep}_s{config['representation_size']}_w{config['window']}" \
                        f"_kmeans{config['kmeans_clusters'][0]}_{config['package']}.cluster"
                gensim_opt.write_cluster_layout(fpath_cluster_layout)

                amdp = AMDP(env=env, tiling_mode=None, dw_clt_layout=gensim_opt.cluster_layout)
                start1 = time.time()
                amdp.solveAbstraction()
                end1 = time.time()
                solve_amdp_time_phases.append(end1 - start1)
            elif ep == second_evolution:
                # min_length_to_save_as_path -= 200

                print("len of paths_period:",len(paths_period))
                # saved_paths_period1 = sorted(paths_period,key=lambda l:len(l))[:num_saved_from_p1]
                saved_paths_period1 = paths_period
                path_episodes.extend(saved_paths_period1)
                paths_period = []

                gensim_opt.sentences = path_episodes
                gensim_opt.get_clusterlayout_from_paths(size=64, window=20, clusters=config['kmeans_clusters'][1], package=config['package'])
                fpath_cluster_layout = folder_cluster_layout + f"/rep{rep}_s{config['representation_size']}_w{config['window']}" \
                        f"_kmeans{config['kmeans_clusters'][1]}_{config['package']}.cluster"
                gensim_opt.write_cluster_layout(fpath_cluster_layout)

                amdp = AMDP(env=env, tiling_mode=None, dw_clt_layout=np.array(gensim_opt.cluster_layout))
                start1 = time.time()
                amdp.solveAbstraction()
                end1 = time.time()
                solve_amdp_time_phases.append(end1 - start1)

            # elif ep == third_evolution:
            #     # min_length_to_save_as_path -= 100
            #
            #     print("len of paths_period:",len(paths_period))
            #     # saved_paths_period2 = sorted(paths_period,key=lambda l:len(l))[:num_saved_from_p2]
            #     saved_paths_period2 = paths_period
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

            env.reset()
            agent.resetEligibility()  # 可以修改

            if ep >= num_randomwalk_episodes and ep-num_randomwalk_episodes > (num_of_episodes-num_randomwalk_episodes)/10: # 可修改
                agent.epsilon -= (0.5) / (num_of_episodes-num_randomwalk_episodes)   ##reduce exploration over time.   # 可修改

            episode_reward = 0
            move_count = 0
            a = agent.policy(env.state, env.actions(env.state))
            path = [str((env.state[0], env.state[1]))]

            while not env.isTerminal(env.state):
                # print("env.isTerminal(env.state):",env.isTerminal(env.state))
                move_count += 1

                if ep < num_randomwalk_episodes:
                    if move_count > max_move_count:
                        break
                    else:
                        new_state = env.step(env.state, a)
                        a_prime = agent.policy(new_state, env.actions(new_state))
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

                    # if abstract_state[0] not in gensim_opt.cluster_labels or new_abstract_state[0] not in gensim_opt.cluster_labels:
                    if False:
                        shaping = 0
                        agent.learn(env.state, a, new_state, a_prime, a_star, r + shaping)  # 可以修改
                        path.append(str((new_state[0], new_state[1])))
                    else:
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
            paths_period.append(path)
            all_path_lengths.append(len(path))
            # if len(path) > min_length_to_save_as_path:
            #     path_episodes.append(path)
        # =====================
        solve_amdp_time_experiments.append(solve_amdp_time_phases)
        reward_list_episodes_experiments.append(reward_list_episodes)
        flags_list_episodes_experiments.append(flags_list_episodes)
        move_count_episodes_experiments.append(move_count_episodes)
        path_episodes_experiments.append(path_episodes)
        flags_found_order_experiments.append(env.flags_found_order)

        # plot flag collection in one experiment
        plt.rcParams['agg.path.chunksize'] = 10000
        d = pd.Series(flags_list_episodes)
        movAv = pd.Series.rolling(d, window=int(num_of_episodes / 30), center=False).mean()
        plt.plot(np.arange(len(movAv)), movAv, label='biased')
        plt.ylabel("Number of Flags")
        plt.xlabel("Episde No.")
        plt.legend(loc=4)
        plt.show()

        end2 = time.time()
        simulation_time_experiments.append(end2 - start2)
        print("last state:",env.state)
        print("all_path_lengths:", all_path_lengths)
        print("len of all_path_lengths:", len(all_path_lengths))

        print("avg and len of random walk period:", mean(len(x) for x in saved_paths_randomwalk),len(saved_paths_randomwalk))
        print("avg and len of period1:", mean([len(x) for x in saved_paths_period1]),len(saved_paths_period1))
        # print("avg and len of period2:", mean([len(x) for x in saved_paths_period2]), len(saved_paths_period2))
        # mean_unreduced = mean(all_path_lengths[:num_randomwalk_episodes])
        # saved_path_lengths = [x for x in all_path_lengths[:num_randomwalk_episodes] if x > min_length_to_save_as_path]
        # if len(saved_path_lengths)>0:
        #     mean_reduced = mean(saved_path_lengths)
        #     print("avg saved_path_lengths of random walk period:", mean_reduced, len(saved_path_lengths))
        # print("avg all_path_lengths of random walk period:",mean_unreduced)

        # mean_unreduced = mean(all_path_lengths[num_randomwalk_episodes:num_randomwalk_episodes+999])
        # saved_path_lengths = [x for x in all_path_lengths[num_randomwalk_episodes:num_randomwalk_episodes+999] if x > min_length_to_save_as_path]
        # if len(saved_path_lengths)>0:
        #     mean_reduced = mean(saved_path_lengths)
        #     print("avg saved_path_lengths of biased walk period1:", mean_reduced, len(saved_path_lengths))
        # print("avg all_path_lengths of biased walk period1:",mean_unreduced)
        #
        # mean_unreduced = mean(all_path_lengths[num_randomwalk_episodes+999:num_randomwalk_episodes+1999])
        # saved_path_lengths = [x for x in all_path_lengths[num_randomwalk_episodes+999:num_randomwalk_episodes+1999] if x > min_length_to_save_as_path]
        # if len(saved_path_lengths)>0:
        #     mean_reduced = mean(saved_path_lengths)
        #     print("avg saved_path_lengths of biased walk period2:", mean_reduced, len(saved_path_lengths))
        # print("avg all_path_lengths of biased walk period2:",mean_unreduced)
        #
        # mean_unreduced = mean(all_path_lengths[num_randomwalk_episodes+1999:])
        # saved_path_lengths=[x for x in all_path_lengths[num_randomwalk_episodes+1999:] if x > min_length_to_save_as_path]
        # if len(saved_path_lengths)>0:
        #     mean_reduced = mean(saved_path_lengths)
        #     print("avg saved_path_lengths of biased walk period3:", mean_reduced, len(saved_path_lengths))
        # print("avg all_path_lengths of biased walk period3:",mean_unreduced)

        #=====================
        print("agent.epsilon:",agent.epsilon)



    solve_amdp_time_experiments_repetitions.append(solve_amdp_time_experiments)
    simulation_time_experiments_repetitions.append(simulation_time_experiments)
    flags_list_episodes_experiments_repetitions.append(flags_list_episodes_experiments)
    reward_list_episodes_experiments_repetitions.append(reward_list_episodes_experiments)
    move_count_episodes_experiments_repetitions.append(move_count_episodes_experiments)
    path_episodes_experiments_repetitions.append(path_episodes_experiments)
    flags_found_order_experiments_repetitions.append(flags_found_order_experiments)

print(flags_list_episodes_experiments_repetitions)
print("flags_list_episodes_experiments_repetitions.shape:", np.array(flags_list_episodes_experiments_repetitions).shape)
print(move_count_episodes_experiments_repetitions)
print("move_count_episodes_experiments_repetitions.shape:", np.array(move_count_episodes_experiments_repetitions).shape)
print(np.sum(np.array(move_count_episodes_experiments_repetitions)))
print(flags_found_order_experiments_repetitions)


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




# if __name__ == "__main__":
#     pass



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

# output_dir = "FExperiments/" + env.maze_name + "qLambdaAlpha" + str(lr) + "Gamma" + str(gamma) + "Lambda" + str(
#     lam) + "Epsilon" + str(agent.epsilon) + "Episodes" + str(num_of_episodes)  # 可修改
# if env.walls == []:
#     output_dir = output_dir + "NoWalls"
# else:
#     output_dir = output_dir + "Walls"
# mkdir_p(output_dir)

output_dir = folder_cluster_layout
## Reward
whenConverged = []
toPickle = []
plt.figure(1)
plotRewards = np.mean(flags_list_episodes_experiments_repetitions, axis=0)
plotSDs = np.std(flags_list_episodes_experiments_repetitions, axis=0)
print("plotRewards.shape", plotRewards.shape)
print("plotSDs.shape", plotSDs.shape)
plotErrors = plotSDs / np.sqrt(10)
plt.rcParams['agg.path.chunksize'] = 10000
for i in range(0, len(plotRewards)):
    d = pd.Series(plotRewards[i])
    s = pd.Series(plotErrors[i])
    movAv = pd.Series.rolling(d, window=int(num_of_episodes / 30), center=False).mean()
    toPickle.append(movAv)
    l, caps, c = plt.errorbar(np.arange(len(movAv)), movAv, label=labs[i], yerr=plotErrors[i], capsize=5, errorevery=int(num_of_episodes / 10))
    for cap in caps:
        cap.set_marker("_")
plt.ylabel("No. Of Flags Collected")
plt.xlabel("Episode No.")
plt.legend(loc=4)
plt.axis([0, num_of_episodes, 0, 3])
# print(whenConverged)

# with open("{}/resultsListPickle".format(output_dir), 'wb') as p:
#     pickle.dump(toPickle, p)

##plt.title("Number of Episodes: " + str(num_of_episodes) + " Alpha: " + str(lr) + " Gamma: " + str(gamma) + " Lambda: " +str(lam) + " Epsilon: "+str(agent.epsilon))


plt.savefig("{}/rewardGraph.png".format(output_dir), dpi=1200, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1)

## Flags Collected
plt.figure(2)
plotFlags = np.mean(flags_list_episodes_experiments_repetitions, axis=0)
plt.rcParams['agg.path.chunksize'] = 10000
for i in range(0, len(plotFlags)):
    d = pd.Series(plotFlags[i])
    movAv = pd.Series.rolling(d, window=int(num_of_episodes / 30), center=False).mean()
    plt.plot(np.arange(len(movAv)),movAv, label=labs[i])
plt.ylabel("Number of Flags")
plt.xlabel("Episde No.")
plt.legend(loc=4)

plt.savefig("{}/rewardGraph_noerrorbar.png".format(output_dir), dpi=1200, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1)

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

plt.show()
