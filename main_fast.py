# from PIL import Image
import time
from abstractions.abstraction import AMDP
from envs.maze_env_general import Maze
from RL_brains.RL_brain_empty_q_3 import WatkinsQLambda

# abstraction_mode = [None, (3, 3), (4, 4), (5, 5), (7, 7), (9, 9), None]   # 可修改
abstraction_mode = [None, None]  # 可修改
env = Maze(maze='open_space')  # initialize env 可修改


num_of_actions = 4
num_of_experiments = len(abstraction_mode)
lr = 0.1
lam = 0.9
gamma = 0.99
omega = 20
epsilon = 0.5   # probability for choosing random action  #可修改
num_of_episodes = 200       # 可修改
num_of_repetitions = 1      # 可修改

# for ploting
solve_amdp_time_experiments_repetitions = []
simulation_time_experiments_repetitions = []
reward_list_episodes_experiments_repetitions = []
flags_list_episodes_experiments_repetitions = []
move_count_episodes_experiments_repetitions = []
path_episodes_experiments_repetitions = []

for rep in range(0, num_of_repetitions):

    solve_amdp_time_experiments = []
    simulation_time_experiments = []
    reward_list_episodes_experiments = []
    flags_list_episodes_experiments = []
    move_count_episodes_experiments = []
    path_episodes_experiments = []

    move_count = 0
    totalMoveCount = 0
    maxflag = 0
    maxIndex = 0
    flagCount = 0
    reward_list_episodes = []
    flags_list_episodes = []
    # env.reset()

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
        # collectingReachability = True

        # env.reset(maze_name=env.maze_name)

        agent = WatkinsQLambda(env.size, num_of_actions, env, epsilon, lr, gamma, lam)  ## resets the agent
        # agent.epsilon = 0.5
        if abstraction_mode[e] == None:  # manuel layout mode
            amdp = AMDP(env=env, tiling_mode=None, dw_clt_layout=None)
        elif type(abstraction_mode[e]) == tuple:  # tiling mode
            amdp = AMDP(env=env, tiling_mode=abstraction_mode[e], dw_clt_layout=None)
        elif type(abstraction_mode[e]) == str:  # deepwalk and clustering abstraction mode
            amdp = AMDP(env=env, tiling_mode=None, dw_clt_layout=abstraction_mode[e])
        else:
            raise Exception("invalid tiling_mode!")
            # print("agent.amdp()[e]:", np.array(amdp.abstraction_layout))

        start1 = time.time()
        if e < num_of_experiments - 1:  # last experiment with no reward shaping
            amdp.solveAbstraction()
        end1 = time.time()
        solve_amdp_time_experiments.append(end1 - start1)

        # agent.reachabilityBins = [[amdp.abstract_state(env.state), [env.state]]]

        print("Begin Training:")
        for ep in range(0, num_of_episodes):
            # print("====episode:", ep)
            episode_reward = 0
            if (ep + 1) % 100 == 0:
                print("episode_100:", ep)

            env.reset()

            if ep > (num_of_episodes / 10):
                agent.epsilon -= (0.5) / num_of_episodes   ##reduce exploration over time.   # 可修改


            # agent.resetEligibility()        #可以修改
            agent.resetEligibility()

            move_count = 0
            a = agent.policy(env.state, env.actions(env.state))
            path = [(env.state[0],env.state[1])]

            while not env.isTerminal(env.state):
                # print("env.isTerminal(env.state):",env.isTerminal(env.state))
                move_count += 1
                ##Select action using policy
                abstract_state = amdp.getAbstractState(env.state)
                new_state = env.step(env.state, a)
                new_abstract_state = amdp.getAbstractState(new_state)
                #print(new_state, new_abstract_state)
                if len(path) <= 2500:
                    path.append((new_state[0],new_state[1]))

                a_prime = agent.policy(new_state, env.actions(new_state))  ##Greedy next-action selected
                a_star = agent.policyNoRand(new_state, env.actions(new_state))
                ## Optimal next action ---- comparison of the two required for Watkins Q-lambda

                r = env.reward(env.state, a, new_state)  ## ground level reward
                episode_reward += r

                if e < num_of_experiments - 1:
                    # print("e:",e)
                    value_new_abstract_state = amdp.getValue(new_abstract_state)
                    value_abstract_state = amdp.getValue(abstract_state)
                    # print("type(value_abstract_state):",type(value_abstract_state))
                    shaping = gamma * value_new_abstract_state * omega - value_abstract_state * omega
                else:
                    # print("e:", e)
                    shaping = 0  # We avoid shaping on the last experiment which we set aside for vanilla Q-Learning

                agent.learn(env.state, a, new_state, a_prime, a_star, r + shaping)  # 可以修改
                # print("q-updated")
                ## updates the Q-table for eligible states according to Q-lambda algorithm

                env.state = new_state
                a = a_prime
            # next steps actions and states set.

            ############# Keep Track of Stuff for each ep ################
            # flagCount += env.flags_collected
            # totalMoveCount += move_count
            reward_list_episodes.append(episode_reward)
            flags_list_episodes.append(env.flags_collected)
            move_count_episodes.append(move_count)
            path_episodes.append(path)

        reward_list_episodes_experiments.append(reward_list_episodes)
        flags_list_episodes_experiments.append(flags_list_episodes)
        move_count_episodes_experiments.append(move_count_episodes)
        path_episodes_experiments.append(path_episodes)

        end2 = time.time()
        simulation_time_experiments.append(end2 - start2)

    solve_amdp_time_experiments_repetitions.append(solve_amdp_time_experiments)
    simulation_time_experiments_repetitions.append(simulation_time_experiments)
    flags_list_episodes_experiments_repetitions.append(flags_list_episodes_experiments)
    reward_list_episodes_experiments_repetitions.append(reward_list_episodes_experiments)
    move_count_episodes_experiments_repetitions.append(move_count_episodes_experiments)
    path_episodes_experiments_repetitions.append(path_episodes_experiments)

    print(flags_list_episodes_experiments_repetitions)
    print(move_count_episodes_experiments_repetitions)

    # path = "paths/q_paths_random.txt"
    # if not os.path.isfile(path):
    #     with open(path,"w") as f:
    #         for repetition in path_episodes_experiments_repetitions:
    #             for exp in repetition:
    #                 for episode in exp:
    #                     for coord in episode:
    #                         f.write(str(coord).replace(' ','')+' ')
    #                     f.write('\n')
    # os.chmod(path,S_IREAD)
    # # print(np.array(path_episodes_experiments_repetitions))
    # print(np.array(path_episodes_experiments_repetitions).shape)


    # print(np.array(path_episodes_experiments_repetitions[0][1][:100]))


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

#
# # labs = ["True", "3x3", "4x4", "5x5", "7x7", "9x9", "10x10", "None"]     # 可修改
# labs = ["True", "5x5", "7x7", "9x9", "10x10", "None"]  # 可修改
#
# output_dir = "FExperiments/" + env.maze_name + "qLambdaAlpha" + str(lr) + "Gamma" + str(gamma) + "Lambda" + str(
#     lam) + "Epsilon" + str(agent.epsilon) + "Episodes" + str(num_of_episodes)  # 可修改
# if env.walls == []:
#     output_dir = output_dir + "NoWalls"
# else:
#     output_dir = output_dir + "Walls"
# mkdir_p(output_dir)
#
# ## Reward
# whenConverged = []
# toPickle = []
# plt.figure(1)
# plotRewards = np.mean(flags_list_episodes_experiments_repetitions, axis=0)
# plotSDs = np.std(flags_list_episodes_experiments_repetitions, axis=0)
# plotErrors = plotSDs / np.sqrt(10)
# plt.rcParams['agg.path.chunksize'] = 10000
# for i in range(0, len(plotRewards)):
#     d = pd.Series(plotRewards[i])
#     s = pd.Series(plotErrors[i])
#     movAv = pd.Series.rolling(d, window=int(num_of_episodes / 10), center=False).mean()
#     toPickle.append(movAv)
#     l, caps, c = plt.errorbar(np.arange(len(movAv)), movAv, label=labs[i], yerr=plotErrors[i], capsize=5,
#                               errorevery=num_of_episodes / 10)
#     for cap in caps:
#         cap.set_marker("_")
# plt.ylabel("No. Of Flags Collected")
# plt.xlabel("Episode No.")
# plt.legend(loc=4)
# plt.axis([0, num_of_episodes, 0, 3])
# print(whenConverged)
#
# with open("{}/resultsListPickle".format(output_dir), 'wb') as p:
#     pickle.dump(toPickle, p)
#
# ##plt.title("Number of Episodes: " + str(num_of_episodes) + " Alpha: " + str(lr) + " Gamma: " + str(gamma) + " Lambda: " +str(lam) + " Epsilon: "+str(agent.epsilon))
#
#
# plt.savefig("{}/rewardGraph.png".format(output_dir), dpi=1200, facecolor='w', edgecolor='w',
#             orientation='portrait', papertype=None, format=None,
#             transparent=False, bbox_inches=None, pad_inches=0.1)
#
# ## Flags Collected
# plt.figure(2)
# plotFlags = np.mean(flags_list_episodes_experiments_repetitions, axis=0)
# plt.rcParams['agg.path.chunksize'] = 10000
# for i in range(0, len(plotFlags)):
#     d = pd.Series(plotFlags[i])
#     movAv = pd.Series.rolling(d, window=1000, center=False).mean()
#     plt.plot(movAv, label=labs[i])
# plt.ylabel("Number of Flags")
# plt.xlabel("Episde No.")
# plt.legend(loc=4)
#
# plt.savefig("{}/rewardGraph_noerrorbar.png".format(output_dir), dpi=1200, facecolor='w', edgecolor='w',
#             orientation='portrait', papertype=None, format=None,
#             transparent=False, bbox_inches=None, pad_inches=0.1)
#
# plt.figure(3)
# plotAbsTimings = np.mean(solve_amdp_time_experiments_repetitions, axis=0)
# for i, v in enumerate(plotAbsTimings):
#     plt.text(i - 0.25, v + 3, str(round(v, 1)), color='blue', fontweight='bold')
# plt.bar(np.arange(len(plotAbsTimings)), plotAbsTimings)
# plt.xticks(np.arange(len(plotAbsTimings)), labs)
# plt.xlabel("Abstraction Used")
# plt.ylabel("Time Taken")
# plt.title("Time Taken to Solve Each Abstraction")
# plt.savefig("{}/AbstractionTime.png".format(output_dir), dpi=1200, facecolor='w', edgecolor='w',
#             orientation='portrait', papertype=None, format=None,
#             transparent=False, bbox_inches=None, pad_inches=0.1)
#
# plt.figure(4)
# plotSimTimings = np.mean(simulation_time_experiments_repetitions, axis=0)
# for i, v in enumerate(plotSimTimings):
#     plt.text(i - 0.30, v + 3, str(round(v, 1)), color='blue', fontweight='bold')
# plt.bar(np.arange(len(plotSimTimings)), plotSimTimings)
# plt.xticks(np.arange(len(plotSimTimings)), labs)
# plt.xlabel("Experiments")
# plt.ylabel("Time Taken In Seconds")
# plt.title("Time Taken To Simulate each experiment with episodes" + str(num_of_episodes))
# plt.savefig("{}/SimulationTime.png".format(output_dir), dpi=1200, facecolor='w', edgecolor='w',
#             orientation='portrait', papertype=None, format=None,
#             transparent=False, bbox_inches=None, pad_inches=0.1)
#
# # plt.show()
