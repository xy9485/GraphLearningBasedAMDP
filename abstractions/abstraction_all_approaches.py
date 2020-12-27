
import copy
import numpy as np
import sys
from gensim_operations.gensim_operation_all_approaches import GensimOperator_Topology,GensimOperator_General
np.set_printoptions(linewidth=400, threshold=sys.maxsize)

class AMDP_Topology_Uniform:
    def __init__(self, env=None, uniform_mode=None, gensim_opt=None,):
        assert env != None, "env can't be None!"
        assert (uniform_mode==None) != (gensim_opt == None), "only one of uniform or gensim_opt can be assigned"
        self.manuel_room_layout = env.room_layout  # np array
        self.goal = env.goal
        self.flags = env.flags
        if uniform_mode:
            self.abstraction_layout = self.do_tiling_v2(uniform_mode)
        elif gensim_opt:
            self.gensim_opt: GensimOperator_Topology = gensim_opt
            self.abstraction_layout = np.array(self.gensim_opt.cluster_layout)
        else:
            raise Exception("invalide mode for AMDP_Topology_Uniform")
        print("print abstraction_layout from AMDP_Topology_Uniform:")
        print(self.abstraction_layout)
        self.flags_abstractions = [self.get_abstract_state((i, j, 0, 0, 0))[0] for (i, j) in self.flags]
        self.goal_abstractions = [self.get_abstract_state((self.goal[0], self.goal[1], 0, 0, 0))[0]]
        print("self.flagRooms from AMDP_Topology_Uniform:", self.flags_abstractions)
        print("self.goalRoom from AMDP_Topology_Uniform:", self.goal_abstractions)

        # self.flags_found = [0, 0, 0]
        self.list_of_abstract_states = None
        self.adjacencies = None
        self.adjacencies_for_each_astate = None
        self.transition_table = None
        self.rewards_table = None
        self.V = None  # with to be set in main.py by self.solveAbstraction()

        self.list_of_abstract_states = self.get_list_of_abstract_state()
        self.adjacencies, self.adjacencies_for_each_astate = self.get_ajacencies_v2()
        # self.transition_table, self.rewards_table = self.get_transitions_and_rewards()
        self.transition_table, self.rewards_table = self.get_transitions_and_rewards_streamlined()

        # # self.solveAbstraction()

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

    def get_abstract_state(self, state):
        abstract_state = [self.abstraction_layout[state[0], state[1]]]
        for i in range(2, len(state)):
            abstract_state.append(state[i])
        return abstract_state

    def get_list_of_abstract_state(self) -> list:
        # build all the potential abstract states
        list_of_abstract_states = []
        for i in range(len(self.abstraction_layout)):
            for j in range(len(self.abstraction_layout[0])):
                for k in range(2):
                    for l in range(2):
                        for m in range(2):
                            temp_astate = self.get_abstract_state((i, j, k, l, m))  # return a list
                            if temp_astate not in list_of_abstract_states:
                                if not self.abstraction_layout[i][j] == "w":
                                    list_of_abstract_states.append(temp_astate)
        return list_of_abstract_states

    def get_ajacencies_v2(self):
        adjacencies = []
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    for i in range(len(self.abstraction_layout) - 1):
                        for j in range(len(self.abstraction_layout[0]) - 1):
                            current = self.get_abstract_state((i, j, k, l, m))
                            down = self.get_abstract_state((i + 1, j, k, l, m))
                            right = self.get_abstract_state((i, j + 1, k, l, m))
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
                        current = self.get_abstract_state((i, j, k, l, m))
                        down = self.get_abstract_state((i + 1, j, k, l, m))
                        if not current == down and not current[0] == "w" and not down[0] == "w":
                            if (current, down) not in adjacencies:
                                adjacencies.append((current, down))
                                adjacencies.append((down, current))
                    # designed for the state on the bottom border of the maze
                    i = len(self.abstraction_layout) - 1
                    for j in range(len(self.abstraction_layout[0]) - 1):
                        current = self.get_abstract_state((i, j, k, l, m))
                        right = self.get_abstract_state((i, j + 1, k, l, m))
                        if not current == right and not current[0] == "w" and not right[0] == "w":
                            if (current, right) not in adjacencies:
                                adjacencies.append((current, right))
                                adjacencies.append((right, current))

        # self.adjacencies = adjacencies
        # get the adjacencies for each abstraction(tiling)
        adjacencies_for_each_abstract_state = []
        for a in self.list_of_abstract_states:
            adj = [a]
            for b in self.list_of_abstract_states:
                if (a, b) in adjacencies:
                    adj.append(b)
            adjacencies_for_each_abstract_state.append(adj)
        adjacencies_for_each_abstract_state.append(["bin", "bin"])
        # self.adjacencies_for_each_astate = adjacencies_for_each_abstract_state

        return adjacencies, adjacencies_for_each_abstract_state

    def get_abstract_actions(self, abstract_state):  # get available adjacent astate and current visiting of F or G
        newlist = []
        for ajacency in self.adjacencies_for_each_astate:
            if ajacency[0] == abstract_state:
                newlist = ajacency[1:]
                if not ajacency[0] == "bin" and ajacency[0][0] in self.flags_abstractions:
                    newlist.append("F" + ajacency[0][0])
                if not ajacency[0] == "bin" and ajacency[0][0] in self.goal_abstractions:
                    newlist.append("G" + ajacency[0][0])
                break
        return newlist


    def get_transitions_and_rewards(self):
        self.list_of_abstract_states.append("bin")
        num_of_abstract_state = len(self.list_of_abstract_states)
        ## Initialise empty action, State, State' transitions and reward
        transition = np.zeros((num_of_abstract_state + 4, num_of_abstract_state, num_of_abstract_state))
        rewards = np.zeros((num_of_abstract_state + 4, num_of_abstract_state, num_of_abstract_state))

        # actions_f = ["F"+ self.flags_tilings[f] for f in range(len(self.flags_tilings))]
        for i in range(num_of_abstract_state):
            ajacency_of_i = self.get_abstract_actions(self.list_of_abstract_states[i])
            # print("len(ajacency_of_i)",len(ajacency_of_i))
            for j in range(num_of_abstract_state):
                # normal transition, nothing happens
                if self.list_of_abstract_states[j] in ajacency_of_i:
                    transition[j, i, j] = 1
                    # print("set normal transition:",[j, i, j])

                # flag collection transition
                for f in range(len(self.flags_abstractions)):
                    action_f = "F" + self.flags_abstractions[f]
                    if action_f in ajacency_of_i:
                        s_prime = copy.deepcopy(self.list_of_abstract_states[i])
                        if s_prime[1 + f] == 0:
                            s_prime[1 + f] = 1
                            if self.list_of_abstract_states[j] == s_prime:
                                transition[num_of_abstract_state + f, i, j] = 1
                                # print("set flag transition:",[num_of_abstract_state + f, i, j])


            # # goal transition
            for g in range(len(self.goal_abstractions)):
                action_g = "G" + self.goal_abstractions[g]
                # if action_g in ajacency_of_i:
                if action_g in ajacency_of_i and sum(self.list_of_abstract_states[i][1:])==3:  #可修改
                    transition[-1, i, -1] = 1
                    # print("goal transition set!!!")
                    # print("set goal transition:",[-1, i, -1])

        # self.transition = transition
        # print("where T1!=T2:",np.array_equal(transition,transition2))
        # print("self.transition:",self.transition)

        # set rewards
        for i in range(num_of_abstract_state):
            if not self.list_of_abstract_states[i]=="bin" and sum(self.list_of_abstract_states[i][1:])==3:    #可修改
                ajacency_of_i = self.get_abstract_actions(self.list_of_abstract_states[i])
                for g in range(len(self.goal_abstractions)):
                    action_g = "G" + self.goal_abstractions[g]
                    if action_g in ajacency_of_i:
                        # if not self.list_of_abstract_states[i] == "bin":
                        rewards[-1, i, -1] = sum(self.list_of_abstract_states[i][1:]) * 1000    #可修改
                        # print("abstract reward set!!!")
                        # print("set reward:", [-1, i, -1])
        # self.rewards = rewards

        # self.transition[-1,-1,-1]=1
        # self.rewards[-1,-1,-1]=1
        return transition, rewards


    # def solve_amdp(self):
    #     print('length of self.list_of_abstract_states:', len(self.list_of_abstract_states))
    #     print('self.list_of_abstract_states:', self.list_of_abstract_states)
    #     values = np.zeros(len(self.list_of_abstract_states))
    #     print("len(values):", len(values))
    #     # print("self.transition_table:",np.argwhere(self.transition_table))
    #     # print("self.rewards_table:",np.argwhere(self.rewards_table))
    #     delta = 0.2
    #     theta = 0.1
    #     print("Value Iteration delta values:")
    #     while delta > theta:
    #         delta = 0
    #         for i in range(0, len(values)):
    #             v = values[i]
    #             list_of_values = []
    #             for a in range(len(self.list_of_abstract_states) + 4):
    #                 value = 0
    #                 for j in range(len(values)):
    #                     value += self.transition_table[a, i, j] * (self.rewards_table[a, i, j] + 0.99 * values[j])
    #                 list_of_values.append(value)
    #             values[i] = max(list_of_values)
    #             delta = max(delta, abs(v - values[i]))
    #         print("delta:", delta)
    #     # print(V)
    #     values -= min(values[:-1])
    #     self.values_of_abstract_states = values
    #     print("self.values_of_abstract_states:")
    #     print(self.values_of_abstract_states)

    def get_transitions_and_rewards_streamlined(self):
        self.list_of_abstract_states.append("bin")
        num_of_abstract_state = len(self.list_of_abstract_states)
        ## Initialise empty action, State, State' transitions and reward
        transition = np.zeros((num_of_abstract_state, num_of_abstract_state, num_of_abstract_state))
        rewards = np.zeros((num_of_abstract_state, num_of_abstract_state, num_of_abstract_state))

        # actions_f = ["F"+ self.flags_tilings[f] for f in range(len(self.flags_tilings))]
        for i in range(num_of_abstract_state):
            ajacency_of_i = self.get_abstract_actions(self.list_of_abstract_states[i])
            # print("len(ajacency_of_i)",len(ajacency_of_i))
            for j in range(num_of_abstract_state):
                # normal transition, nothing happens
                if self.list_of_abstract_states[j] in ajacency_of_i:
                    transition[j, i, j] = 1
                    # print("set normal transition:",[j, i, j])

                # flag collection transition
                for f in range(len(self.flags_abstractions)):
                    action_f = "F" + self.flags_abstractions[f]
                    if action_f in ajacency_of_i:
                        s_prime = copy.deepcopy(self.list_of_abstract_states[i])
                        if s_prime[1 + f] == 0:
                            s_prime[1 + f] = 1
                            if self.list_of_abstract_states[j] == s_prime:
                                transition[j, i, j] = 1
                                # print("set flag transition:",[num_of_abstract_state + f, i, j])


            # # goal transition
            for g in range(len(self.goal_abstractions)):
                action_g = "G" + self.goal_abstractions[g]
                # if action_g in ajacency_of_i:
                if action_g in ajacency_of_i and sum(self.list_of_abstract_states[i][1:])==3:  #可修改
                    transition[-1, i, -1] = 1
                    # print("goal transition set!!!")
                    # print("set goal transition:",[-1, i, -1])

        # self.transition = transition
        # print("where T1!=T2:",np.array_equal(transition,transition2))
        # print("self.transition:",self.transition)

        # set rewards
        for i in range(num_of_abstract_state):
            if not self.list_of_abstract_states[i]=="bin" and sum(self.list_of_abstract_states[i][1:])==3:    #可修改
                ajacency_of_i = self.get_abstract_actions(self.list_of_abstract_states[i])
                for g in range(len(self.goal_abstractions)):
                    action_g = "G" + self.goal_abstractions[g]
                    if action_g in ajacency_of_i:
                        # if not self.list_of_abstract_states[i] == "bin":
                        rewards[-1, i, -1] = sum(self.list_of_abstract_states[i][1:]) * 1000    #可修改
                        # print("abstract reward set!!!")
                        # print("set reward:", [-1, i, -1])
        # self.rewards = rewards

        # self.transition[-1,-1,-1]=1
        # self.rewards[-1,-1,-1]=1
        return transition, rewards

    def solve_amdp(self):   # streamlined solve_amdp
        print('length of self.list_of_abstract_states:', len(self.list_of_abstract_states))
        print('self.list_of_abstract_states:', self.list_of_abstract_states)
        values = np.zeros(len(self.list_of_abstract_states))
        print("len(values):",len(values))
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
                for a in range(len(self.list_of_abstract_states)):
                    # value = 0
                    # for j in range(len(V)):
                    #     value += self.transition_table[a, i, j] * (self.rewards_table[a, i, j] + 0.99 * V[j])
                    value = self.transition_table[a, i, a] * (self.rewards_table[a, i, a] + 0.99 * values[a])
                    list_of_values.append(value)
                values[i] = max(list_of_values)
                delta = max(delta, abs(v - values[i]))
            print("delta:", delta)
        # print(V)
        values -= min(values[:-1])
        self.values_of_abstract_states = values
        print("self.values_of_abstract_states:")
        print(self.values_of_abstract_states)

    def get_value(self, astate):
        value = self.values_of_abstract_states[self.list_of_abstract_states.index(astate)]
        # print("value:",value)
        return value





class AMDP_General:
    def __init__(self, env=None, gensim_opt=None):  # tiling_mode is same with tiling_size
        assert (env!=None) and (gensim_opt!=None), "env and gensim_opt need to be assigned"

        # self.manuel_layout = env.room_layout     # np array
        self.goal = env.goal
        # self.flags = env.flags

        self.gensim_opt: GensimOperator_General = gensim_opt
        print("self.gensim_opt.sentences[:5]:", self.gensim_opt.sentences[:5])

        self.list_of_abstract_states = np.arange(self.gensim_opt.num_clusters).tolist()

        self.dict_gstates_astates = dict(zip(self.gensim_opt.words, self.gensim_opt.cluster_labels.tolist()))
        print("len(gensim_opt.words), len(gensim_opt.cluster_labels):", len(self.gensim_opt.words), len(self.gensim_opt.cluster_labels.tolist()))

        print("start setting amdp transition and reward...")
        self.set_transition_and_rewards()

    def get_abstract_state(self, state):
        if not isinstance(state, str):
            state = str(state)
        return self.dict_gstates_astates[state]

    def set_transition_and_rewards(self):
        num_abstract_states = len(self.list_of_abstract_states) + 1     #+1 for absorbing abstract state
        transition = np.zeros(shape=(num_abstract_states, num_abstract_states, num_abstract_states))
        rewards = np.zeros(shape=(num_abstract_states, num_abstract_states, num_abstract_states))
        for sentence in self.gensim_opt.sentences:
            for i in range(len(sentence)):
                if i < (len(sentence)-1):
                    # index1 = self.list_of_ground_states.index(sentence[i])
                    # index2 = self.list_of_ground_states.index(sentence[i+1])
                    # cluster_label1 = self.list_of_abstract_states[index1]
                    # cluster_label2 = self.list_of_abstract_states[index2]
                    cluster_label1 = self.get_abstract_state(sentence[i])
                    cluster_label2 = self.get_abstract_state(sentence[i+1])
                    if not cluster_label1 == cluster_label2:
                        transition[cluster_label2, cluster_label1, cluster_label2] = 1
                        # transition[cluster_label1, cluster_label2, cluster_label1] = 1
                else:
                    # index1 = self.list_of_ground_states.index(sentence[i])
                    # cluster_label1 = self.list_of_abstract_states[index1]
                    cluster_label1 = self.get_abstract_state(sentence[i])
                state_in_tuple = eval(sentence[i])
                if state_in_tuple == (self.goal[0], self.goal[1], 1, 1, 1):
                    transition[-1, cluster_label1, -1] = 1
                    rewards[-1, cluster_label1, -1] = 3000
        self.num_abstract_states = num_abstract_states
        self.transition = transition
        self.rewards = rewards

    def solve_amdp(self):
        values = np.zeros(self.num_abstract_states)
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
                    # value = 0
                    # for j in range(len(V)):
                    # value += self.transition[a, i, j] * (self.rewards[a, i, j] + 0.99 * V[j])
                    value = self.transition[a, i, a] * (self.rewards[a, i, a] + 0.99 * values[a])
                    list_of_values.append(value)
                values[i] = max(list_of_values)
                delta = max(delta, abs(v - values[i]))
            print("delta:", delta)
        # print(V)
        values -= min(values[:-1])
        self.values_of_abstract_states = values
        print("self.values_of_abstract_states:")
        print(self.values_of_abstract_states)

    def get_value(self, astate):
        assert isinstance(astate, int), "astate has to be int"
        value = self.values_of_abstract_states[astate]
        # print("value:",value)
        return value




if __name__ == "__main__":
    from envs.maze_env_general import Maze

    env = Maze(maze='open_space')  # initialize env 可修改
    amdp = AMDP_Topology_Uniform(env=env, tiling_mode=None, dw_clt_layout=None)
    # print(amdp.abstraction_layout)
    for i in range(len(amdp.adjacencies_for_each_astate)):
        print(amdp.adjacencies_for_each_astate[i] )
    # print("self.rewards_table:", np.argwhere(amdp.rewards_table))
    amdp.solveAbstraction()
    for item in amdp.list_of_abstract_states:
        print("astate and value:", item, amdp.getValue(item))

