
import copy
import numpy as np
import sys
np.set_printoptions(linewidth=400, threshold=sys.maxsize)
import pandas as pd
from pprint import pprint


class AMDP:
    def __init__(self, env=None, tiling_mode=None, dw_clt_layout=None, ):  # tiling_mode is same with tiling_size
        if env == None:
            raise Exception("env can't be None!")

        self.manuel_layout = env.room_layout     #np array
        self.goal = env.goal
        self.flags = env.flags

        if tiling_mode:
            self.abstraction_layout = self.doTiling2(tiling_mode)
        elif len(dw_clt_layout) > 0:
            self.abstraction_layout = np.array(dw_clt_layout)
        else:
            self.abstraction_layout = self.manuel_layout
        print("self.abstraction_layout:")
        print(self.abstraction_layout)

        self.flags_abstractions = [self.getAbstractState((i, j, 0, 0, 0))[0] for (i, j) in self.flags]
        self.goal_abstractions = [self.getAbstractState((self.goal[0], self.goal[1], 0, 0, 0))[0]]
        print("self.flagRooms:", self.flags_abstractions)
        print("self.goalRoom:", self.goal_abstractions)
        #
        # # if tilingSize is None:
        # #     self.abstraction_layout = self.manuel_layout
        # # else:
        # #     self.abstraction_layout = self.doTiling2(tilingSize)
        #
        # # self.flags_tilings = [self.abstraction_layout[self.flags[0]],
        # #                       self.abstraction_layout[self.flags[1]],
        # #                       self.abstraction_layout[self.flags[2]]]
        # # self.goal_tiling = [self.abstraction_layout[self.goal[0]]]
        #
        self.flags_found = [0, 0, 0]
        self.list_of_abstract_states = None
        self.adjacencies = None
        self.adjacencies_for_each_astate = None
        self.transition_table = None
        self.rewards_table = None
        self.V = None  # with to be set in main.py by self.solveAbstraction()
        self.list_of_abstract_states = self.getListOfAbstractState()

        self.adjacencies, self.adjacencies_for_each_astate = self.getAjacencies2()
        self.transition_table, self.rewards_table = self.getTransitionsAndRewardsWithPB()
        # # self.solveAbstraction()

    def doTiling(self, tilingSize):
        columns = len(self.manuel_layout[0])
        rows = len(self.manuel_layout)
        tilingLayout = self.manuel_layout.copy().tolist()
        tilingLabel = (1, 1)
        currentY = 0
        for i in range(rows):
            if currentY < (tilingSize[1]):
                currentY += 1
            else:
                tilingLabel = (tilingLabel[0] + 1, tilingLabel[1])
                currentY = 1

            tilingLabel = (tilingLabel[0], 1)
            currentX = 0
            for j in range(columns):
                if currentX < (tilingSize[0]):
                    # print(str(tilingLabel))
                    tilingLayout[i][j] = str(tilingLabel)
                    currentX += 1
                else:
                    tilingLabel = (tilingLabel[0], tilingLabel[1] + 1)
                    tilingLayout[i][j] = str(tilingLabel)
                    currentX = 1
        print('abstraction_layout')
        print(np.array(tilingLayout))
        return tilingLayout

    def doTiling2(self, tilingSize, ignorewalls=True):
        columns = len(self.manuel_layout[0])
        rows = len(self.manuel_layout)
        tilingLayout = self.manuel_layout.copy().tolist()
        tilingLabel = (1, 1)
        for i in range(rows):
            if i != 0 and (i % tilingSize[1]) == 0:
                tilingLabel = (tilingLabel[0] + 1, 1)
            else:
                tilingLabel = (tilingLabel[0], 1)
            for j in range(columns):
                if j != 0 and (j % tilingSize[0]) == 0:
                    tilingLabel = (tilingLabel[0], tilingLabel[1] + 1)
                if ignorewalls:
                    tilingLayout[i][j] = str(tilingLabel)
                if not ignorewalls and not self.manuel_layout[i][j] == "w":
                    tilingLayout[i][j] = str(tilingLabel)
        return np.array(tilingLayout)

    def getAbstractState(self, state):
        abstract_state = [self.abstraction_layout[state[0], state[1]]]
        for i in range(2, len(state)):
            abstract_state.append(state[i])
        return abstract_state

    def getListOfAbstractState(self):
        # build all the potential abstract states
        list_of_abstract_states = []
        for i in range(len(self.abstraction_layout)):
            for j in range(len(self.abstraction_layout[0])):
                for k in range(2):
                    for l in range(2):
                        for m in range(2):
                            temp_astate = self.getAbstractState((i, j, k, l, m))  # return a list
                            if temp_astate not in list_of_abstract_states:
                                if not self.abstraction_layout[i][j] == "w":
                                    list_of_abstract_states.append(temp_astate)

        return list_of_abstract_states

    def getAjacencies(self):
        adjacencies = []
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    for i in range(len(self.abstraction_layout) - 1):
                        for j in range(len(self.abstraction_layout[0]) - 1):
                            current = self.getAbstractState((15 - i, j, k, l, m))
                            up = self.getAbstractState((15 - i - 1, j, k, l, m))
                            right = self.getAbstractState((15 - i, j + 1, k, l, m))
                            if not current == up and not current[0] == "w" and not up[0] == "w":
                                if (current, up) not in adjacencies:
                                    adjacencies.append((current, up))
                                    adjacencies.append((up, current))
                            if not current == right and not current[0] == "w" and not right[0] == "w":
                                if (current, right) not in adjacencies:
                                    adjacencies.append((current, right))
                                    adjacencies.append((right, current))
                        # designed for the state on the right border of the maze
                        j = len(self.abstraction_layout[0]) - 1
                        current = self.getAbstractState((15 - i, j, k, l, m))
                        up = self.getAbstractState((15 - i - 1, j, k, l, m))
                        if not current == up and not current[0] == "w" and not up[0] == "w":
                            if (current, up) not in adjacencies:
                                adjacencies.append((current, up))
                                adjacencies.append((up, current))
        self.adjacencies = adjacencies
        # get the adjacencies for each abstraction(tiling)
        adjacencies_for_each_abstract_state = []
        for a in self.list_of_abstract_states:
            adj = [a]
            for b in self.list_of_abstract_states:
                if (a, b) in self.adjacencies:
                    adj.append(b)
            adjacencies_for_each_abstract_state.append(adj)
        adjacencies_for_each_abstract_state.append(["bin", "bin"])
        self.adjacencies_for_each_astate = adjacencies_for_each_abstract_state

    def getAjacencies2(self):
        adjacencies = []
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    for i in range(len(self.abstraction_layout) - 1):
                        for j in range(len(self.abstraction_layout[0]) - 1):
                            current = self.getAbstractState((i, j, k, l, m))
                            down = self.getAbstractState((i + 1, j, k, l, m))
                            right = self.getAbstractState((i, j + 1, k, l, m))
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
                        current = self.getAbstractState((i, j, k, l, m))
                        down = self.getAbstractState((i + 1, j, k, l, m))
                        if not current == down and not current[0] == "w" and not down[0] == "w":
                            if (current, down) not in adjacencies:
                                adjacencies.append((current, down))
                                adjacencies.append((down, current))
                    # designed for the state on the bottom border of the maze
                    i = len(self.abstraction_layout) - 1
                    for j in range(len(self.abstraction_layout[0]) - 1):
                        current = self.getAbstractState((i, j, k, l, m))
                        right = self.getAbstractState((i, j + 1, k, l, m))
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

    def getAbstractActions(self, abstractState):  # get available adjacent astate and current visiting of F or G
        newlist = []
        for ajacency in self.adjacencies_for_each_astate:
            if ajacency[0] == abstractState:
                newlist = ajacency[1:]
                if not ajacency[0] == "bin" and ajacency[0][0] in self.flags_abstractions:
                    newlist.append("F" + ajacency[0][0])
                if not ajacency[0] == "bin" and ajacency[0][0] in self.goal_abstractions:
                    newlist.append("G" + ajacency[0][0])
                break
        return newlist
        # for astate in item[1:]:
        #     if astate[0] in self.flags_tilings:
        #         newlist.append("F"+astate[0])
        #     if astate[0] in self.goal_tiling:
        #         newlist.append("G"+astate[0])

    def setTransitionsAdvanced(self):  # consider the case that flags and goal are in the same tiling
        num_of_abstraction = len(self.list_of_abstract_states)
        rewards = np.zeros((num_of_abstraction + 4, num_of_abstraction, num_of_abstraction))
        transition = np.zeros((num_of_abstraction + 4, num_of_abstraction, num_of_abstraction))

        # actions_f = ["F"+ self.flags_tilings[f] for f in range(len(self.flags_tilings))]
        for i in range(num_of_abstraction):
            overlap = None
            ajacencies_of_i = self.getAbstractActions(self.list_of_abstract_states[i])
            for j in range(num_of_abstraction):
                # normal transition, nothing happens
                if self.list_of_abstract_states[j] in ajacencies_of_i:
                    transition[j, i, j] = 1

                # flag collection transition
                for f in range(len(self.flags_tilings)):
                    action_f = "F" + self.flags_tilings[f]
                    # action = [("F", self.flags_tilings[f]) for f in range(len(self.flags_tilings))]
                    if action_f in ajacencies_of_i:
                        if self.flags_tilings[f] == self.goal_tiling[0]:
                            overlap = f
                            continue
                        else:
                            s_prime = self.list_of_abstract_states[i]
                            s_prime[1 + f] = 1
                            s_prime[0] = self.flags_tilings[f]
                            if self.list_of_abstract_states[j] == s_prime:
                                # transition[num_of_abstraction + f, i, j] = 1
                                transition[j, i, j] = 1
                            # if s_prime[0] == self.goal_tiling[0]:
                            #     transition[j, i, j] = 1
                            #     earlystop=1

                # goal transition
                for g in range(len(self.goal_tiling)):
                    action_g = "G" + self.goal_tiling[g]
                    if action_g in ajacencies_of_i:
                        if not overlap == None:
                            s_prime = self.list_of_abstract_states[i]
                            s_prime[1 + overlap] = 1
                            s_prime[0] = self.goal_tiling[0]
                            if self.list_of_abstract_states[j] == s_prime:
                                transition[j, i, j] = 1
                        else:
                            s_prime = self.list_of_abstract_states[i]
                            s_prime[0] = self.goal_tiling[0]
                            if self.list_of_abstract_states[j] == s_prime:
                                transition[j, i, j] = 1

    def setTransitions(self):  # not consider flag and goal in the same tiling
        num_of_abstraction = len(self.list_of_abstract_states)
        rewards = np.zeros((num_of_abstraction + 4, num_of_abstraction, num_of_abstraction))
        transition = np.zeros((num_of_abstraction + 4, num_of_abstraction, num_of_abstraction))

        # actions_f = ["F"+ self.flags_tilings[f] for f in range(len(self.flags_tilings))]
        for i in range(num_of_abstraction):
            overlap = None
            ajacencies_of_i = self.getAbstractActions(self.list_of_abstract_states[i])
            for j in range(num_of_abstraction):
                # normal transition, nothing happens
                astate_j = self.list_of_abstract_states[j]
                if astate_j in ajacencies_of_i:
                    if astate_j[0] not in self.flags_tilings:
                        if astate_j[0] not in self.goal_tiling:
                            transition[j, i, j] = 1
                            continue
                # flag collection transition
                for f in range(len(self.flags_tilings)):
                    action_f = "F" + self.flags_tilings[f]
                    # action = [("F", self.flags_tilings[f]) for f in range(len(self.flags_tilings))]
                    if action_f in ajacencies_of_i:
                        s_prime = self.list_of_abstract_states[i]
                        s_prime[1 + f] = 1
                        s_prime[0] = self.flags_tilings[f]
                        if self.list_of_abstract_states[j] == s_prime:
                            transition[j, i, j] = 1
                # goal transition
                for g in range(len(self.goal_tiling)):
                    action_g = "G" + self.goal_tiling[g]
                    if action_g in ajacencies_of_i:
                        s_prime = self.list_of_abstract_states[i]
                        s_prime[0] = self.goal_tiling[0]
                        if self.list_of_abstract_states[j] == s_prime:
                            transition[j, i, j] = 1

        self.transition = transition

    def getTransitionsAndRewardsWithPB(self):
        self.list_of_abstract_states.append("bin")
        num_of_abstract_state = len(self.list_of_abstract_states)
        ## Initialise empty action, State, State' transitions and reward
        transition = np.zeros((num_of_abstract_state + 4, num_of_abstract_state, num_of_abstract_state))
        rewards = np.zeros((num_of_abstract_state + 4, num_of_abstract_state, num_of_abstract_state))

        # actions_f = ["F"+ self.flags_tilings[f] for f in range(len(self.flags_tilings))]
        for i in range(num_of_abstract_state):
            ajacency_of_i = self.getAbstractActions(self.list_of_abstract_states[i])
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
                ajacency_of_i = self.getAbstractActions(self.list_of_abstract_states[i])
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

    def solveAbstraction(self):
        print('===================lenth of self.listOfAbstractStates:', len(self.list_of_abstract_states))
        print('===================self.listOfAbstractStates:', self.list_of_abstract_states)
        V = np.zeros(len(self.list_of_abstract_states))
        print("len(V):",len(V))
        # print("self.transition_table:",np.argwhere(self.transition_table))
        # print("self.rewards_table:",np.argwhere(self.rewards_table))
        delta = 0.2
        theta = 0.1
        print("Value Iteration delta values:")
        while delta > theta:
            delta = 0
            for i in range(0, len(V)):
                v = V[i]
                listOfValues = []
                for a in range(len(self.list_of_abstract_states)+4):
                    value = 0
                    for j in range(len(V)):
                        value += self.transition_table[a, i, j] * (self.rewards_table[a, i, j] + 0.99 * V[j])
                    listOfValues.append(value)
                V[i] = max(listOfValues)
                delta = max(delta, abs(v - V[i]))
            print("delta:", delta)
        # print(V)
        V -= min(V[:-1])
        self.V = V
        print("self.V")
        print(self.V)

    def getValue(self, astate):
        value = self.V[self.list_of_abstract_states.index(astate)]
        # print("value:",value)
        return value

if __name__ == "__main__":
    from maze_env_general import Maze

    env = Maze(maze='open_space')  # initialize env 可修改
    amdp = AMDP(env=env, tiling_mode=None, dw_clt_layout=None)
    # print(amdp.abstraction_layout)
    for i in range(len(amdp.adjacencies_for_each_astate)):
        print(amdp.adjacencies_for_each_astate[i] )
    # print("self.rewards_table:", np.argwhere(amdp.rewards_table))
    amdp.solveAbstraction()
    for item in amdp.list_of_abstract_states:
        print("astate and value:", item, amdp.getValue(item))

