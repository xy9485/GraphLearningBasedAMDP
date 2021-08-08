import copy
import math

import numpy as np
import time
import sys
import random


class Maze:

    def __init__(self, stochasticity=None, maze='basic', big=0):
        print("env oh yeah!!!!!")
        self.maze_name = maze
        self.big = big
        self.stochasticity = stochasticity
        self.num_of_actions = 4
        self.room_layout = self.get_room_layout() # return a nparray
        self.size = (len(self.room_layout), len(self.room_layout[0]))
        print("maze.size:", self.size)
        self.reset()
        self.start_state = self.state
        self.valid_coords, self.valid_states = self.get_valid_coords_and_states()
        print("len(env.valid_coords)", len(self.valid_coords))
        print("len(env.valid_states)", len(self.valid_states))
        self.print_maze_info()


    def print_maze_info(self):
        print("env.name:", self.maze_name)
        print("env.big:", self.big)
        print("env.flags:", self.flags, self.room_layout[self.flags[0][0], self.flags[0][1]],
              self.room_layout[self.flags[1][0], self.flags[1][1]], self.room_layout[self.flags[2][0], self.flags[2][1]])
        print("env.goal:", self.goal, self.room_layout[self.goal[0], self.goal[1]])
        print("env.state:", self.state)

    def get_room_layout(self):

        A = 0
        B = 1
        C = 2
        D = 3
        E = 4
        F = 5
        G = 6
        H = 7
        I = 8
        J = 9
        K = 10
        L = 11
        M = 12
        N = 13
        O = 14
        P = 15
        Q = "q"
        R = "r"
        S = "s"
        T = "t"
        U = "u"
        V = "v"
        X = "x"
        Y = "y"
        Z = "z"
        AA = "aa"
        BB = "bb"
        CC = "cc"
        DD = "dd"
        EE = "ee"
        FF = "ff"
        GG = "gg"
        SS = "ss"
        W = "w"
        # Z = np.ones(10)
        if self.maze_name == 'basic':
            ## "True" layout determined by doorways.
            room_layout = [[C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
                          [C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
                          [C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
                          [C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
                          [W, W, W, W, C, W, W, W, W, W, W, D, D, D, D, W, F, F, F, F, F],
                          [B, B, B, B, B, B, B, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
                          [B, B, B, B, B, B, W, E, E, E, E, D, D, D, D, W, F, F, F, F, F],
                          [B, B, B, B, B, B, W, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
                          [W, A, W, W, W, W, W, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
                          [A, A, A, A, A, A, W, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
                          [A, A, A, A, A, A, W, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
                          [A, A, A, A, A, A, W, W, E, W, W, W, W, W, W, W, F, F, F, F, F],
                          [A, A, A, A, A, A, W, G, G, G, G, G, G, G, G, W, F, F, F, F, F],
                          [A, A, A, A, A, A, W, G, G, G, G, G, G, G, G, W, F, F, F, F, F],
                          [A, A, A, A, A, A, W, G, G, G, G, G, G, G, G, F, F, F, F, F, F],
                          [A, A, A, A, A, A, W, G, G, G, G, G, G, G, G, W, F, F, F, F, F]]
            room_layout = np.array(room_layout)
            # self.walls = np.argwhere(roomLayout == 'w').tolist()

        elif self.maze_name == 'basic2':
            ## "True" layout determined by doorways.
            room_layout = [[C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
                          [C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
                          [C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
                          [C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
                          [W, W, W, W, Z, W, W, W, W, W, W, D, D, D, D, W, F, F, F, F, F],
                          [B, B, B, B, B, B, Z, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
                          [B, B, B, B, B, B, W, E, E, E, Z, D, D, D, D, W, F, F, F, F, F],
                          [B, B, B, B, B, B, W, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
                          [W, Z, W, W, W, W, W, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
                          [A, A, A, A, A, A, W, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
                          [A, A, A, A, A, A, W, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
                          [A, A, A, A, A, A, W, W, Z, W, W, W, W, W, W, W, F, F, F, F, F],
                          [A, A, A, A, A, A, W, G, G, G, G, G, G, G, G, W, F, F, F, F, F],
                          [A, A, A, A, A, A, W, G, G, G, G, G, G, G, G, W, F, F, F, F, F],
                          [A, A, A, A, A, A, W, G, G, G, G, G, G, G, G, Z, F, F, F, F, F],
                          [A, A, A, A, A, A, W, G, G, G, G, G, G, G, G, W, F, F, F, F, F]]
            room_layout = np.array(room_layout)
        elif self.maze_name == "simple":
            room_layout = [[D, D, D, F, F, F, F, F, F, F, W, H, H, H, H, H],  ## simple2
                           [D, D, D, W, F, F, F, F, F, F, H, H, H, H, H, H],
                           [D, D, D, W, F, F, F, F, F, F, W, H, H, H, H, H],
                           [D, W, W, W, W, W, W, W, W, W, W, H, H, H, H, H],
                           [G, G, G, G, G, G, G, W, E, E, W, H, H, H, H, H],
                           [G, G, G, G, G, G, G, G, E, E, W, H, H, H, H, H],
                           [G, G, G, G, G, G, G, G, E, E, W, W, W, W, B, B],
                           [W, W, W, G, G, G, G, W, E, E, E, E, E, W, B, B],
                           [A, A, W, W, W, W, W, W, E, E, E, E, E, W, B, B],
                           [A, A, C, C, C, C, C, C, E, E, E, E, E, W, B, B],
                           [A, A, W, C, C, C, C, C, E, E, E, E, E, W, B, B]]
            room_layout = np.array(room_layout)

        elif self.maze_name == 'strips':
            room_layout = [[A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],  ## Strips
                          [A, A, W, B, B, B, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, B, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, F, G],
                          [A, A, W, B, B, W, C, C, C, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, C, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, A, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, E, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, E, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, D, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G]]
            room_layout = np.array(room_layout)
            # self.walls = np.argwhere(roomLayout == 'w').tolist()

        elif self.maze_name == 'strips2':
            room_layout = [[A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],  ## Strips
                          [A, A, W, B, B, B, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, B, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, F, G, G],
                          [A, A, W, B, B, W, C, C, C, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, C, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, A, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, E, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, E, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, D, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G]]
            room_layout = np.array(room_layout)
            # self.walls = np.argwhere(roomLayout == 'w').tolist()

        elif self.maze_name == 'strips3':
            room_layout = [[A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],  ## Strips
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, Z, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, Z, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, Z, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, Z, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, Z, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, Z, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
                          [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G]]
            room_layout = np.array(room_layout)
            # self.walls = np.argwhere(roomLayout == 'w').tolist()

        elif self.maze_name == 'spiral':
            room_layout = [[C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, B, B],  ## Spiral
                          [C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, B, B],
                          [D, D, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, B, B],
                          [D, D, W, G, G, G, G, G, G, G, G, G, G, G, G, F, F, W, B, B],
                          [D, D, W, G, G, G, G, G, G, G, G, G, G, G, G, F, F, W, B, B],
                          [D, D, W, H, H, W, W, W, W, W, W, W, W, W, W, F, F, W, B, B],
                          [D, D, W, H, H, W, K, K, K, K, K, K, J, J, W, F, F, W, B, B],
                          [D, D, W, H, H, W, K, K, K, K, K, K, J, J, W, F, F, W, B, B],
                          [D, D, W, H, H, W, L, L, W, W, W, W, J, J, W, F, F, W, B, B],
                          [D, D, W, H, H, W, L, L, M, M, M, W, J, J, W, F, F, W, B, B],
                          [D, D, W, H, H, W, L, L, M, M, M, W, J, J, W, F, F, W, B, B],
                          [D, D, W, H, H, W, W, W, W, W, W, W, J, J, W, F, F, W, B, B],
                          [D, D, W, H, H, I, I, I, I, I, I, I, I, I, W, F, F, W, B, B],
                          [D, D, W, H, H, I, I, I, I, I, I, I, I, I, W, F, F, W, B, B],
                          [D, D, W, W, W, W, W, W, W, W, W, W, W, W, W, F, F, W, B, B],
                          [D, D, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W, B, B],
                          [D, D, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W, B, B],
                          [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, B, B],
                          [A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
                          [A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A]]
            room_layout = np.array(room_layout)
            # self.walls = np.argwhere(roomLayout == 'w').tolist()

        elif self.maze_name == 'open_space':
            room_layout = [[C, C, C, C, C, C, C, C, C, C, C, C, C, E, E, E, E, W, W, W],  ## Open Space
                          [W, W, W, W, W, C, C, C, C, C, C, C, C, E, E, W, E, E, W, W],
                          [W, W, W, W, W, C, C, C, C, W, W, W, D, E, E, E, W, E, E, W],
                          [W, W, W, W, W, C, C, D, D, D, D, D, D, E, E, E, W, W, E, E],
                          [B, B, B, B, B, B, D, D, D, D, D, D, D, E, E, E, E, W, W, E],
                          [B, B, B, B, B, B, D, D, D, D, D, D, D, E, E, E, E, E, E, E],
                          [B, B, B, B, B, B, D, D, D, W, D, D, D, E, E, E, E, E, E, E],
                          [B, B, B, B, B, B, D, D, W, W, W, D, D, E, E, E, E, E, E, E],
                          [B, B, B, B, B, B, D, W, W, W, W, W, F, F, F, F, F, F, W, W],
                          [W, W, W, B, B, B, H, H, W, W, W, F, F, F, F, F, F, W, W, W],
                          [W, W, W, B, B, B, H, H, H, W, H, F, F, F, F, F, F, F, W, W],
                          [A, A, A, B, B, B, H, H, H, H, H, H, W, W, G, G, G, G, G, G],
                          [A, A, A, B, B, B, H, H, H, H, H, H, W, W, G, G, G, G, G, G],
                          [A, A, A, B, B, B, H, H, H, H, H, G, G, G, G, G, G, G, G, G],
                          [A, A, A, B, B, B, H, H, H, H, H, G, G, G, G, G, G, G, G, G],
                          [A, A, A, W, W, W, W, H, H, H, W, G, G, G, G, G, G, G, G, G],
                          [A, A, A, W, W, W, W, H, H, H, W, G, G, G, G, W, W, G, G, G],
                          [A, A, A, W, W, W, W, H, H, H, W, G, G, G, G, W, W, G, G, G],
                          [A, A, A, W, W, W, W, H, H, H, W, G, G, G, G, W, W, G, G, G],
                          [A, A, A, A, A, A, A, H, H, H, G, G, G, G, G, W, W, G, G, G]]
            room_layout = np.array(room_layout)
            # self.walls = np.argwhere(roomLayout == 'w').tolist()

        elif self.maze_name == 'high_connectivity':
            room_layout =[[A, A, A, W, H, H, H, I, I, I, I, I, I, I, W, J, J, J, J, J],  ##High Connectivity
                          [A, A, A, W, H, H, H, W, I, I, I, I, I, I, J, J, J, J, J, J],
                          [A, A, A, H, H, H, H, W, I, I, I, I, I, I, W, J, J, J, J, J],
                          [A, A, A, W, W, W, H, W, W, W, I, W, Z, W, W, J, J, J, J, J],
                          [A, A, A, W, G, G, G, G, G, W, K, W, P, P, W, J, J, J, J, J],
                          [A, A, A, W, G, G, G, G, G, K, K, W, P, P, W, W, W, J, J, J],
                          [A, A, A, G, G, G, G, G, G, W, K, W, P, P, P, P, W, Z, W, W],
                          [A, A, A, W, W, W, W, G, G, W, K, W, P, P, P, P, P, P, P, P],
                          [A, A, A, A, A, A, W, F, W, W, W, W, P, P, P, P, P, P, P, P],
                          [A, A, A, A, A, A, F, F, F, F, F, W, P, P, P, P, P, P, P, P],
                          [A, A, A, A, A, A, W, F, F, F, F, P, P, P, P, P, P, P, P, P],
                          [W, W, W, Z, W, W, W, Z, W, W, W, W, P, W, W, W, W, P, W, W],
                          [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
                          [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, O, N, N, N],
                          [B, B, B, B, Z, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
                          [B, B, B, B, E, W, E, E, E, W, W, Z, W, W, O, O, W, N, N, N],
                          [W, C, W, W, E, W, W, W, E, W, L, L, L, W, O, W, W, W, N, W],
                          [C, C, C, W, W, W, D, D, D, W, L, L, L, W, W, M, M, M, M, M],
                          [C, C, C, C, D, D, D, D, D, W, L, L, L, L, M, M, M, M, M, M],
                          [C, C, C, W, W, W, D, D, D, L, L, L, L, L, W, M, M, M, M, M]]
            room_layout = np.array(room_layout)
            # self.walls = np.argwhere(roomLayout == 'w').tolist()

        elif self.maze_name == 'low_connectivity':
            room_layout = [[A, A, A, W, H, H, H, I, I, I, I, I, I, I, W, J, J, J, J, J],  ##Low Connectivity
                          [A, A, A, W, H, H, H, W, I, I, I, I, I, I, J, J, J, J, J, J],
                          [A, A, A, W, H, H, H, W, I, I, I, I, I, I, W, J, J, J, J, J],
                          [A, A, A, W, W, W, H, W, W, W, I, W, W, W, W, J, J, J, J, J],
                          [A, A, A, W, G, G, G, G, G, W, K, W, P, P, W, J, J, J, J, J],
                          [A, A, A, W, G, G, G, G, G, K, K, W, P, P, W, W, W, J, J, J],
                          [A, A, A, W, G, G, G, G, G, W, K, W, P, P, P, P, W, W, W, W],
                          [A, A, A, W, W, W, W, G, G, W, K, W, P, P, P, P, P, P, P, P],
                          [A, A, A, A, A, A, W, F, W, W, W, W, P, P, P, P, P, P, P, P],
                          [A, A, A, A, A, A, W, F, F, F, F, W, P, P, P, P, P, P, P, P],
                          [A, A, A, A, A, A, W, F, F, F, F, P, P, P, P, P, P, P, P, P],
                          [W, W, W, B, W, W, W, W, W, W, W, W, P, W, W, W, W, W, W, W],
                          [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
                          [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, O, N, N, N],
                          [B, B, B, B, E, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
                          [B, B, B, B, E, W, E, E, E, W, W, W, W, W, O, O, W, N, N, N],
                          [W, C, W, W, E, W, W, W, E, W, L, L, L, W, O, W, W, W, N, W],
                          [C, C, C, W, W, W, D, D, D, W, L, L, L, W, W, M, M, M, M, M],
                          [C, C, C, W, D, D, D, D, D, W, L, L, L, L, M, M, M, M, M, M],
                          [C, C, C, W, W, W, D, D, D, L, L, L, L, L, W, M, M, M, M, M]]
            room_layout = np.array(room_layout)
            # self.walls = np.argwhere(roomLayout == 'w').tolist()

        elif self.maze_name == 'low_connectivity2':
            room_layout = [[A, A, A, W, H, H, H, I, I, I, I, I, I, I, W, J, J, J, J, J],  ##Low Connectivity
                          [A, A, A, W, H, H, H, W, I, I, I, I, I, I, J, J, J, J, J, J],
                          [A, A, A, W, H, H, H, W, I, I, I, I, I, I, W, J, J, J, J, J],
                          [A, A, A, W, W, W, H, W, W, W, K, W, W, W, W, J, J, J, J, J],
                          [A, A, A, W, G, G, G, G, G, W, K, W, P, P, W, J, J, J, J, J],
                          [A, A, A, W, G, G, G, G, G, W, K, W, P, P, W, W, W, J, J, J],
                          [A, A, A, W, G, G, G, G, G, W, K, W, P, P, P, P, W, W, W, W],
                          [A, A, A, W, W, W, W, G, G, W, K, W, P, P, P, P, P, P, P, P],
                          [A, A, A, A, A, A, W, F, W, W, W, W, P, P, P, P, P, P, P, P],
                          [A, A, A, A, A, A, W, F, F, F, F, W, P, P, P, P, P, P, P, P],
                          [A, A, A, A, A, A, Z, F, F, F, F, P, P, P, P, P, P, P, P, P],
                          [W, W, W, B, W, W, W, W, W, W, W, W, P, W, W, W, W, W, W, W],
                          [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
                          [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, O, N, N, N],
                          [B, B, B, B, E, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
                          [B, B, B, B, E, W, E, E, E, W, W, W, W, W, O, O, W, N, N, N],
                          [W, C, W, W, E, W, W, W, E, W, L, L, L, W, O, W, W, W, N, W],
                          [C, C, C, W, W, W, D, D, D, W, L, L, L, W, W, M, M, M, M, M],
                          [C, C, C, W, D, D, D, D, D, W, L, L, L, L, M, M, M, M, M, M],
                          [C, C, C, W, W, W, D, D, D, L, L, L, L, L, W, M, M, M, M, M]]
            room_layout = np.array(room_layout)
            # self.walls = np.argwhere(roomLayout == 'w').tolist()

        elif self.maze_name == 'low_connectivity3':
            room_layout = [[A, A, A, W, H, H, H, Z, I, I, I, I, I, I, W, J, J, J, J, J],  ##Low Connectivity3
                          [A, A, A, W, H, H, H, W, I, I, I, I, I, I, Z, J, J, J, J, J],
                          [A, A, A, W, H, H, H, W, I, I, I, I, I, I, W, J, J, J, J, J],
                          [A, A, A, W, W, W, Z, W, W, W, K, W, W, W, W, J, J, J, J, J],
                          [A, A, A, W, G, G, G, G, G, W, K, W, P, P, W, J, J, J, J, J],
                          [A, A, A, W, G, G, G, G, G, W, K, W, P, P, W, W, W, J, J, J],
                          [A, A, A, W, G, G, G, G, G, W, K, W, P, P, P, P, W, W, W, W],
                          [A, A, A, W, W, W, W, G, G, W, K, W, P, P, P, P, P, P, P, P],
                          [A, A, A, A, A, A, W, Z, W, W, W, W, P, P, P, P, P, P, P, P],
                          [A, A, A, A, A, A, W, F, F, F, F, W, P, P, P, P, P, P, P, P],
                          [A, A, A, A, A, A, W, F, F, F, F, Z, P, P, P, P, P, P, P, P],
                          [W, W, W, Z, W, W, W, W, W, W, W, W, Z, W, W, W, W, W, W, W],
                          [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
                          [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, Z, N, N, N],
                          [B, B, B, B, Z, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
                          [B, B, B, B, B, W, E, E, E, W, W, W, W, W, O, O, W, N, N, N],
                          [W, Z, W, W, B, W, W, W, Z, W, L, L, L, W, O, W, W, W, Z, W],
                          [C, C, C, W, W, W, D, D, D, W, L, L, L, W, W, M, M, M, M, M],
                          [C, C, C, W, D, D, D, D, D, W, L, L, L, L, Z, M, M, M, M, M],
                          [C, C, C, W, W, W, D, D, D, Z, L, L, L, L, W, M, M, M, M, M]]
            room_layout = np.array(room_layout)
            # self.walls = np.argwhere(roomLayout == 'w').tolist()

        elif self.maze_name.startswith('external'):
            # path = f"/Users/yuan/PycharmProjects/Masterthesis/external_mazes/{self.maze_name}.txt"
            # path = f"/home/xue/projects/masterthesis/external_mazes/{self.maze_name}.txt"
            path = f"/workspace/masterthesis/external_mazes/{self.maze_name}.txt"
            # path = f"external_mazes/{self.maze_name}.txt"
            room_layout = []
            with open(path, "r") as f:
                content = f.readlines()
                content = [x.strip() for x in content]
                for item in content:
                    item = item.replace('#', 'w')
                    # item = item.replace('.', '0')
                    row = [x for x in item]
                    room_layout.append(row)
            room_layout = np.array(room_layout)
            # self.walls = np.argwhere(roomLayout == 'w').tolist()

        else:
            raise Exception("invalide maze name")

        opens_small = np.argwhere(room_layout == 'z').tolist()
        self.opens_small = {str(i) for i in opens_small}
        self.choose_stochastic_opens()

        if self.big:
            newLayout = []
            for l in room_layout:
                nextLine = []
                for x in l:
                    nextLine.extend([x, x, x])
                newLayout.append(nextLine)
                newLayout.append(nextLine)
                newLayout.append(nextLine)
            room_layout = np.array(newLayout)
        walls = np.argwhere(room_layout == 'w').tolist()
        self.walls = {str(i) for i in walls}
        self.walls_original = copy.deepcopy(self.walls)
        traps = np.argwhere(room_layout == 't').tolist()
        self.traps = {str(i) for i in traps}
        opens = np.argwhere(room_layout == 'z').tolist()
        self.opens = {str(i) for i in opens}
        # len1 = len(random.sample(self.opens,1)[0])
        # len2 = len(random.sample(self.opens_2be_stochastic2,1)[0])
        return room_layout


    def choose_stochastic_opens(self):
        if self.stochasticity:
            opens_2be_stochastic1 = set(random.sample(self.opens_small, math.floor(len(self.opens_small) * self.stochasticity['p2be_stochastic'])))
            opens_2be_stochastic2 = set()
            if self.big:
                for term in opens_2be_stochastic1:
                    for t1 in range(3):
                        for t2 in range(3):
                            x = (eval(term)[0]) * 3 + t1
                            y = (eval(term)[1]) * 3 + t2
                            opens_2be_stochastic2.add(f"[{x}, {y}]")
            else:
                opens_2be_stochastic2 = opens_2be_stochastic1
            self.opens_2be_stochastic2 = opens_2be_stochastic2

    def update_walls(self, move_count):
        if self.stochasticity and move_count % self.stochasticity['interval'] == 0:
            self.walls = self.walls_original
            if np.random.randint(100) < self.stochasticity['p2be_closed'] * 100:    # probability for chosen opens to become untraversible
                self.walls = self.walls_original.union(self.opens_2be_stochastic2)

    # def update_traps(self):
    #     traps = np.argwhere(self.room_layout == 't').tolist()
    #     self.traps = {str(i) for i in traps}



    def isTerminal(self, state):
        return (state[0], state[1]) == self.goal  # self.goal is a tuple

    def reset(self):
        self.flags_found_order = []
        self.flags_collected = 0
        self.flags_collected2 = [0, 0, 0]
        if self.maze_name == 'basic':
            # self.state = (6, 4, 0, 0, 0)          #v1
            # self.flags = [(0, 5), (15, 0), (15, 20)]
            # self.goal = (14, 1)

            self.state = (6, 1, 0, 0, 0)        # v2
            self.flags = [(0, 5), (0, 7), (4, 20)]
            self.goal = (14, 1)

            # self.state = (6, 1, 0, 0, 0)        # v3
            # self.flags = [(0, 5), (1, 14), (1, 16)]
            # self.goal = (14, 1)

            # self.state = (6, 1, 0, 0, 0)        # v4
            # self.flags = [(0, 5), (15, 7), (1, 16)]
            # self.goal = (15, 18)

            # self.state = (15, 0, 0, 0, 0)    # v5
            # self.flags = [(10, 14), (0, 7), (0, 16)]
            # self.goal = (0, 5)
            # self.traps = [(15, 7), (15, 8)]
        elif self.maze_name == 'basic2':
            # self.state = (6, 4, 0, 0, 0)          #v1
            # self.flags = [(0, 5), (15, 0), (15, 20)]
            # self.goal = (14, 1)

            self.state = (6, 1, 0, 0, 0)        # v2
            self.flags = [(0, 5), (0, 7), (4, 20)]
            self.goal = (14, 1)

            # self.state = (6, 1, 0, 0, 0)        # v3
            # self.flags = [(0, 5), (1, 14), (1, 16)]
            # self.goal = (14, 1)

            # self.state = (6, 1, 0, 0, 0)        # v4
            # self.flags = [(0, 5), (15, 7), (1, 16)]
            # self.goal = (15, 18)

            # self.state = (15, 0, 0, 0, 0)    # v5
            # self.flags = [(10, 14), (0, 7), (0, 16)]
            # self.goal = (0, 5)
            # self.traps = [(15, 7), (15, 8)]
        elif self.maze_name == "simple":
            self.state = (2, 2, 0, 0, 0)  # v2
            self.flags = [(0, 1), (0, 2), (0, 3)]
            self.goal = (1, 1)
        elif self.maze_name == 'strips':
            self.state = (0, 0, 0, 0, 0)
            self.flags = [(15, 11), (19, 0), (4, 19)]
            self.goal = (18, 1)
        elif self.maze_name == 'strips2':
            self.state = (0, 0, 0, 0, 0)
            self.flags = [(18, 3), (15, 11), (19, 19)]
            self.goal = (18, 1)
        elif self.maze_name == 'strips3':
            self.state = (0, 0, 0, 0, 0)
            self.flags = [(18, 3), (15, 11), (19, 19)]
            self.goal = (18, 1)
        elif self.maze_name == 'spiral':
            # self.state = (19, 0, 0, 0, 0)       #v1
            # self.flags = [(0, 19), (15, 6), (6, 6)]
            # self.goal = (13, 13)

            self.state = (19, 0, 0, 0, 0)       #v2
            self.flags = [(0, 19), (15, 6), (10, 10)]
            self.goal = (0, 0)

            # self.state = (19, 0, 0, 0, 0)         # v3
            # self.flags = [(3, 3), (16, 16), (10, 10)]
            # self.goal = (16, 18)
        elif self.maze_name == 'open_space':
            self.state = (19, 0, 0, 0, 0)
            self.flags = [(0, 0), (2, 17), (13, 14)]
            # self.goal = (19, 3)  # v1
            self.goal = (19, 19)   # v2
        elif self.maze_name == 'high_connectivity':
            self.state = (19, 0, 0, 0, 0)
            self.flags = [(0, 1), (2, 18), (5, 6)]
            self.goal = (15, 0)
        elif self.maze_name == 'low_connectivity':
            self.state = (19, 0, 0, 0, 0)
            self.flags = [(0, 1), (2, 18), (5, 6)]
            self.goal = (15, 0)
            # self.goal = (4, 19)
        elif self.maze_name == 'low_connectivity2':
            self.state = (19, 0, 0, 0, 0)
            self.flags = [(0, 1), (2, 18), (5, 6)]
            self.goal = (15, 0)
            # self.goal = (4,19)
        elif self.maze_name == 'low_connectivity3':
            self.state = (19, 0, 0, 0, 0)
            self.flags = [(0, 1), (2, 18), (5, 6)]
            self.goal = (15, 0)
            # self.goal = (4,19)
        elif self.maze_name == "external_maze21x21_1":
            self.state = (1, 1, 0, 0, 0)
            self.goal = (1, 19)
            # self.goal = (19, 19)
            self.flags = [(10,1),(1,17),(17,3)]
        elif self.maze_name == "external_maze21x21_2":
            self.state = (0, 0, 0, 0, 0)
            self.goal = (1, 19)
            # self.goal = (19, 19)
            self.flags = [(10, 1), (1, 17), (17, 3)]
        elif self.maze_name == "external_maze21x21_3":
            self.state = (0, 0, 0, 0, 0)
            self.goal = (20, 20)
            # self.goal = (19, 19)
            self.flags = [(9, 1), (1, 17), (15, 11)]
        elif self.maze_name == "external_maze31x31_1":
            self.state = (0, 0, 0, 0, 0)
            # self.goal = (1, 20)
            self.goal = (29, 29)
            self.flags = [(15, 2), (7, 27), (29, 15)]
        elif self.maze_name == "external_maze31x31_2":
            self.state = (0, 0, 0, 0, 0)
            # self.goal = (1, 20)
            self.goal = (5, 20)
            self.flags = [(29, 2), (15, 16), (5, 23)]
        elif self.maze_name == "external_maze61x61_1":
            self.state = (0, 0, 0, 0, 0)
            self.goal = (1, 57)
            self.flags = [(9, 1), (45, 58), (59, 25)]

        elif self.maze_name == "external_low_connectivity_1":
            self.state = (0, 0, 0, 0, 0)
            self.goal = (1, 18)
            self.flags = [(10, 16), (12, 16), (15, 15)]
        else:
            print("no matched maze")
        if self.big:
            self.state = tuple([i*3 for i in self.state])
            self.goal = tuple([i*3 for i in self.goal])
            flags_big = []
            for item in self.flags:
                item_prime = tuple([i*3 for i in item])
                flags_big.append(item_prime)
            self.flags = flags_big

    def get_valid_coords_and_states(self):
        valid_coords = []
        valid_states = []
        template = self.room_layout
        templateX = len(template[0])  # num of columns
        templateY = len(template)  # num of rows
        for i in range(templateY):
            for j in range(templateX):
                if template[i, j] != "w":
                    current_coord = (i, j)
                    valid_coords.append(current_coord)
                else:
                    continue
                for k in range(2):
                    for l in range(2):
                        for m in range(2):
                            current_state = [i, j, k, l, m]
                            if current_coord in self.flags:
                                index = self.flags.index(current_coord)
                                current_state[2+index] = 1
                            valid_states.append(tuple(current_state))

        return set(valid_coords), set(valid_states)
        # return list(set(valid_coords)), list(set(valid_states))

    def isMovable(self, state):
        # check if wall is in the way or already out of the bounds
        if state[0] < 0 or state[0] > (len(self.room_layout) - 1):
            return False
        elif state[1] < 0 or state[1] > (len(self.room_layout[0]) - 1):
            return False
        elif str([state[0], state[1]]) in self.walls:
            return False
        else:
            return True

    def actions(self, state):
        actions = []
        for a in range(0, 4):
            if self.isMovable(self.step(state, a)):
                actions.append(a)
        return actions

    def step(self, state, action):
        newCoord = (0, 0)
        if action == 0:  # right
            newCoord = (state[0], state[1] + 1)
        elif action == 1:  # down
            newCoord = (state[0] + 1, state[1])
        elif action == 2:  # left
            newCoord = (state[0], state[1] - 1)
        elif action == 3:  # up
            newCoord = (state[0] - 1, state[1])
        else:
            pass
        if newCoord == self.flags[0]:
            return (newCoord[0], newCoord[1], 1, state[3], state[4])
        elif newCoord == self.flags[1]:
            return (newCoord[0], newCoord[1], state[2], 1, state[4])
        elif newCoord == self.flags[2]:
            return (newCoord[0], newCoord[1], state[2], state[3], 1)
        else:
            return (newCoord[0], newCoord[1], state[2], state[3], state[4])

    def reward(self, state, action, state_prime):
        flag_number = -1
        for index in range(0, len(self.flags)):
            if (state_prime[0], state_prime[1]) == self.flags[index]:
                flag_number = index
                # if index not in self.flags_found_order:
                #     self.flags_found_order.append(index)
        if flag_number > -1:
            if state[flag_number + 2] == 0:
                self.flags_collected += 1
                # return 10000 * self.flags_collected
                return 10000
        if str([state_prime[0], state_prime[1]]) in self.traps:
            return -100000
        if (state_prime[0], state_prime[1]) == self.goal and self.flags_collected == 3:
        # if (state_prime[0], state_prime[1]) == self.goal:
            # return self.flags_collected * 1000
            return 0
        return -1

# M=Maze()
# print(M.walls)
if __name__ == "__main__":
    Maze(maze='external_maze2_21x21')
    print("bingo!")