import numpy as np
import time
import sys


class Maze():

    def __init__(self, maze='basic', big=0):
        print("env oh yeah!!!!!")
        self.maze_name = maze
        self.big = big
        self.room_layout = self.getRoomLayout() # return a nparray
        self.size = (len(self.room_layout), len(self.room_layout[0]))
        print("maze.size:", self.size)
        self.reset()
        self.valid_nodes = self.get_valid_nodes()
        print("len(env.valid_nodes)", len(self.valid_nodes))

    def getRoomLayout(self):

        A = "a"
        B = "b"
        C = "c"
        D = "d"
        E = "e"
        F = "f"
        G = "g"
        H = "h"
        I = "i"
        J = "j"
        K = "k"
        L = "l"
        M = "m"
        N = "n"
        O = "o"
        P = "p"
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
        rooms = []
        if self.maze_name == 'basic':
            ## "True" layout determined by doorways.
            roomLayout = [[C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
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
            roomLayout = np.array(roomLayout)
            self.walls = np.argwhere(roomLayout == 'w').tolist()
            if self.big:
                newLayout = []
                for l in roomLayout:
                    nextLine = []
                    for x in l:
                        nextLine.extend([x, x, x])
                    newLayout.append(nextLine)
                    newLayout.append(nextLine)
                    newLayout.append(nextLine)
                roomLayout = np.array(newLayout)
                self.walls = np.argwhere(roomLayout == 'w').tolist()
            return roomLayout

        # if self.maze_name == 'big_basic':
        #     ## "True" layout determined by doorways.
        #     roomLayout = [[C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
        #                   [C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
        #                   [C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
        #                   [C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
        #                   [W, W, W, W, C, W, W, W, W, W, W, D, D, D, D, W, F, F, F, F, F],
        #                   [B, B, B, B, B, B, B, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
        #                   [B, B, B, B, B, B, W, E, E, E, E, D, D, D, D, W, F, F, F, F, F],
        #                   [B, B, B, B, B, B, W, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
        #                   [W, A, W, W, W, W, W, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
        #                   [A, A, A, A, A, A, W, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
        #                   [A, A, A, A, A, A, W, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
        #                   [A, A, A, A, A, A, W, W, E, W, W, W, W, W, W, W, F, F, F, F, F],
        #                   [A, A, A, A, A, A, W, G, G, G, G, G, G, G, G, W, F, F, F, F, F],
        #                   [A, A, A, A, A, A, W, G, G, G, G, G, G, G, G, W, F, F, F, F, F],
        #                   [A, A, A, A, A, A, W, G, G, G, G, G, G, G, G, F, F, F, F, F, F],
        #                   [A, A, A, A, A, A, W, G, G, G, G, G, G, G, G, W, F, F, F, F, F]]
        #     newLayout = []
        #     for l in roomLayout:
        #         nextLine = []
        #         for x in l:
        #             nextLine.extend([x, x, x])
        #         newLayout.append(nextLine)
        #         newLayout.append(nextLine)
        #         newLayout.append(nextLine)
        #     roomLayout = np.array(newLayout)
        #     self.walls = np.argwhere(roomLayout == 'w').tolist()
        #     return roomLayout

        if self.maze_name == 'strips':
            roomLayout = [[A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],  ## Strips
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
            roomLayout = np.array(roomLayout)
            self.walls = np.argwhere(roomLayout == 'w').tolist()
            return roomLayout

        if self.maze_name == 'strips2':
            roomLayout = [[A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],  ## Strips
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
            roomLayout = np.array(roomLayout)
            self.walls = np.argwhere(roomLayout == 'w').tolist()
            return roomLayout

        if self.maze_name == 'spiral':
            roomLayout = [[C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, B, B],  ## Spiral
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
            roomLayout = np.array(roomLayout)
            self.walls = np.argwhere(roomLayout == 'w').tolist()
            return roomLayout

        if self.maze_name == 'open_space':
            roomLayout = [[C, C, C, C, C, C, C, C, C, C, C, C, C, E, E, E, E, W, W, W],  ## Open Space
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
            roomLayout = np.array(roomLayout)
            self.walls = np.argwhere(roomLayout == 'w').tolist()
            return roomLayout

        if self.maze_name == 'high_connectivity':
            roomLayout = [[A, A, A, W, H, H, H, I, I, I, I, I, I, I, W, J, J, J, J, J],  ##High Connectivity
                          [A, A, A, W, H, H, H, W, I, I, I, I, I, I, J, J, J, J, J, J],
                          [A, A, A, H, H, H, H, W, I, I, I, I, I, I, W, J, J, J, J, J],
                          [A, A, A, W, W, W, H, W, W, W, I, W, P, W, W, J, J, J, J, J],
                          [A, A, A, W, G, G, G, G, G, W, K, W, P, P, W, J, J, J, J, J],
                          [A, A, A, W, G, G, G, G, G, K, K, W, P, P, W, W, W, J, J, J],
                          [A, A, A, G, G, G, G, G, G, W, K, W, P, P, P, P, W, P, W, W],
                          [A, A, A, W, W, W, W, G, G, W, K, W, P, P, P, P, P, P, P, P],
                          [A, A, A, A, A, A, W, F, W, W, W, W, P, P, P, P, P, P, P, P],
                          [A, A, A, A, A, A, F, F, F, F, F, W, P, P, P, P, P, P, P, P],
                          [A, A, A, A, A, A, W, F, F, F, F, P, P, P, P, P, P, P, P, P],
                          [W, W, W, B, W, W, W, F, W, W, W, W, P, W, W, W, W, P, W, W],
                          [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
                          [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, O, N, N, N],
                          [B, B, B, B, E, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
                          [B, B, B, B, E, W, E, E, E, W, W, O, W, W, O, O, W, N, N, N],
                          [W, C, W, W, E, W, W, W, E, W, L, L, L, W, O, W, W, W, N, W],
                          [C, C, C, W, W, W, D, D, D, W, L, L, L, W, W, M, M, M, M, M],
                          [C, C, C, C, D, D, D, D, D, W, L, L, L, L, M, M, M, M, M, M],
                          [C, C, C, W, W, W, D, D, D, L, L, L, L, L, W, M, M, M, M, M]]
            roomLayout = np.array(roomLayout)
            self.walls = np.argwhere(roomLayout == 'w').tolist()
            return roomLayout

        if self.maze_name == 'low_connectivity':
            roomLayout = [[A, A, A, W, H, H, H, I, I, I, I, I, I, I, W, J, J, J, J, J],  ##Low Connectivity
                          [A, A, A, W, H, H, H, W, I, I, I, I, I, I, J, J, J, J, J, J],
                          [A, A, A, W, H, H, H, W, I, I, I, I, I, I, W, J, J, J, J, J],
                          [A, A, A, W, W, W, H, W, W, W, I, W, W, W, W, J, J, J, J, J],
                          [A, A, A, W, G, G, G, G, G, W, K, W, P, P, W, J, J, J, J, J],
                          [A, A, A, W, G, G, G, G, G, K, K, W, P, P, W, W, W, J, J, J],
                          # in this line there is an open wchich doesn't exist in paper
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
            roomLayout = np.array(roomLayout)
            self.walls = np.argwhere(roomLayout == 'w').tolist()
            if self.big:
                newLayout = []
                for l in roomLayout:
                    nextLine = []
                    for x in l:
                        nextLine.extend([x, x, x])
                    newLayout.append(nextLine)
                    newLayout.append(nextLine)
                    newLayout.append(nextLine)
                roomLayout = np.array(newLayout)
                self.walls = np.argwhere(roomLayout == 'w').tolist()
            return roomLayout

        # if self.maze_name == 'big_low_connectivity':
        #     roomLayout = [[A, A, A, W, H, H, H, I, I, I, I, I, I, I, W, J, J, J, J, J],  ##Low Connectivity
        #                   [A, A, A, W, H, H, H, W, I, I, I, I, I, I, J, J, J, J, J, J],
        #                   [A, A, A, W, H, H, H, W, I, I, I, I, I, I, W, J, J, J, J, J],
        #                   [A, A, A, W, W, W, H, W, W, W, I, W, W, W, W, J, J, J, J, J],
        #                   [A, A, A, W, G, G, G, G, G, W, K, W, P, P, W, J, J, J, J, J],
        #                   [A, A, A, W, G, G, G, G, G, K, K, W, P, P, W, W, W, J, J, J],# in this line there is an open which doesn't exist in paper
        #                   [A, A, A, W, G, G, G, G, G, W, K, W, P, P, P, P, W, W, W, W],
        #                   [A, A, A, W, W, W, W, G, G, W, K, W, P, P, P, P, P, P, P, P],
        #                   [A, A, A, A, A, A, W, F, W, W, W, W, P, P, P, P, P, P, P, P],
        #                   [A, A, A, A, A, A, W, F, F, F, F, W, P, P, P, P, P, P, P, P],
        #                   [A, A, A, A, A, A, W, F, F, F, F, P, P, P, P, P, P, P, P, P],
        #                   [W, W, W, B, W, W, W, W, W, W, W, W, P, W, W, W, W, W, W, W],
        #                   [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
        #                   [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, O, N, N, N],
        #                   [B, B, B, B, E, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
        #                   [B, B, B, B, E, W, E, E, E, W, W, W, W, W, O, O, W, N, N, N],
        #                   [W, C, W, W, E, W, W, W, E, W, L, L, L, W, O, W, W, W, N, W],
        #                   [C, C, C, W, W, W, D, D, D, W, L, L, L, W, W, M, M, M, M, M],
        #                   [C, C, C, W, D, D, D, D, D, W, L, L, L, L, M, M, M, M, M, M],
        #                   [C, C, C, W, W, W, D, D, D, L, L, L, L, L, W, M, M, M, M, M]]
        #     newLayout = []
        #     for l in roomLayout:
        #         nextLine = []
        #         for x in l:
        #             nextLine.extend([x, x, x])
        #         newLayout.append(nextLine)
        #         newLayout.append(nextLine)
        #         newLayout.append(nextLine)
        #     roomLayout = np.array(newLayout)
        #     self.walls = np.argwhere(roomLayout == 'w').tolist()
        #     return roomLayout

        if self.maze_name.startswith('external'):
            # path = f"../external_mazes/{self.maze_name}.txt"
            path = f"/home/xue/projects/masterthesis/external_mazes/{self.maze_name}.txt"
            # path = f"external_mazes/{self.maze_name}.txt"
            roomLayout = []
            with open(path, "r") as f:
                content = f.readlines()
                content = [x.strip() for x in content]
                for item in content:
                    item = item.replace('#', 'w')
                    # item = item.replace('.', '0')
                    row = [x for x in item]
                    roomLayout.append(row)
            roomLayout = np.array(roomLayout)
            self.walls = np.argwhere(roomLayout == 'w').tolist()
            if self.big:
                newLayout = []
                for l in roomLayout:
                    nextLine = []
                    for x in l:
                        nextLine.extend([x, x, x])
                    newLayout.append(nextLine)
                    newLayout.append(nextLine)
                    newLayout.append(nextLine)
                roomLayout = np.array(newLayout)
                self.walls = np.argwhere(roomLayout == 'w').tolist()
            return roomLayout

    def isTerminal(self, state):
        return (state[0], state[1]) == self.goal  # self.goal is a tuple

    def reset(self):
        self.flags_found_order = []
        self.flags_collected = 0
        self.flags_collected2 = [0, 0, 0]
        if self.big:
            if self.maze_name == 'low_connectivity':
                # self.walls = np.argwhere(self.room_layout == 'w').tolist()
                self.state = (19 * 3, 0 * 3, 0, 0, 0)
                self.flags = [(0 * 3, 1 * 3), (2 * 3, 18 * 3), (5 * 3, 6 * 3)]
                self.goal = (15 * 3, 0 * 3)
            elif self.maze_name == 'basic':
                # self.walls = np.argwhere(self.room_layout == 'w').tolist()
                self.state = (6 * 3, 4 * 3, 0, 0, 0)
                self.flags = [(0 * 3, 5 * 3), (15 * 3, 0 * 3), (15 * 3, 20 * 3)]
                self.goal = (14 * 3, 1 * 3)
            elif self.maze_name == "external_maze21x21_1":
                self.state = (1 * 3, 1 * 3, 0, 0, 0)
                self.goal = (1 * 3, 19 * 3) # big version can set goal like this, this is not a mistake
                # self.goal = (19 * 3, 19 * 3)
                self.flags = [(10 * 3, 1 * 3), (1 * 3, 17 * 3), (17 * 3, 3 * 3)]
            elif self.maze_name == "external_maze21x21_2":
                self.state = (0, 0, 0, 0, 0)
                self.goal = (1 * 3, 19 * 3)
                # self.goal = (19 * 3, 19 * 3)
                self.flags = [(10 * 3, 1 * 3), (1 * 3, 17 * 3), (17 * 3, 3 * 3)]
            elif self.maze_name == "external_maze21x21_3":
                self.state = (0, 0, 0, 0, 0)
                # self.goal = (1*3, 19*3)
                self.goal = (19 * 3, 19 * 3)
                self.flags = [(10*3, 1*3), (1*3, 17*3), (17*3, 3*3)]
            elif self.maze_name == "external_maze31x31_1":
                self.state = (0*3, 0*3, 0, 0, 0)
                self.goal = (1*3, 20*3)
                # self.goal = (29*3, 29*3)
                self.flags = [(15*3, 2*3), (7*3, 27*3), (29*3, 15*3)]
            elif self.maze_name == "external_maze31x31_2":
                self.state = (0, 0, 0, 0, 0)
                # self.goal = (1, 20)
                self.goal = (5*3, 20*3)
                self.flags = [(6*3, 1*3), (15*3, 16*3), (27*3, 27*3)]
            elif self.maze_name == "external_maze61x61_1":
                self.state = (0, 0, 0, 0, 0)
                self.goal = (1*3, 57*3)
                self.flags = [(9*3, 1*3), (45*3, 58*3), (59*3, 25*3)]
            else:
                print("no matching for big version")
            return


        if self.maze_name == 'basic':
            # self.walls = np.argwhere(self.room_layout == 'w').tolist()
            self.state = (6, 4, 0, 0, 0)
            # self.prev_state = (6,4,0,0,0)
            self.flags = [(0, 5), (15, 0), (15, 20)]
            self.goal = (14, 1)
        if self.maze_name == 'strips':
            # self.walls = np.argwhere(self.room_layout == 'w').tolist()
            self.state = (0, 0, 0, 0, 0)
            self.flags = [(15, 11), (19, 0), (4, 19)]
            self.goal = (18, 1)
        if self.maze_name == 'strips2':
            # self.walls = np.argwhere(self.room_layout == 'w').tolist()
            self.state = (0, 0, 0, 0, 0)
            self.flags = [(15, 11), (19, 0), (4, 19)]
            self.goal = (18, 1)
        if self.maze_name == 'spiral':
            # self.walls = np.argwhere(self.roomLayout == 'w').tolist()
            self.state = (19, 0, 0, 0, 0)
            self.goal = (13, 13)
            self.flags = [(0, 19), (15, 6), (6, 6)]
        if self.maze_name == 'open_space':
            # self.walls = np.argwhere(self.room_layout == 'w').tolist()
            self.state = (19, 0, 0, 0, 0)
            self.flags = [(0, 0), (2, 17), (13, 14)]
            self.goal = (19, 3)
            # self.goal = (5, 3)   # 纯属试一试
        if self.maze_name == 'high_connectivity':
            # self.walls = np.argwhere(self.room_layout == 'w').tolist()
            self.state = (19, 0, 0, 0, 0)
            self.flags = [(0, 1), (2, 18), (5, 6)]
            self.goal = (15, 0)
        if self.maze_name == 'low_connectivity':
            # self.walls = np.argwhere(self.room_layout == 'w').tolist()
            self.state = (19, 0, 0, 0, 0)
            self.flags = [(0, 1), (2, 18), (5, 6)]
            self.goal = (15, 0)

        if self.maze_name == "external_maze21x21_1":
            self.state = (1, 1, 0, 0, 0)
            self.goal = (19, 19)
            self.flags = [(10,1),(1,17),(17,3)]
        if self.maze_name == "external_maze21x21_2":
            self.state = (0, 0, 0, 0, 0)
            self.goal = (19, 19)
            self.flags = [(10, 1), (1, 17), (17, 3)]
        if self.maze_name == "external_maze21x21_3":
            self.state = (0, 0, 0, 0, 0)
            self.goal = (1, 19)
            self.flags = [(10, 1), (1, 17), (17, 3)]
        if self.maze_name == "external_maze31x31_1":
            self.state = (0, 0, 0, 0, 0)
            # self.goal = (1, 20)
            self.goal = (29, 29)
            self.flags = [(15, 2), (7, 27), (29, 15)]
        if self.maze_name == "external_maze31x31_2":
            self.state = (0, 0, 0, 0, 0)
            # self.goal = (1, 20)
            self.goal = (5, 20)
            self.flags = [(6, 1), (15, 16), (27, 27)]
        if self.maze_name == "external_maze61x61_1":
            self.state = (0, 0, 0, 0, 0)
            self.goal = (1, 57)
            self.flags = [(9, 1), (45, 58), (59, 25)]
        if self.maze_name == "external_low_connectivity_1":
            self.state = (0, 0, 0, 0, 0)
            self.goal = (1, 18)
            self.flags = [(10, 16), (12, 16), (15, 15)]

    def isMovable(self, state):
        # check if wall is in the way or already out of the bounds
        if state[0] < 0 or state[0] > (len(self.room_layout) - 1):
            return False
        elif state[1] < 0 or state[1] > (len(self.room_layout[0]) - 1):
            return False
        elif [state[0], state[1]] in self.walls:
            return False
        else:
            return True

    def get_valid_nodes(self):
        valid_node_coords = []
        template = self.room_layout
        templateX = len(template[0])  # num of columns
        templateY = len(template)  # num of rows
        for i in range(templateY):
            for j in range(templateX):
                if template[i, j] != "w":
                    current = (i, j)
                    valid_node_coords.append(current)
        return valid_node_coords


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
                if index not in self.flags_found_order:
                    self.flags_found_order.append(index)
        if flag_number > -1:
            if state[flag_number + 2] == 0:
                self.flags_collected += 1
                # return 10000 * self.flags_collected
                return 10000
        if (state_prime[0], state_prime[1]) == self.goal:
            # print("hit goal")
            return self.flags_collected * 1000  # 可修改
            # # return (self.flags_collected ** 2) * 1000    #可修改
            # return 1000
        return -1  # -1 is also worth trying

# M=Maze()
# print(M.walls)
if __name__ == "__main__":
    Maze(maze='external_maze2_21x21')
    print("bingo!")