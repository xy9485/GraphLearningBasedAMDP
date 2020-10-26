import gensim.models
import numpy as np
import os
from envs.maze_env_general import Maze
from sklearn.cluster import KMeans
from sklearn import preprocessing  # to normalise existing X
import copy
from nltk.cluster import KMeansClusterer
import nltk


class GensimOperator:
    def __init__(self, fpath_paths, fpath_embedding, fpath_cluster_layout, env):
        self.fpath_paths = fpath_paths
        self.fpath_embedding = fpath_embedding
        self.fpath_cluster_layout = fpath_cluster_layout
        self.env = env
        self.wv = None

    def write_embedding(self, size, window, min_count=5, workers=4):
        paths = []
        with open(self.fpath_paths, 'r') as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            for item in content:
                path = [x for x in item.split()]
                paths.append(path)
        print("paths.shape:",np.array(paths).shape)
        # print(type(paths[0][0]))

        model = gensim.models.Word2Vec(sentences=paths, min_count=min_count, size=size, workers=workers, window=window, sg=1)
        # print(type(model.wv[('(0,7)')]))
        # print(model.wv.most_similar(positive=['(0,7)'],topn=10))
        # print(model.wv.vocab)
        self.wv = model.wv
        if not os.path.isfile(self.fpath_embedding):
            with open(self.fpath_embedding, "w") as f:
                for i, word in enumerate(model.wv.vocab):
                    # print(type(word))
                    f.write(word)
                    for item in model.wv[word]:
                        f.write(' ' + str(item))
                    f.write('\n')
            print("file: " + self.fpath_embedding + " is saved")
        else:
            print("file: " + self.fpath_embedding + " is all ready there")

        # os.chmod(path_embedding, S_IREAD)

    def read_embedding(self):
        embedding = []
        with open(self.fpath_embedding, "r") as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            for item in content:
                emd = [x for x in item.split()]
                embedding.append(emd)
        return embedding

    def get_valid_node_coords(self):
        valid_node_coords = []
        template = self.env.getRoomLayout()
        templateX = len(template[0])  # num of columns
        templateY = len(template)  # num of rows
        for i in range(templateY):
            for j in range(templateX):
                if template[i, j] != "w":
                    current = (i, j)
                    valid_node_coords.append(current)
        return valid_node_coords

    def check_unvisited_nodes(self):
        embedding = np.array(self.read_embedding())
        visited_node_coords = embedding[:, 0]
        valid_node_coords = self.get_valid_node_coords()
        print(len(valid_node_coords), len(visited_node_coords))
        for i in valid_node_coords:
            if str(i).replace(' ', '') not in visited_node_coords:
                print("not visited: ",i)

    def get_cluster_layout_kmeans(self, package='sklearn', clusters=None, init='k-means++'):
        embedding = self.read_embedding()
        pure_embedding = []
        for row in embedding:
            newrow = [float(x) for x in row[1:]]
            pure_embedding.append(newrow)

        if package == 'sklearn':
            norm_pure_embedding = preprocessing.normalize(pure_embedding)
            kmeans_labels = KMeans(n_clusters=clusters, init=init).fit_predict(np.array(norm_pure_embedding))
            # kmeans = KMeans(n_clusters=12, init=init).fit(np.array(norm_pure_embedding))
            # kmeans_labels = kmeans.labels_
            print(kmeans_labels)
            print("len(kmeans_labels): ", len(kmeans_labels))

            roomlayout_prime = copy.deepcopy(self.env.getRoomLayout()).tolist()
            print(roomlayout_prime)
            for i in range(len(kmeans_labels)):
                coord = eval(embedding[i][0])
                label = str(kmeans_labels[i])
                roomlayout_prime[coord[0]][coord[1]] = label
            self.cluster_layout = roomlayout_prime
        if package == 'nltk':
            kclusterer = KMeansClusterer(clusters, distance=nltk.cluster.util.cosine_distance, repeats=25)
            pure_embedding = [np.array(f) for f in pure_embedding]
            assigned_clusters = kclusterer.cluster(pure_embedding, assign_clusters=True)
            roomlayout_prime = copy.deepcopy(self.env.getRoomLayout()).tolist()
            print(roomlayout_prime)
            for i in range(len(pure_embedding)):
                coord = eval(embedding[i][0])
                label = str(assigned_clusters[i])
                roomlayout_prime[coord[0]][coord[1]] = label
            self.cluster_layout = roomlayout_prime


    def write_cluster_layout(self):
        if not os.path.isfile(self.fpath_cluster_layout):
            with open(self.fpath_cluster_layout, "w") as f:
                for row in self.cluster_layout:
                    for item in row:
                        f.write(item + '\t')
                    f.write('\n')
            print("file: " + self.fpath_cluster_layout + " is saved")
        else:
            print("file: " + self.fpath_cluster_layout + " is all ready there")


    def prettyprint2(self):
        listOfLists = self.cluster_layout
        for row in range(len(listOfLists)):
            print('+' + '--+' * len(listOfLists[0]))
            print('|', end='')
            for col in range(len(listOfLists[row])):
                if len(listOfLists[row][col]) == 1:
                    print(' ' + listOfLists[row][col], end='|')
                else:
                    print(listOfLists[row][col], end='|')
            print(' ')  # To change lines
        print('+' + '--+' * (len(listOfLists[0])))




if __name__ == "__main__":
    fpath_embedding = "embeddings/basic/random_paths/ep_100_maxmove_1000_length_200+/s64_w20.embedding"
    fpath_paths = "paths/random_paths_ep200_maxmove2500_len500+.txt"
    fpath_cluster_layout = "cluster_layout/random_paths_len500+_s64_w20_skipgram_kmeans12_norm_nltk.cluster"
    maze = 'open_space'

    env = Maze(maze=maze)
    gen_opt = GensimOperator(fpath_paths, fpath_embedding, fpath_cluster_layout, env)
    gen_opt.write_embedding(size=64,window=20)
    # gen_opt.check_unvisited_nodes()
    # gen_opt.get_cluster_layout_kmeans(package='nltk', clusters=12)
    # gen_opt.prettyprint2()
    # gen_opt.write_cluster_layout()
    # print(gen_opt.wv.most_similar(positive=["(19,19)"],topn=60))

# gen_opt.check_unvisited_nodes()
# cluster_layout = gen_opt.get_cluster_layout_kmeans(clusters=10)
# gen_opt.write_cluster_layout()
