import gensim.models
import numpy as np
import os
from envs.maze_env_general import Maze
from sklearn.cluster import KMeans
from sklearn import preprocessing  # to normalise existing X
from sklearn.metrics import pairwise_distances_argmin_min
import copy
from nltk.cluster import KMeansClusterer
import nltk
import time
from envs.maze_env_general_all_approaches import Maze


class GensimOperator_Topology:
    def __init__(self, env:Maze):
        # self.sentences = sentences # 2-D list, strings inside
        self.env = env
        self.wv = None
        self.cluster_layout = None
        self.cluster_labels = None
        self.num_of_sentences_last_time = 0
        print("GensimOperator initialized!")

    def get_cluster_layout(self, sentences, size, window, clusters, skip_gram=1, min_count=5, workers=32, negative=5,
                                     package='sklearn'):
        print("start gensim Word2Vec model training...")
        # self.sentences = sentences
        if len(sentences) == self.num_of_sentences_last_time:
            model = self.model
        else:
            start = time.time()
            model = gensim.models.Word2Vec(sentences=sentences, min_count=min_count, size=size, workers=workers,
                                           window=window, sg=skip_gram, negative=negative)
            end = time.time()
            print(f"internal w2v training time: {end-start}")
        self.wv = model.wv
        embeddings = []
        words = []
        for i, word in enumerate(self.wv.vocab):
            words.append(word)
            embeddings.append(self.wv[word])
        # print(words[0],type(words[0]))
        print("start check unvisited nodes...")
        self.check_unvisited_nodes(words)

        print("start clustering...")
        start = time.time()
        if package == 'sklearn':
            norm_embeddings = preprocessing.normalize(embeddings)
            # estimator = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=300, tol=0.00001, ).fit(
            #     np.array(norm_embeddings))
            # kmeans_labels = estimator.labels_
            # kmeans_centers = estimator.cluster_centers_
            # print("kmeans_centers:",kmeans_centers)
            kmeans_labels = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=300, tol=0.00001,).fit_predict(np.array(norm_embeddings))
            self.cluster_labels = kmeans_labels
            # closests, _ = pairwise_distances_argmin_min(kmeans_centers, np.array(norm_embeddings))
            # print("len(kmeans_centers),len(closests):", len(kmeans_centers), len(closests))
            # self.closest_coords_centers = [[eval(words[i]),str(kmeans_labels[i])] for i in closests]
            # print(kmeans_labels)
            # print("len(kmeans_labels): ", len(kmeans_labels))
            roomlayout_prime = copy.deepcopy(self.env.room_layout).tolist()
            # print(roomlayout_prime)
            for i in range(len(kmeans_labels)):
                coord = eval(words[i])
                label = str(kmeans_labels[i])
                roomlayout_prime[coord[0]][coord[1]] = label
            self.cluster_layout = roomlayout_prime
        if package == 'nltk':
            # embeddings = preprocessing.normalize(embeddings)
            kclusterer = KMeansClusterer(clusters, distance=nltk.cluster.util.cosine_distance, repeats=10,
                                         normalise=True, avoid_empty_clusters=True)
            embeddings = [np.array(f) for f in embeddings]
            assigned_clusters = kclusterer.cluster(embeddings, assign_clusters=True)
            self.cluster_labels = assigned_clusters
            roomlayout_prime = copy.deepcopy(self.env.room_layout).tolist()
            # print(roomlayout_prime)
            for i in range(len(embeddings)):
                coord = eval(words[i])
                label = str(assigned_clusters[i])
                roomlayout_prime[coord[0]][coord[1]] = label
            self.cluster_layout = roomlayout_prime
        # print(self.cluster_layout)
        end = time.time()
        print(f"internal k-means time: {end-start}")
        self.model = model
        self.num_of_sentences_last_time = len(sentences)

    def check_unvisited_nodes(self, words):
        valid_node_coords = []
        template = self.env.room_layout
        templateX = len(template[0])  # num of columns
        templateY = len(template)  # num of rows
        for i in range(templateY):
            for j in range(templateX):
                if template[i, j] != "w":
                    current = (i, j)
                    valid_node_coords.append(current)
        visited_node_coords = words
        print("len(valid_node_coords), len(visited_node_coords):", len(valid_node_coords), len(visited_node_coords))
        # for i in valid_node_coords:
        #     if str(i).replace(' ', '') not in visited_node_coords:
        #         print("not visited: ", i)
        for i in valid_node_coords:
            if str(i) not in visited_node_coords:
                print("not visited: ", i)

    def write_cluster_layout(self,fpath_cluster_layout, check=0):
        if check == 1:
            if not os.path.isfile(fpath_cluster_layout):
                with open(fpath_cluster_layout, "w") as f:
                    for row in self.cluster_layout:
                        for item in row:
                            f.write(item + '\t')
                        f.write('\n')
                print("file: " + fpath_cluster_layout + " is saved")
            else:
                print("file: " + fpath_cluster_layout + " is all ready there")
        else:
            with open(fpath_cluster_layout, "w") as f:
                for row in self.cluster_layout:
                    for item in row:
                        f.write(item + '\t')
                    f.write('\n')
            print("file: " + fpath_cluster_layout + " is saved")

class GensimOperator_General:
    def __init__(self, env):
        # self.sentences = sentences   # 2-D list, strings inside
        self.env = env
        self.wv = None
        self.cluster_labels = None
        self.num_of_sentences_last_time = 0
        print("GensimOperator_General initialized!!")

    def get_cluster_labels(self, sentences, size, window, clusters, skip_gram=1, min_count=5, workers=32, negative=5,
                            package='sklearn'):
        self.sentences = sentences
        self.num_clusters = clusters
        print("start gensim Word2Vec model training...")
        if len(sentences) == self.num_of_sentences_last_time:
            model = self.model
        else:
            start = time.time()
            model = gensim.models.Word2Vec(sentences=sentences, min_count=min_count, size=size, workers=workers,
                                           window=window, sg=skip_gram, negative=negative)
            end = time.time()
            print(f"internal w2v training time: {end-start}")
        self.wv = model.wv
        embeddings = []
        words = []
        for i, word in enumerate(self.wv.vocab):
            words.append(word)
            embeddings.append(self.wv[word])
        self.words = words
        print("gensim_opt.words", self.words[:10])
        self.embeddings = embeddings
        print("GensimOperator_General self.words:", self.words[:20])
        # print(words[0],type(words[0]))
        print("start check unvisited nodes...")
        self.check_unvisited_nodes(self.words)

        print("start clustering...")
        start = time.time()
        if package == 'sklearn':
            norm_embeddings = preprocessing.normalize(self.embeddings)
            kmeans_labels = KMeans(n_clusters=clusters, init='k-means++').fit_predict(np.array(norm_embeddings))
            self.cluster_labels = kmeans_labels
            print("gensim_opt.cluster_labels:", self.cluster_labels[:10])

        if package == 'nltk':
            embeddings = preprocessing.normalize(self.embeddings)
            kclusterer = KMeansClusterer(clusters, distance=nltk.cluster.util.cosine_distance, repeats=25)
            embeddings = [np.array(f) for f in embeddings]
            assigned_clusters = kclusterer.cluster(embeddings, assign_clusters=True)
            self.cluster_labels = assigned_clusters

        end = time.time()
        print(f"internal k-means time: {end-start}")
        self.model = model
        self.num_of_sentences_last_time = len(sentences)

    def check_unvisited_nodes(self, words):
        valid_states = []
        template = self.env.room_layout
        templateX = len(template[0])  # num of columns
        templateY = len(template)  # num of rows
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    for i in range(templateY):
                        for j in range(templateX):
                            if template[i, j] != "w":
                                current = (i, j, k, l, m)
                                valid_states.append(current)
        visited_states = words
        print("len(valid_node_coords), len(visited_node_coords):", len(valid_states), len(visited_states))
        # for i in valid_node_coords:
        #     if str(i).replace(' ', '') not in visited_node_coords:
        #         print("not visited: ", i)
        for i in valid_states:
            if str(i) not in visited_states:
                print("not visited: ", str(i))



# =====================================


# if __name__ == "__main__":
#     def get_paths_from_file(fpath_paths):
#         paths = []
#         with open(fpath_paths, 'r') as f:
#             content = f.readlines()
#             content = [x.strip() for x in content]
#             for item in content:
#                 path = [x for x in item.split()]
#                 paths.append(path)
#         return paths
#
#
#     fpath_paths = "paths/basic/random_paths/ep_100_maxmove_1000_length_200+.txt"
#     fpath_cluster_layout = "cluster_layout/basic/random_paths/random_paths_len200+_s64_w20_skipgram_kmeans10_nltk.cluster"
#     maze = 'basic'
#
#     env = Maze(maze=maze)
#     paths = get_paths_from_file(fpath_paths)
#     gen_opt = GensimOperator(paths, fpath_cluster_layout, env)
#     gen_opt.get_clusterlayout_from_paths(size=64, window=20, clusters=10, package='nltk')
#     # gen_opt.prettyprint2()
#     gen_opt.write_cluster_layout()
#     # print(gen_opt.wv.most_similar(positive=["(19,19)"],topn=60))

# gen_opt.check_unvisited_nodes()
# cluster_layout = gen_opt.get_cluster_layout_kmeans(clusters=10)
# gen_opt.write_cluster_layout()