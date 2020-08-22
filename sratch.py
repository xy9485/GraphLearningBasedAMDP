import numpy as np
import pandas as pd
import random
from sklearn import preprocessing  # to normalise existing X
import os
# to_be_append = pd.Series(
#                     [0,2] * 4,
#                 )
# df = pd.DataFrame({'A': [0, 1, 2, 3, 4],
#                    'B': [5, 6, 7, 8, 9],
#                    'C': ['a', 'b', 'c', 'd', 'e']})
#
data = np.array([
     [1,2,3],
     [1,2,3],
     [1,2,3],
     [4,5,6],
     ])

# data.fill(0)
l1 = [['123','344','123'],['23','4535345','ssada']]
q_table2 = [(1,2,3)]
q_table2.append((2,3,6))
str = "(19,1)"
config = "random_paths_len500+_s64_w20_kmeans8"

# with open("paths/random_paths_ep200_maxmove2500_len500+.txt", "r") as f:
#      content = f.readlines()
#      content = [x.strip() for x in content]
#      print("len(content): ",len(content))
#      num=0
#      for item in content:
#           path = [x for x in item.split()]
#           if path[-1] != "(19,3)":
#                print(len(path))
#                num += 1
#      print("num: ",num)

# X = [[ 1., -1.,  2.],
#      [ 2.,  0.,  0.],
#      [ 0.,  1., -1.]]
# X_normalized = preprocessing.normalize(X, norm='l2')
# print(X_normalized)
#
#
# a=float("(19,2)")
# print(a)

config = {
    'maze': 'high_connectivity',
    'mode': 'random_paths',
    'ep': 100,
    'max_move_count': 1000,
    'min_length_to_save': 200,
    'representation_size': 64,
    'window': 20,
    'kmeans_clusters': 12,
    'package': 'nltk'
}
# config = "random_paths_ep100_maxmove1000_len200+_s64_w20_kmeans12"
fpath_paths = f"paths/{config['maze']}/{config['mode']}/ep_{config['ep']}_maxmove_{config['max_move_count']}_length_" \
              f"{config['min_length_to_save']}+.txt"

# makedirs(fpath_paths)

fpath_embedding = f"embeddings/{config['maze']}/{config['mode']}/ep_{config['ep']}_maxmove_{config['max_move_count']}_length_" \
                  f"{config['min_length_to_save']}+/s{config['representation_size']}_w{config['window']}.embedding"

fpath_cluster_layout = f"cluster_layout/{config['maze']}/{config['mode']}/ep_{config['ep']}_maxmove_{config['max_move_count']}_length_" \
                       f"{config['min_length_to_save']}+/s{config['representation_size']}_w{config['window']}" \
                       f"_kmeans{config['kmeans_clusters']}_package_{config['package']}.cluster"

if not os.path.isfile(fpath_embedding):
    with open(fpath_cluster_layout, "w") as f:
        f.write("jjj")

print(fpath_paths)
print(fpath_embedding)
print(fpath_cluster_layout)