import numpy
import pandas as pd
import random
from sklearn import preprocessing  # to normalise existing X
import os

list1 = [[0, 1, 5], [3, 5], [4, 0, 1, 5]]
print(sorted(list1, key=lambda l: len(l)))

folder_cluster_layout = f"cluster_layout/{config['maze']}/{config['mode']}/rp{config['rp']}_ep{config['ep']}_paths" \
                        f"{num_randomwalk_episodes}_{num_saved_from_p1}_{num_saved_from_p2}"