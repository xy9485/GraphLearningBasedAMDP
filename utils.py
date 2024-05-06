import argparse
import numpy as np

def parse_args():
    cli = argparse.ArgumentParser()
    cli.add_argument("--approach_name", type=str, choices=['uniform, topology'], help="name of the approach")
    cli.add_argument("--maze", type=str, default='basic', choices=['low_connectivity2', 'external_maze21x21_1', 'external_maze31x31_2', 'strips2', 'spiral', 'basic', 'open_space', 'high_connectivity'], help='names of maze to choose')
    cli.add_argument("--big", type=int, default=0, help="big or small maze")
    cli.add_argument("--e_mode", type=str, default='sarsa', choices=['sarsa', 'softmax'], help="pure exploration policy: sarsa or softmax")
    cli.add_argument("--e_start", type=str, default='random', choices=['random', 'last', 'mix'], help="during pure exploration, how to choose start state for each episode: random or last")
    cli.add_argument("--e_eps", type=int, default=1000, help="number of episodes for pure exploration")
    cli.add_argument("--mm", type=int, default=100, help="max step for each episode")
    cli.add_argument("--ds_factor", type=float, default=0.5, help="down sampling factor for sentences")
    cli.add_argument("--ds_repetitions", type=int, default=2, help="number of repetitions for down sampling")

    cli.add_argument("--q_eps", type=int, default=500, help="number of episodes for ground learning")
    cli.add_argument("--repetitions", type=int, default=2, help="number of repetitions")
    cli.add_argument("--rep_size", type=int, default=128, help="number of dimentions of state representation")
    cli.add_argument("--win_size", type=int, default=50, help="window size for word2vec")
    cli.add_argument("--sg", type=int, default=1, help="skip-gram, otherwise CBOW")
    cli.add_argument("--numbers_of_clusters", type=int, nargs='+', help="number of abstract states for Uniform will be matched with the number of clusters")
    cli.add_argument("--k_means_pkg", type=str, default='sklearn', choices=['sklearn', 'nltk'],help="k-means package: sklearn or nltk")
    cli.add_argument("--interpreter", type=str, default='R', help="L or R")
    cli.add_argument("--std_factor", type=float, default=1/np.sqrt(10), help="std factor for plotting")

    cli.add_argument("--stochasticity", type=float, nargs= 3, default=[0.0, 0.25, 1], help="stochasticity of the environment")

    cli.add_argument("--print_to_file", action='store_true', help="print output to file")
    cli.add_argument("--show", action='store_true', help="show plots")
    cli.add_argument("--save", action='store_true', help="save plots")

    args = cli.parse_args()
    return args