## The repository for paper: [Graph learning-based generation of abstractions for reinforcement learning]

The evaluations involve two types of tasks: flag-collection and navigation.

In flag-collection domain, the agent is tasked to traverse the environment to collect 3 flags and then bringen back to the goal state.

In navigation domain, the agent just need to find the optimal path from the start state to the goal state.

### Instructions for tasks of flag-collection

Use graph-based AMDP, with stochasticity config: by the interval of 1 step, 50% of doors are closed with probability of 25%. To remove stochasticity, set the first argument for --stochasticity to 0.0.

```
python entrance.py --approach topology --maze basic --big 1 --e_eps 1000 -mm 100 --q_eps 500 --repetitions 10 --rep_size 128 --win_size 50 --numbers_of_clusters 9 16 --stochasticity 0.5 0.25 1 --print_to_file
```

Use uniform AMDP

```
python entrance.py --approach uniform --maze basic --big 1 --e_eps 1000 -mm 100 --q_eps 500 --repetitions 10 --numbers_of_clusters 9 16  --print_to_file
```

### Instructions for tasks of navigation

Use graph-based AMDP

```
python entrance.py --approach topology --maze basic --big 1 --e_eps 1000 -mm 100 --q_eps 500 --repetitions 10 --rep_size 128 --win_size 50 --numbers_of_clusters 9 16 --print_to_file
```

Use uniform AMDP

```
python entrance.py --approach topology --maze basic --big 1 --e_eps 1000 -mm 100 --q_eps 500 --repetitions 10 --numbers_of_clusters 9 16 --print_to_file
```