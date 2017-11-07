from __future__ import division, print_function
from collections import defaultdict
from igraph import Graph
import numpy as np
import random

# Assumptions/premise:
#  * k-regular graph with fixed number of nodes
#  * n behaviors available for adoption, with fixed cost > 1/n (not every node can adopt every behavior)
#  * each node has 1 total resources available
#  * threshold for adoption is
#  * behaviors spread in epochs until steady state is reached (no new behaviors adopted)
#  * nodes do not change their adopted behaviors (once adopted, they will stick with it till the end)
#  * nodes can only adopt 1 new behavior per round, and they pick the one with the greatest number of neighbors that
#    have adopted it
#  * ties are broken randomly
#
# Nodes are assigned z_r at the beginning (fixed over time), which can be thought of as the number of posts they
# will read per epoch (day). For simplicity, we assume that each node has a fixed broadcast signal of the behaviors it
# has adopted (this could also be chosen from a Poisson distribution, but I've chosen to keep it fixed for now - for
# example, you could assume that someone's behaviors were always visible on their profile page, and the read action
# count is the number of people that someone will check up on per day).
#
# At each epoch, each node selects at random up to z_r adjacent nodes, tallies the number of nodes it sees adopting each
# behavior, and adopts the one with the highest count if it has available resources. If it has no resources left or has
# already adopted all visible behaviors, no change is made.
#
# To start, k nodes are chosen at random and initialized with a behavior. All other nodes start out with no behaviors.

random.seed(1234567)
np.random.seed(1234567)

# Degree of each node in the regular graph
GRAPH_DEGREE = 10
# Total number of behaviors available
BEHAVIOR_COUNT = 10
# Maximum number of behaviors each node can adopt (behavior_cost = 1/max_behaviors)
MAX_BEHAVIORS = 10
# Number of nodes in the graph
NODE_COUNT = 1000
# Poisson parameter controlling number of read actions a node will take per epoch
#READ_LAMBDA = 4
# Poisson parameter controlling number of write/post/broadcast actions a node will take per epoch
#POST_LAMBDA = 2

# Number of nodes to choose at random to adopt each behavior at t=0
SEEDS_PER_BEHAVIOR = 1

g = Graph.K_Regular(NODE_COUNT, GRAPH_DEGREE)

# Shortcuts
behaviors = range(BEHAVIOR_COUNT)
nodes = range(NODE_COUNT)


def init(read_lambda):
    # Initialize nodes to no behaviors adopted (2d array indexed by node and behavior)
    global adopted_behaviors
    adopted_behaviors = [[False] * BEHAVIOR_COUNT for i in range(NODE_COUNT)]

    global read_capacities
    read_capacities = [0] * NODE_COUNT
    zero_count = 0
    for node in nodes:
        read_capacities[node] = min(np.random.poisson(read_lambda), len(g.neighbors(node)))
        if read_capacities[node] == 0:
            zero_count += 1
    print("{} nodes have read capacity 0 ({:.2f}% non-zero)".format(zero_count, 100 * (1 - zero_count / len(nodes))))

    # Seed initial behaviors
    for behavior in behaviors:
        seeded = 0
        while seeded < SEEDS_PER_BEHAVIOR:
            node = random.randint(0, NODE_COUNT - 1)
            if adopted_behaviors[node][behavior]:
                continue
            adopted_behaviors[node][behavior] = True
            seeded += 1


# Behavioral capacity: total number of behaviors adopted divided by total resources available. Given that we fix the
# resource amount to 1 per node, this is total number of behaviors divided by number of nodes
def calc_utilization():
    total_adopted_behaviors = 0
    for i in nodes:
        for j in behaviors:
            total_adopted_behaviors += adopted_behaviors[i][j]

    return total_adopted_behaviors / NODE_COUNT


def run_epoch():
    changed = False
    for node in nodes:
        if sum(adopted_behaviors[node]) >= MAX_BEHAVIORS or read_capacities[node] == 0:
            continue

        # Get a random sampling of nodes we'll check on
        read_set = np.random.choice(g.neighbors(node), size=read_capacities[node], replace=False)

        # Tally up the behaviors we've observed
        observed_behaviors = defaultdict(int)
        for neighbor in read_set:
            for behavior in behaviors:
                observed_behaviors[behavior] += adopted_behaviors[neighbor][behavior]

        # Exclude behaviors we've already adopted
        for behavior in behaviors:
            if adopted_behaviors[node][behavior]:
                del observed_behaviors[behavior]

        # Sort remaining observed behaviors
        sorted_behaviors = sorted(observed_behaviors, key=observed_behaviors.get, reverse=True)
        max_count = observed_behaviors[sorted_behaviors[0]]
        if max_count == 0:
            continue  # No non-adopted behaviors observed

        # Handle tie-breaking
        ties = []
        for behavior in sorted_behaviors:
            if observed_behaviors[behavior] == max_count:
                ties.append(behavior)
        selected_behavior = np.random.choice(ties)
        adopted_behaviors[node][selected_behavior] = True
        changed = True
    return changed


for read_lambda in range(1, GRAPH_DEGREE + 1):
    print("-- lambda_read={} --".format(read_lambda))
    init(read_lambda)

    for i in range(1000):
        print(" {:3d}: {:.6f}".format(i, calc_utilization()))
        have_delta = run_epoch()
        if not have_delta:
            break

    print("")