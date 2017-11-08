# Assumptions/premise:
#  * k-regular graph with fixed number of nodes
#  * n behaviors available for adoption, with fixed cost > 1/n (not every node can adopt every behavior)
#  * each node has 1 total resources available
#  * behaviors spread in epochs until steady state is reached (no new behaviors adopted)
#  * nodes do not change their adopted behaviors (once adopted, they will stick with it till the end)
#  * for simplicity, the threshold for adoption is effectively 0 - nodes can only adopt 1 new behavior per round, and
#    they pick the one with the greatest number of neighbors that have adopted it
#  * ties are broken randomly
#
# Nodes are assigned z_r at the beginning (fixed over time), which can be thought of as the number of posts they
# will read per epoch (day). Nodes are also assigned z_p at the beginning, which can be seen as the number of posts they
# will broadcast to their immediate neighbors. Each post is assumed to exhibit all behaviors that the node has adopted.
# z_r is chosen from a Poisson distribution with a variable parameter (read_lambda), and z_p can be fixed at 1 across
# the network, or selected from a Poisson distribution with a different parameter (write_lambda).
#
# At each epoch, each node selects at random up to z_r posts from all that are visible to it (posts made by its
# neighbors according to their z_p values), and tallies the number of nodes it sees adopting each behavior type, then
# adopts the one behavior with the highest count if it has available resources. If it has no resources left or has
# already adopted all visible behaviors, no change is made.
#
# To start, k nodes are chosen at random and initialized with a behavior. All other nodes start out with no adopted
# behaviors.
#
# Results:
#  1. Behavior propagation is constrained by the number of nodes that do not post or read at all. The resulting value of
#     B at equilibrium is approximately (1 - e^-k_post)*(1 - e^-k_read)
#  2. Higher lambda values increase the rate of propagation throughout the network (TBD: see if we can find any
#     relationship, or make some graphs showing how B progresses per epoch and stuff)
#  3. Above some threshold value of lambda (where everyone is posts at least once and everyone reads at least once),
#     there is no change to the outcome

from __future__ import division, print_function
from collections import defaultdict
from igraph import Graph
import numpy as np
import random


random.seed(1234567)
np.random.seed(1234567)

# Degree of each node in the regular graph
GRAPH_DEGREE = 10
# Total number of behaviors available
BEHAVIOR_COUNT = 10
# Maximum number of behaviors each node can adopt (behavior_cost = 1/max_behaviors)
MAX_BEHAVIORS = 10
# Number of nodes in the graph
NODE_COUNT = 500

# Number of nodes to choose at random to adopt each behavior at t=0
SEEDS_PER_BEHAVIOR = 1

g = Graph.K_Regular(NODE_COUNT, GRAPH_DEGREE)

# Shortcuts
behaviors = range(BEHAVIOR_COUNT)
nodes = range(NODE_COUNT)


def init(read_lambda, post_lambda=0):
    """
    Initializes global structures for running the simulation
    :param read_lambda: Poisson parameter used when determining how many posts a node will read per epoch
    :param post_lambda: Poisson parameter used to determine how many posts a node will make per epoch, or 0 to uniformly
    set this to 1 across all nodes
    """
    # Initialize nodes to no behaviors adopted (2d array indexed by node and behavior)
    global adopted_behaviors
    adopted_behaviors = [[False] * BEHAVIOR_COUNT for _ in range(NODE_COUNT)]

    global post_capacities
    if post_lambda:
        post_capacities = [0] * NODE_COUNT
        zero_count = 0
        for node in nodes:
            post_capacities[node] = np.random.poisson(post_lambda)
            if post_capacities[node] == 0:
                zero_count += 1
        print("{} nodes have post capacity 0 ({:.2f}% non-zero)".format(zero_count, 100 * (1 - zero_count / len(nodes))))
    else:
        post_capacities = [1] * NODE_COUNT

    # Initialize post lists - these are the set of other nodes that a given node could possibly observe
    global post_lists
    post_lists = [[] for _ in nodes]
    for node in nodes:
        for neighbor in g.neighbors(node):
            post_lists[node].extend([neighbor] * post_capacities[neighbor])

    # Initialize read capacities, clamping to the maximum number of posts visible to the node
    global read_capacities
    read_capacities = [0] * NODE_COUNT
    zero_count = 0
    for node in nodes:
        read_capacities[node] = min(np.random.poisson(read_lambda), len(post_lists[node]))
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

        # Get a random sampling of posts we'll read
        read_set = np.random.choice(post_lists[node], size=read_capacities[node], replace=False)

        # Tally up the behaviors we've observed (we see all behaviors adopted by the neighbor making the post)
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


for post_lambda in range(GRAPH_DEGREE + 1):
    for read_lambda in range(1, GRAPH_DEGREE + 1):
        print("-- lambda_read={} lambda_post={} --".format(read_lambda, post_lambda))
        init(read_lambda, post_lambda)

        for i in range(1000):
            print(" {:3d}: {:.2f}".format(i, calc_utilization()))
            have_delta = run_epoch()
            if not have_delta:
                break

        print("")