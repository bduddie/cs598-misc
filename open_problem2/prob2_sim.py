from collections import defaultdict
from igraph import Graph
from scipy.stats import norm
import numpy as np
import random

random.seed(12345678)
np.random.seed(12345678)

# Nodes can be described by their attributes, which can intuitively be thought of as the degree to which each category
# represents aspects of a person's identity, such as their interests, job, etc. Attributes are weighted by randomly
# assigning a total of 100 attribute points across available categories, by iterating over each attribute ID in random
# order, and randomly selecting an integer within the range of [0, points_left], then this number of points is assigned
# to the attribute. If points remain after iterating over all attribute IDs, they are all assigned to an attribute ID at
# random. This produces a distribution whereby nodes are likely to strongly identify with one attribute, then have
# decreasing interest in remaining attributes.
#
# Nodes also have higher affinities for other "similar" nodes, measured via cosine similarity.
#
# Nodes have a target friendship capacity, which is selected from a Poisson distribution. This friendship capacity
# drives the likelihood that edges will be formed or broken. The probability that a node will attempt a new connection
# is given by the inverse of the CDF of the normal distribution centered on their friendship capacity with standard
# deviation 1, while the probability that an edge may be broken is the opposite.
#
# If a node "decides" (via random chance falling within the normal CDF-based threshold) to make a new edge, it selects
# another node that is not in its neighborhood at random, then uses the similarity score as a threshold for another
# random decision. If the random number is below the threshold, a new bond is formed, such that highly similar nodes
# have high chance of becoming friends. The same principle works for breaking bonds in the inverse, with the node
# randomly selecting from its neighborhood and having a higher chance of breaking a bond with the node if it is less
# similar.
NUM_ATTRS = 3
MEAN_EDGE_CAPACITY = 5
MAX_NODES = 50
MAX_EPOCHS = 2000


attrs = [i for i in range(NUM_ATTRS)]

class NodeDetails:
    # Maps (id1, id2) --> similarity, where id1 < id2
    similarity_cache = {}

    def __init__(self, id):
        self.id = id
        self.attrs = NodeDetails.gen_attrs()
        self.friend_capacity = np.random.poisson(MEAN_EDGE_CAPACITY)
        print("Created node {} with attrs {} and capacity {}".format(self.id, self.attrs, self.friend_capacity))

    @staticmethod
    def gen_attrs():
        node_attrs = [0] * NUM_ATTRS
        points_left = 100

        # Iterate over each attribute index in random order
        for attr in random.sample(attrs, len(attrs)):
            node_attrs[attr] = random.randint(0, points_left)
            points_left = points_left - node_attrs[attr]

        # If points are left over (likely), add them all to the attribute with the most points, breaking ties randomly
        if points_left > 0:
            max_points = max(node_attrs)
            for attr in random.sample(attrs, len(attrs)):
                if node_attrs[attr] == max_points:
                    node_attrs[attr] += points_left
                    break

        return node_attrs

    def check_actions(self, num_friends):
        """
        Checks whether this node should attempt to make and/or break a bond
        :param num_friends: Current number of friends (i.e. the node's degree)
        :return: (try_make_bond, try_break_bond) as a pair of bools
        """
        threshold = norm.cdf(num_friends, loc=self.friend_capacity, scale=1)
        try_make_bond = random.random() >= threshold
        try_break_bond = random.random() < threshold
        return try_make_bond, try_break_bond

    def similarity(self, other):
        id1, id2 = min(self.id, other.id), max(self.id, other.id)
        if (id1, id2) not in NodeDetails.similarity_cache:
            # Cosine similarity: a dot b / (magnitude(a) * magnitude(b))
            sim = np.dot(self.attrs, other.attrs) / (np.linalg.norm(self.attrs) * np.linalg.norm(other.attrs))
            NodeDetails.similarity_cache[(id1, id2)] = sim
            return sim

        return NodeDetails.similarity_cache[(id1, id2)]


class FriendshipGraph:
    def __init__(self):
        self.g = Graph()

        # Seed the graph with an initial set of nodes
        self.node_info = [NodeDetails(i) for i in range(NUM_ATTRS)]
        self.g.add_vertices(NUM_ATTRS)

    def add_node(self):
        self.node_info.append(NodeDetails(self.g.vcount()))
        self.g.add_vertex()

    def run_epoch(self):
        # Add a new node (one at a time)
        if self.g.vcount() < MAX_NODES:
            self.add_node()

        # Add/remove edges
        for node_id in range(self.g.vcount()):
            num_friends = self.g.degree(node_id)
            node = self.node_info[node_id]
            try_add, try_remove = node.check_actions(num_friends)

            if try_add and num_friends < self.g.vcount() - 1:
                while True:
                    candidate_id = random.randint(0, self.g.vcount() - 1)
                    if candidate_id != node_id and not self.g.are_connected(candidate_id, node_id):
                        break
                if random.random() <= node.similarity(self.node_info[candidate_id]):
                    # print("adding tie b/w {} and {}".format(node_id, candidate_id))
                    self.g.add_edge(node_id, candidate_id)

            if try_remove and num_friends > 0:
                candidate_id = random.choice(self.g.neighborhood(node_id))
                if random.random() > node.similarity(self.node_info[candidate_id]):
                    # print("breaking tie b/w {} and {}".format(node_id, candidate_id))
                    self.g.delete_edges(self.g.get_eid(node_id, candidate_id))


if __name__ == '__main__':
    fg = FriendshipGraph()
    for _ in range(MAX_EPOCHS):
        fg.run_epoch()
        #print(fg.g)

    print(fg.g)

    sim_total = 0
    sim_count = 0
    for node in fg.node_info:
        print("{}: {}".format(node.id, node.attrs))
        for friend_id in fg.g.neighborhood(node.id):
            if friend_id == node.id:
                continue
            friend = fg.node_info[friend_id]
            print("  {} sim {:.6f}".format(friend_id, node.similarity(friend)))
            sim_total += node.similarity(friend)
            sim_count += 1

    print("avg similarity among friends {:.6f}".format(sim_total / sim_count))

    sim_total = sim_count = 0
    for node in fg.node_info:
        for other_id in range(node.id, len(fg.node_info)):
            sim_total += node.similarity(fg.node_info[other_id])
            sim_count += 1
    print("avg similarity among all nodes {:.6f}".format(sim_total / sim_count))

    # degree distribution
    degrees = defaultdict(int)
    for node_id in range(MAX_NODES):
        degrees[fg.g.degree(node_id)] += 1
    print(degrees)

    pass