# A demo of using the tree
# Each child is_a parent
import pickle
import networkx as nx

if __name__ == '__main__':
    with open('tree.pkl', 'rb') as f:
        tree = pickle.load(f)
    print(nx.shortest_path(tree, 'C1285228', 'C0409878'))   # shortest path between 2 cui
    print(nx.shortest_path(tree, 'C0409878', 'C1285228'))   # an error will be raised if there's no path