import pickle
import networkx as nx
import argparse

def get_apsp(tree_path, output_path, cutoff=3):
	with open(tree_path, 'rb') as f:
		tree = pickle.load(f)

	utree = tree.to_undirected()
	apsp = dict(nx.all_pairs_shortest_path_length(utree, cutoff=cutoff))
	pickle.dump(asps, open(output_path, 'wb'))



def run(args):
	get_apsp(args.tree_path, args.output_path, args.cutoff)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tree_path",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--cutoff",
        default=3,
        type=int,
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
