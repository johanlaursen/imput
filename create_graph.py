import peartree as pt # used to construct network
import networkx as nx
import pickle


def main():
    path = "data/gtfs_rejsekort.zip"

    feed = pt.get_representative_feed(path)
    # start and end are used to only select part of the data
    start = 0*60*60  # 0:00
    end = 36*60*60  # 36:00 because data is retarded

    G = pt.load_feed_as_graph(feed, start, end)

    # Relabels nodes by removing 04LSS_ from each node
    relabel_dic = {node:node[6:] for node in G.nodes()}
    G = nx.relabel_nodes(G, relabel_dic, copy=False)

    with open("rejsekort_graph_cleaned.gpickle", "wb") as f:
        pickle.dump(G, f)

if __name__ == "__main__":
    main()