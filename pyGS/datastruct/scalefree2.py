import numpy as np
import networkx as nx
from tqdm import tqdm

def scalefree2(N, S, alpha=1.2, pasign=0.62, max_attempts=1000):
    """
    Create a scalefree network with N nodes and specific sparsity.
    Args:
    N : int - Number of nodes in the network
    S : float - Sparsity of the network
    alpha : float - Power value in the power law distribution, default 1.2
    pasign : float - Probability of activation sign, default 0.62

    Returns:
    A : NumPy array - Adjacency matrix of the network
    """
    G = nx.Graph()
    G.add_nodes_from(range(N))

    # Initial random links
    while len(G.edges) < 2:
        a, b = np.random.choice(N, 2, replace=False)
        G.add_edge(a, b, weight=(np.random.rand() < pasign) * 2 - 1)

    # List of all nodes
    node_list = list(G.nodes)

    target_edges = int(S * N * (N - 1) / 2)
    attempts = 0

    # Main loop to add new links, with progress bar and attempt limit
    with tqdm(total=target_edges, desc="Building network") as pbar:
        while len(G.edges) < target_edges and attempts < max_attempts:
            degrees = np.array([G.degree(n) for n in node_list])
            prob = (degrees ** alpha) / np.sum(degrees ** alpha)

            # Normalize probability distribution to avoid potential floating-point inaccuracies
            prob /= prob.sum()

            # Choose a node based on the calculated probabilities
            new_link = np.random.choice(node_list, p=prob)

            # Choose another node, ensuring no self-loop
            possible_nodes = [n for n in node_list if n != new_link]
            another_node = np.random.choice(possible_nodes)

            # Add the new link if it's unique
            if not G.has_edge(new_link, another_node):
                G.add_edge(new_link, another_node, weight=(np.random.rand() < pasign) * 2 - 1)
                pbar.update(1)
            attempts += 1

    # Convert graph to adjacency matrix
    A = nx.to_numpy_array(G, nodelist=range(N), weight='weight')
    return A

# Example usage:
N = 100  # Number of nodes
S = 0.1  # Sparsity of the network
A = scalefree2(N, S)
print(A)
