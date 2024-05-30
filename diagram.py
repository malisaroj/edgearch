import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes with positions
nodes = {
    "GRU_t_minus_1": (1, 3),
    "GRU_t": (3, 3),
    "GRU_t_plus_1": (5, 3),
    "Attention_t_minus_1": (1, 2),
    "Attention_t": (3, 2),
    "Attention_t_plus_1": (5, 2),
    "BiLSTM_t_minus_1": (1, 1),
    "BiLSTM_t": (3, 1),
    "BiLSTM_t_plus_1": (5, 1),
    "Input_t_minus_1": (1, 0),
    "Input_t": (3, 0),
    "Input_t_plus_1": (5, 0),
}

# Add edges
edges = [
    ("GRU_t_minus_1", "GRU_t"),
    ("GRU_t", "GRU_t_plus_1"),
    ("BiLSTM_t_minus_1", "Attention_t_minus_1"),
    ("BiLSTM_t", "Attention_t"),
    ("BiLSTM_t_plus_1", "Attention_t_plus_1"),
    ("Attention_t_minus_1", "GRU_t_minus_1"),
    ("Attention_t", "GRU_t"),
    ("Attention_t_plus_1", "GRU_t_plus_1"),
    ("Input_t_minus_1", "BiLSTM_t_minus_1"),
    ("Input_t", "BiLSTM_t"),
    ("Input_t_plus_1", "BiLSTM_t_plus_1"),
]

# Add nodes and edges to the graph
G.add_nodes_from(nodes.keys())
G.add_edges_from(edges)

# Define node positions
pos = nodes

# Draw nodes and edges
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=3000, font_size=10, font_weight="bold", arrows=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels={edge: '' for edge in edges}, font_size=8)

# Show the plot
plt.show()
