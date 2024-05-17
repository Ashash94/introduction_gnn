import networkx as nx
import matplotlib.pyplot as plt

H = nx.DiGraph()

H.add_nodes_from ([(0, {'color': 'blue', 'size': 250 }),
                   (1, {'color': 'yellow', 'size': 400 }),
                   (2, {'color': 'orange', 'size': 150 }),
                   (3, {'color': 'red', 'size': 600 })])

H.add_edges_from([(0,1),
                  (1,2),
                  (1,0),
                  (1,3),
                  (2,3),
                  (3,0)])

colors = list(nx.get_node_attributes(H,'color').values())
sizes = list(nx.get_node_attributes(H,'size').values())

# nx.draw(H, with_labels = True, node_color = colors, node_size = sizes)
# plt.savefig("graphH.png")

G = H.to_undirected()
nx.draw(G, with_labels = True, node_color = colors, node_size = sizes)
plt.savefig("graphG.png")