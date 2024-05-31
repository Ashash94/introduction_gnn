import networkx as nx
import matplotlib.pyplot as plt

H = nx.DiGraph()  # DiGraph = Directed Graph, crée un graph vide


# À ce graph, on ajoute 4 noeuds qui prennent trois critères : un attribut, une couleur et un poids.
H.add_nodes_from(
    [
        (0, {"color": "blue", "size": 250}),
        (1, {"color": "yellow", "size": 400}),
        (2, {"color": "orange", "size": 150}),
        (3, {"color": "red", "size": 600}),
    ]
)

# On ajoute des arrêtes qui relieront les noeuds, dans le cas du graph directionnel l'ordre des attributs à son importance. ex : [(*noeud de départ*, *noeud d'arrivée*), ...]
H.add_edges_from([(0, 1), (1, 2), (1, 0), (1, 3), (2, 3), (3, 0)])

# Ici la variable "colors" correspond la liste des couleurs.
colors = list(nx.get_node_attributes(H, "color").values())

# Ici la variable "sizes" correspond la liste des poids.
sizes = list(nx.get_node_attributes(H, "size").values())


# Graph directionnel (pour se faire, il faut décommenter les deux lignes suivantes et commenter les lignes qui initie "G" jusqu'à la fin du code.)
nx.draw(
    H, with_labels=True, node_color=colors, node_size=sizes
)  # La commande "draw" sert à dessiner le graph
plt.savefig(
    "./graph/graphH.png"
)  # Ici le graph dessiné est sauvegardé dans le dossier et nommé comme souhaité

# Pour faire ce graph, décommenter les lignes ci-dessous et commenter les deux au-dessus

# G = H.to_undirected()
# nx.draw(G, with_labels=True, node_color=colors, node_size=sizes)
# plt.savefig("./graph/graphG.png")
