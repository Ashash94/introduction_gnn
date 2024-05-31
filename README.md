# introduction_gnn
Introduction au GNN (Graph Neural Network)

Afin de m'initier aux graphs, j'ai suivi le tuto de [Datacamp](https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial). On y retrouve du code pour créer des graphs à l'aide de NetworkX et à classifier les noeuds avec deux types d'architectures de GNN : GCN & GAT. Leurs principes diffèrent notamment par leur méthodes d'agrégation des données.

**GCN** : Utilise des opérations de convolution (multiplication de matrices) pour propager des informations entre les noeuds dans un graphe et rassembler les carastéristiques des noeuds voisins pour mettre à jour la représentation du noeud traité. Le GCN est un bon choix pour des tâches nécessitant une approche plus uniforme de l'information graphique.

**GAT** : Utilise des mécanismes d'attention. Des poids d'attention sont attribués aux noeuds voisins pour mettre en évidence les liens forts entre ceux-ci et le noeud cible afin de se concentrer sur les voisins les plus pertinents lors du processus du propagation d'informations. Ke GAT est particulièrement utile pour les tâches impliquant des informations relationnelles complexes.

## Structure
```bash
project/
│
├── graph/ # Dossier à créer
│   └── Les fichiers correspondent aux graphs ainsi qu'aux graphiques en nuages de points et en courbes au format .png générés par
│       les différents codes .py
├── .gitignore
├── functions.pygg
├── graph_nx.py
├── graph_torch_gcn.py
├── graph_torch_gat.py
├── requirements.txt
└── README.md
```

## Jeu de données
Le dataset *cora* provient de l'ensemble de données de citations "Planetoid" disponible dans Pytorch. Nous exploiterons le premier graph du dataset duquel on classifiera les noeuds. 
Voici les informations sur ce graph : 
*Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])*
- *"x=[2708, 1433]"* : reprend la taille de la matrice, 2708 noeuds et 1433 éléments qui décrivent les caractéristiques de chaque noeud.
- *"edge_index=[2, 10556]"* : La première valeur correspond au nombre de colonne dans le tenseur de l'index d'arrêtes. Chacune des colonnes contient l'indice des noeuds que les arrêtes relient. Puis la seconde valeur correspond au nombre d'arrêtes dans le graph.
- *"y=[2708]"* : C'est la valeur cible de prédiction. Dans ce cas précis, nous souhaitons la classification des 2708 noeuds présents dans notre graph, chaque noeud doit correspondre à une classe.
- *train_mask=[2708], val_mask=[2708], test_mask=[2708]* : chaque noeud parmi les 2708 seront séparés et répartis dans l'un des trois groupes :
"train" = données d'entraînement, "val" = données de validation et "test" = données de test.

## Fichiers 

### **.gitignore**
Ce fichier permet d'ignorer les dossiers *graph* et *data* ainsi que les fichiers *.png* et le *pycache*.

### **functions.py**
Ce fichier contient les classes et les fonctions nécessaires à l'exécution des fichiers *graph_torch_gcn.py* et *graph_torch_gat.py* qui utilisent les opérateurs de convolutions [GCN](https://arxiv.org/abs/1609.02907) (Graph Convulotional Network) et [GAT](https://arxiv.org/abs/1710.10903) (Graph Attention Network). Le détail est dans le code.

### **graph_nx.py**
Le fichier *graph_nx.py* permet de créer un graph avec la bibliothèque Netwrorx. Chaque noeud a un attribut, une couleur et un poids. Les arrêtes ont deux coordonées, les attributs des deux noeuds qu'ils relient. Dans le cas du graph directionnel, l'ordre d'écriture est importante. Ex : *I* et *J* sont deux noeuds dans un graph et on souhaite que l'arrête parte de *J* vers *I* dans ce cas-là on l'écrit "[(J, I)]". 

### **graph_torch_gcn.py**
Ce code renvoie les résultats des epochs selon la méthode GCN et génère deux graphiques en nuage de points.

### **graph_torch_gat.py**
Ce code renvoie les résultats des epochs selon la méthode GAT et génère deux graphiques un en courbe et un autre en nuage de points.

### **requirements.txt**

Les bibliothèques nécessaires à l'exécution du code sont listées dans ce fichier.

- **networkx** : La bibliothèque NetworkX est utilisée pour créer les graphs dans le fichier *graph_nx.py*.
- **torch** : Du package Pytorch est extrait le dataset *Cora* sur lequel sera entrainé les modèles de classification des noeuds avec différentes couches de convolution (GCN & GAT).
- **matplotlib** : Pour la visualisation de la classification des graphs et de la pertinence du modèle sur les données de validation et de test.
- **scikit-learn** : Le module *sklearn.manifold* duquel est importé la classe TSNE (pour "t-Distributed Stochastic Neighbor Embedding") pour réduire la dimensionnnalité des données.
- **numpy** : Cette bibliothèque est utilisée pour manipuler des matrices ou des tableaux multidimensionnels.
- **datetime** : Pour obtenir l'heure à laquelle sont générés les graphiques en nuage de point afin d'en tirer le timestamp qui sera ajouté au nom auquel est enregistré le graphique.