from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import (
    TSNE,
)  # Classe avec laquelle on visualise des ensembles de données de haute dimensionnalité dans un espace à deux dimensions afin de faciliter l'identification de clusters ou de tendances dans les données.
from datetime import datetime

from torch_geometric.datasets import (
    Planetoid,
)  # Ensemble de données duquel sera extrait le graph utilisé pour la classification des noeuds
from torch_geometric.transforms import (
    NormalizeFeatures,
)  # Fonction qui normalise les caractéristiques dans un contexte d'apprentissage automatique

dataset = Planetoid(
    root="data/Planetoid", name="Cora", transform=NormalizeFeatures()
)  # Cette variable appelle le jeu de données *Cora*

# Ajouter des commentaires pour bien documenter


class GCN(torch.nn.Module):  # Définition de la classe GCN

    def __init__(self, hidden_channels):
        super().__init__()  # Appel au constructeur de la classe parente torch.nn.Module
        torch.manual_seed(1234567)  # Pour fixer la graine de générations aléatoires
        self.conv1 = GCNConv(
            dataset.num_features, hidden_channels
        )  # Variable contenant la première couche de convolution de graph avec deux critères d'entrées le nombre d'attributs par noeud ainsi que les canaux cachés.
        self.conv2 = GCNConv(
            hidden_channels, dataset.num_classes
        )  # Variable contenant la seconde couche de convolution de graph avec deux critères d'entrées les canaux cachés ainsi que le nombre de classes.

    def forward(self, x, edge_index):  # Méthode de propagation
        x = self.conv1(x, edge_index)  # Première couche de convolution
        x = x.relu()  # Activation ReLU après la première couche de convolution
        x = F.dropout(
            x, p=0.5, training=self.training
        )  # Désactivation de neurones entre deux couches
        x = self.conv2(x, edge_index)
        return x  # Renvoi des sorties de la dernière couche


def train_gcn(
    model, optimizer, criterion, data
):  # Fonction qui entraîne le modèle et retourne la perte du modèle
    model.train()  # Mise en mode "entraînement" du modèle
    optimizer.zero_grad()  # Réinitialisation des gradients accumulés dans le stockage du gradient de l'optimiseur
    out = model(data.x, data.edge_index)  # Application du modèle au dataset
    print(out)
    loss = criterion(
        out[data.train_mask], data.y[data.train_mask]
    )  # Calcul de la perte en utilisant la méthode choisit dans la variable criterion
    loss.backward()  # Calcul des gradients du modèle par rapport à la perte qui est nécessaire à l'étape suivante
    optimizer.step()  # MAJ des poids du modèle lors de l'apprentissage
    return loss


def test_gcn(model, data):  # Fonction qui teste le modèle et la précision du test
    model.eval()  # Mise en mode "évaluation" du modèle
    out = model(data.x, data.edge_index)  # Application du modèle au dataset
    print(out)
    pred = out.argmax(
        dim=1
    )  # Prédiction du modèle en prenant l'indice de la classe avec la valeur maximale pour chaque noeud
    print(pred)
    test_correct = (
        pred[data.test_mask] == data.y[data.test_mask]
    )  # Comparaison des prédictions avec les vraies étiquettes pour les noeuds de test
    test_acc = int(test_correct.sum()) / int(
        data.test_mask.sum()
    )  # Calcul le nombre de prédictions correctes à diviser avec les nombre total de noeuds de test
    return test_acc


def visualize_gcn(
    h, color
):  # Fonction de visualisation du graphique en nuage de points
    z = TSNE(n_components=2).fit_transform(
        h.detach().cpu().numpy()
    )  # Réduction de la multidimensionnalité de l'objet en deux dimensions

    current_date = datetime.now()
    ts_int = int(
        round(current_date.timestamp())
    )  # Variable qui récupère le timestamp au moment où est généré le graphique pour l'inclure dans le

    plt.figure(figsize=(10, 10))  # Taille de l'espace graphique
    plt.xticks([])  # Nom de l'abscisse
    plt.yticks([])  # Nom de l'ordonnée

    plt.scatter(
        z[:, 0], z[:, 1], s=70, c=color, cmap="Set2"
    )  # Choix du type de graphique et ses propriétés
    plt.savefig(f"./graph/scatter_gcn{ts_int}.png")  # Sauvegarde du graphique
    plt.show()


class GAT(torch.nn.Module):  # Définition de la classe GCN
    def __init__(self, hidden_channels, heads):
        super().__init__()  # Appel au constructeur de la classe parente torch.nn.Module
        torch.manual_seed(1234567)  # Pour fixer la graine de générations aléatoires
        self.conv1 = GATConv(
            dataset.num_features, hidden_channels, heads
        )  # Variable contenant la première couche de convolution de graph avec trois critères d'entrées le nombre d'attributs par noeud ainsi que les canaux cachés et les heads.
        self.conv2 = GATConv(
            heads * hidden_channels, dataset.num_classes, heads
        )  # Variable contenant la seconde couche de convolution de graph avec trois critères d'entrées les heads multiplié par les canaux cachés, le nombre de classes ainsi que les heads

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)  # Désactivation de neurones
        x = self.conv1(x, edge_index)  # Première couche de convolution
        x = F.elu(
            x
        )  # Fonction d'activition qui introduit la non-linéarité dans le modèle (exploite les valeurs négatives contrairement au ReLU)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # Renvoi des sorties de la dernière couche de convolution


def train_gat(
    model, optimizer, criterion, data
):  # Fonction qui entraîne le modèle et retourne la perte du modèle
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


def test_gat(model, data, mask):  # Fonction qui teste le modèle et la précision du test
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = pred[mask] == data.y[mask]
    acc = int(correct.sum()) / int(mask.sum())
    return acc


# Fonction qui trace valeurs d'accélération de validation ...


def curve_gat(val_acc_all, test_acc_all):
    plt.figure(figsize=(12, 8))
    plt.plot(
        np.arange(
            1, len(val_acc_all) + 1
        ),  # ... selon les indices créés par la fonction np.arrange qui prend 1 et len(val_acc_all)
        val_acc_all,
        label="Validation accuracy",
        c="blue",
    )
    plt.plot(
        np.arange(1, len(test_acc_all) + 1),
        test_acc_all,
        label="Testing accuracy",
        c="red",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accurarcy")
    plt.title("GATConv")
    plt.legend(loc="lower right", fontsize="x-large")
    plt.savefig("./graph/gat_loss.png")
    plt.show()


def visualize_gat(
    h, color
):  # Fonction de visualisation du graphique en nuage de points
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    current_date = datetime.now()
    ts_int = int(round(current_date.timestamp()))

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.savefig(f"./graph/scatter_gat{ts_int}.png")
    plt.show()
