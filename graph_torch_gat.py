import torch
from functions import GAT, curve_gat, dataset, train_gat, test_gat, visualize_gat

# Renvoi d'informations sur le data set

print(f"Dataset: {dataset}")
print("======================")
print(f"Number of graphs : {len(dataset)}")
print(f"Number of features : {dataset.num_features}")
print(f"Number of classes : {dataset.num_classes}")

# Premier graph du dataset
data = dataset[0]

print(data)

model = GAT(hidden_channels=8, heads=8)

optimizer = torch.optim.Adam(
    model.parameters(), lr=0.005, weight_decay=5e-4
)  # Optimisation avec Adam avec 3 paramètres en entrée
criterion = (
    torch.nn.CrossEntropyLoss()
)  # Fonction de perte d'entropie croisée pour mesurer la différence entre les probabilités prédites et les labels des noeuds


val_acc_all = (
    []
)  # Création d'une liste qui reprendra la valeur de la validation accuracy à chaque epoch
test_acc_all = (
    []
)  # Création d'une liste qui reprendra la valeur du test accuracy à chaque epoch

# Pour chaque itération
for epoch in range(1, 101):
    val_mask = data.val_mask
    test_mask = data.test_mask
    loss = train_gat(model, optimizer, criterion, data)
    val_acc = test_gat(model, data, val_mask)
    test_acc = test_gat(model, data, test_mask)
    val_acc_all.append(val_acc)
    test_acc_all.append(test_acc)
    print(
        f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}"
    )

# Création d'un graphique en courbe qui reprend l'évolution de la précision de la validation et du test
curve_gat(val_acc_all, test_acc_all)

model.eval()

# Création du graphique en nuage de point pour clusteriser
out = model(data.x, data.edge_index)
visualize_gat(out, color=data.y)
