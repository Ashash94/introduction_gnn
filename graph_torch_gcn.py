import torch

from functions import GCN, visualize_gcn, dataset, train_gcn, test_gcn

# Renvoi d'informations sur le data set

print(f"Dataset: {dataset}")
print("======================")
print(f"Number of graphs : {len(dataset)}")
print(f"Number of features : {dataset.num_features}")
print(f"Number of classes : {dataset.num_classes}")

# Premier graph du dataset
data = dataset[0]

print(data)

model = GCN(hidden_channels=16)

optimizer = torch.optim.Adam(
    model.parameters(), lr=0.01, weight_decay=5e-4
)  # Optimisation avec Adam avec 3 paramètres en entrée
criterion = (
    torch.nn.CrossEntropyLoss()
)  # Fonction de perte d'entropie croisée pour mesurer la différence entre les probabilités prédites et les labels des noeuds

model.eval()

out = model(data.x, data.edge_index)
visualize_gcn(out, color=data.y)  # première visualisation après le premier entrainement

for epoch in range(1, 101):
    # Ajouter un early stoping
    loss = train_gcn(
        model, optimizer, criterion, data
    )  # valeur retournée par la fonction train_gcn
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

test_acc = test_gcn(model, data)  # valeur retournée par la fonction test_gcn
print(f"Test Accuracy: {test_acc:.4f}")

model.eval()
out = model(data.x, data.edge_index)
visualize_gcn(out, color=data.y)  # seconde visualisation après le second entrainement
