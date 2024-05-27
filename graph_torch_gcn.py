import torch

from functions import GCN, visualize_gcn, dataset, train_gcn, test_gcn

print(f"Dataset: {dataset}")
print("======================")
print(f"Number of graphs : {len(dataset)}")
print(f"Number of features : {dataset.num_features}")
print(f"Number of classes : {dataset.num_classes}")

data = dataset[0]

print(data)

model = GCN(hidden_channels=16)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

model.eval()

out = model(data.x, data.edge_index)
visualize_gcn(out, color=data.y)

for epoch in range(1, 101):
    loss = train_gcn(model, optimizer, criterion, data)
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

test_acc = test_gcn(model, data)
print(f"Test Accuracy: {test_acc:.4f}")

model.eval()
out = model(data.x, data.edge_index)
visualize_gcn(out, color=data.y)
