import torch
from functions import GAT, curve_gat, dataset, train_gat, test_gat, visualize_gat

data = dataset[0]

print(data)

model = GAT(hidden_channels=8, heads=8)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


val_acc_all = []
test_acc_all = []

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

curve_gat(val_acc_all, test_acc_all)

model.eval()

out = model(data.x, data.edge_index)
visualize_gat(out, color=data.y)
