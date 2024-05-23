from functions import GCN, visualize, dataset

print(f'Dataset: {dataset}')
print('======================')
print(f'Number of graphs : {len(dataset)}')
print(f'Number of features : {dataset.num_features}')
print(f'Number of classes : {dataset.num_classes}')

data = dataset[0]

print(data)

model = GCN(hidden_channels=16)
print(model)

model.eval()

out = model(data.x, data.edge_index)
visualize(out, color=data.y)