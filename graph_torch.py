from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Planetoid', name = 'Cora', transform = NormalizeFeatures())

print(f'Dataset: {dataset}')
print('======================')
print(f'Number of graphs : {len(dataset)}')
print(f'Number of features : {dataset.num_features}')
print(f'Number of classes : {dataset.num_classes}')

data = dataset[0]

print(data)