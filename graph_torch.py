from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

dataset = Planetoid(root='data/Planetoid', name = 'Cora', transform = NormalizeFeatures())

print(f'Dataset: {dataset}')
print('======================')
print(f'Number of graphs : {len(dataset)}')
print(f'Number of features : {dataset.num_features}')
print(f'Number of classes : {dataset.num_classes}')

data = dataset[0]

print(data)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN(hidden_channels=16)
print(model)

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.savefig('graph_scatter.png')
    plt.show()

model.eval()

out = model(data.x, data.edge_index)
visualize(out, color=data.y)