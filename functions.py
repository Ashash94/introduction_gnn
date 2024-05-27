from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from datetime import datetime

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root="data/Planetoid", name="Cora", transform=NormalizeFeatures())


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


def visualize_gcn(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    current_date = datetime.now()
    ts_int = int(round(current_date.timestamp()))

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.savefig(f"./graph/scatter_gcn{ts_int}.png")
    plt.show()


def train_gcn(model, optimizer, criterion, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


def test_gcn(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GATConv(dataset.num_features, hidden_channels, heads)
        self.conv2 = GATConv(heads * hidden_channels, dataset.num_classes, heads)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train_gat(model, optimizer, criterion, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


def test_gat(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = pred[mask] == data.y[mask]
    acc = int(correct.sum()) / int(mask.sum())
    return acc


def curve_gat(val_acc_all, test_acc_all):
    plt.figure(figsize=(12, 8))
    plt.plot(
        np.arange(1, len(val_acc_all) + 1),
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


def visualize_gat(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    current_date = datetime.now()
    ts_int = int(round(current_date.timestamp()))

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.savefig(f"./graph/scatter_gat{ts_int}.png")
    plt.show()
