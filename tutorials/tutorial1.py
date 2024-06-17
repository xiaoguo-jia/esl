import torch_geometric 
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='tutorial1', name='Cora')

# properties of dataset
print("number graphs: ", len(dataset))
print("number classes: ", dataset.num_classes)
print("number node features: ", dataset.num_node_features)      # each node has this many features
print("number edge features: ", dataset.num_edge_features)

# print dataset shapes
print(dataset.data)

# edge_index is the edge list, shape of edge_index is 2 * number of edges
# gives two arrays, one for the source node and one for the target node
print("edge index:", dataset.data.edge_index.shape)
print(dataset.data.edge_index)

# one array, bool depending on if the node is in the training set
print("train mask:", dataset.data.train_mask.shape)
print(dataset.data.train_mask)

# x is the node feature, shape of x is number of nodes x node features
print("x:", dataset.data.x.shape)
print(dataset.data.x)

# y is the target (labelm category that the node belongs in), shape of y is number of nodes * 1
print("y:", dataset.data.y.shape)
print(dataset.data.y)

import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

data = dataset[0]           # there is only one dataset and that dataset only has one graph

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # num_features is the number of features for each node (1433), num_classes is the number of classes (7)
        # input is num_features, output is num_classes
        self.conv = SAGEConv(data.num_features, dataset.num_classes, aggr='max')

    def forward(self):
        x = self.conv(data.x, data.edge_index)
        return F.log_softmax(x, dim=1)
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

best_val_acc = test_acc = 0
for epoch in range(10):
    train
    _, val_acc, temp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = temp_test_acc
    log = 'Epoch: {:03d}, Val: {:.4f}, Test: {:.4f}'

    print(log.format(epoch, best_val_acc, test_acc))