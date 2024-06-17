# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# import torch
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv
# from torch_geometric.loader import DataLoader

# # Load the data
# file_path = 'UNSW_NB15_training-set.csv'
# df = pd.read_csv(file_path)

# # drop the 'attack_cat' column
# df = df.drop(columns=['attack_cat'])

# # Encode categorical features
# categorical_columns = ['proto', 'service', 'state']
# label_encoders = {}
# for col in categorical_columns:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#     label_encoders[col] = le

# # Normalize numerical features
# numerical_columns = df.columns.difference(['id', 'label'] + categorical_columns)
# scaler = StandardScaler()
# df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# # Prepare features and labels
# features = df.drop(['id', 'label'], axis=1).values
# labels = df['label'].values

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# # Construct a graph (for simplicity, we will use k-NN graph)
# from sklearn.neighbors import kneighbors_graph

# def construct_graph(X, k=5):
#     A = kneighbors_graph(X, k, mode='connectivity', include_self=True)
#     edge_index = np.array(A.nonzero())
#     return torch.tensor(edge_index, dtype=torch.long)

# edge_index = construct_graph(X_train)

# # Prepare PyTorch Geometric data object
# data = Data(x=torch.tensor(X_train, dtype=torch.float), edge_index=edge_index, y=torch.tensor(y_train, dtype=torch.long))

# # Define GNN Model
# class GNN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(GNN, self).__init__()
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, output_dim)
    
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = torch.relu(x)
#         x = self.conv2(x, edge_index)
#         return torch.log_softmax(x, dim=1)

# # Instantiate the model, define the optimizer and loss function
# model = GNN(input_dim=X_train.shape[1], hidden_dim=64, output_dim=2)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# loss_fn = torch.nn.CrossEntropyLoss()

# # Training loop
# def train(data):
#     model.train()
#     optimizer.zero_grad()
#     out = model(data)
#     loss = loss_fn(out, data.y)
#     loss.backward()
#     optimizer.step()
#     return loss.item()

# # Train the model
# epochs = 100
# for epoch in range(epochs):
#     loss = train(data)
#     if epoch % 10 == 0:
#         print(f'Epoch {epoch}, Loss: {loss}')

# # Test the model
# def test(data):
#     model.eval()
#     with torch.no_grad():
#         logits = model(data)
#         pred = logits.argmax(dim=1)
#         accuracy = (pred == data.y).sum().item() / len(data.y)
#     return accuracy

# # Convert test set to PyTorch Geometric data object
# test_edge_index = construct_graph(X_test)
# test_data = Data(x=torch.tensor(X_test, dtype=torch.float), edge_index=test_edge_index, y=torch.tensor(y_test, dtype=torch.long))

# # Evaluate the model
# accuracy = test(test_data)
# print(f'Test Accuracy: {accuracy}')


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.nn import Linear, ReLU, Dropout, CrossEntropyLoss
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import matplotlib.pyplot as plt

# Load the data
file_path = 'UNSW_NB15_training-set.csv'
df = pd.read_csv(file_path)

# Drop the 'attack_cat' column
df = df.drop(columns=['attack_cat'])

# Encode categorical features
categorical_columns = ['proto', 'service', 'state']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Normalize numerical features
numerical_columns = df.columns.difference(['id', 'label'] + categorical_columns)
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Prepare features and labels
features = df.drop(['id', 'label'], axis=1).values
labels = df['label'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Construct a graph (for simplicity, we will use k-NN graph)
from sklearn.neighbors import kneighbors_graph

def construct_graph(X, k=5):
    A = kneighbors_graph(X, k, mode='connectivity', include_self=True)
    edge_index = np.array(A.nonzero())
    return torch.tensor(edge_index, dtype=torch.long)

edge_index = construct_graph(X_train)

# Prepare PyTorch Geometric data object
data = Data(x=torch.tensor(X_train, dtype=torch.float), edge_index=edge_index, y=torch.tensor(y_train, dtype=torch.long))

# Define GNN Model
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Instantiate the model, define the optimizer and loss function
input_dim = X_train.shape[1]
hidden_dim = 300
output_dim = 2
dropout_rate = 0.5

model = GNN(input_dim, hidden_dim, output_dim, dropout_rate)
optimizer = Adam(model.parameters(), lr=0.0005)
loss_fn = CrossEntropyLoss()

# Training loop
def train_model(model, data, optimizer, loss_fn, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# Train the model
train_model(model, data, optimizer, loss_fn)

# Test the model
def test_model(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        accuracy = (pred == data.y).sum().item() / len(data.y)
    return accuracy

# Convert test set to PyTorch Geometric data object
test_edge_index = construct_graph(X_test)
test_data = Data(x=torch.tensor(X_test, dtype=torch.float), edge_index=test_edge_index, y=torch.tensor(y_test, dtype=torch.long))

# Evaluate the model
accuracy = test_model(model, test_data)
print(f'Test Accuracy: {accuracy}')

# # Visualize the graph
# def visualize_graph(edge_index, X):
#     G = nx.Graph()
#     edge_index = edge_index.numpy()
#     for i in range(edge_index.shape[1]):
#         G.add_edge(edge_index[0, i], edge_index[1, i])
    
#     pos = nx.spring_layout(G)
#     plt.figure(figsize=(10, 10))
#     nx.draw(G, pos, node_size=50, node_color=X[:, 0], cmap=plt.get_cmap('coolwarm'), edge_color='gray', with_labels=False)
#     plt.show()

# # Visualize the training graph
# visualize_graph(edge_index, X_train)
