import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

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

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
batch_size = 512
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the DNN model
class DNN(nn.Module):
    def __init__(self, input_dim):
        super(DNN, self).__init__()
        self.hidden = nn.Linear(input_dim, 300)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(300, 2)
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.dropout(x)
        x = self.output(x)
        return torch.log_softmax(x, dim=1)

# Initialize the model, loss function, and optimizer
model = DNN(input_dim=X_train.shape[1])
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        X_batch, y_batch = batch
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss / len(train_loader)}')

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy}')
