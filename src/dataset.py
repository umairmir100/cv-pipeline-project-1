import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Step 1- Load data
df = pd.read_csv("data/fmnist_small.csv")

# print(df.head())


# Step 2 Pre-Processing

# train test split

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalise the data 

X_train = X_train / 255.0
X_test = X_test / 255.0

# print(X_train)


# Create a Datatset and Dataloader class

class dataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32).reshape(-1,1,28,28)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    


# create dataset objects
train_dataset = dataset(X_train, y_train)
test_dataset = dataset(X_test, y_test)

# create dataloader objects
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# print(len(train_dataloader))

