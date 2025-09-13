import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms as T
from PIL import Image


# Step 1- Load data
df = pd.read_csv("data/fmnist_small.csv")

# print(df.head())


# Step 2 Pre-Processing

# train test split

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply transformations here

custom_transform = T.Compose([

    T.RandomHorizontalFlip(p=0.5),  # Flip left/right (works for most clothing)
    T.RandomRotation(degrees=15),   # Small rotations simulate different orientations
    T.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Slight shifting
    T.RandomResizedCrop(28, scale=(0.8, 1.0)),  # Random zoom-in
    T.ToTensor(),                   # Convert to tensor [0,1]
    T.Normalize((0.5,), (0.5,))     # Normalize to [-1, 1]   
])


# print(X_train)


# Create a Datatset and Dataloader class

class dataset(Dataset):
    def __init__(self, features, labels, transform=custom_transform):
        self.features = torch.tensor(features, dtype=torch.float32).reshape(-1,1,28,28)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform= transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

    

        # image resize (28, 28)
        img = self.features[idx].reshape(28,28)

        img = img.numpy()

        # convert to np.uint8
        img = img.astype(np.uint8)


        # convert arr to PIL image
        img = Image.fromarray(img)

        # Transformation apply
        img = self.transform(img) # apply all transformation we did above with converted to tensor

        return img, torch.tensor(self.labels[idx], dtype=torch.long)
    


# create dataset objects
train_dataset = dataset(X_train, y_train)
test_dataset = dataset(X_test, y_test)

# create dataloader objects
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# print(len(train_dataloader))

