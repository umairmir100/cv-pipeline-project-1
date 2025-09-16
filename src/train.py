from model import MyModel
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import train_dataloader, test_dataloader
import yaml


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open("config.yaml") as f:
    config = yaml.safe_load(f)

# extract config parameters
LR = config["training"]["lr"]
EPOCHS = config["training"]["epochs"]

model = MyModel(input_features_channel=1, num_classes=config["model"]["num_classes"])
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
loss_func= torch.nn.CrossEntropyLoss()



# Training loop

for epoch in range(EPOCHS):
  for batch_features, batch_labels in train_dataloader:

    # get em on gpu as well
    batch_features = batch_features.to(device)
    batch_labels = batch_labels.to(device)

    # forward
    out = model(batch_features)

    # loss
    loss = loss_func(out, batch_labels)

    # back prop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f'Epoch {epoch + 1} loss {loss.item()}')

# Train eval
model.eval()
total = 0
correct = 0

with torch.no_grad():
    for batch_features, batch_labels in train_dataloader:
      batch_features = batch_features.to(device)
      batch_labels = batch_labels.to(device)

      out = model(batch_features)
      _, pred = torch.max(out, 1)

      total = total + batch_labels.shape[0]
      correct = correct + (pred == batch_labels).sum().item()

    accuracy= correct/total

    print("train_accuracy: ",accuracy)


    # test eval

total = 0
correct = 0

with torch.no_grad():
    for batch_features, batch_labels in test_dataloader:
      batch_features = batch_features.to(device)
      batch_labels = batch_labels.to(device)

      out = model(batch_features)
      _, pred = torch.max(out, 1)

      total = total + batch_labels.shape[0]
      correct = correct + (pred == batch_labels).sum().item()

    accuracy= correct/total

    print("test_accuracy: ",accuracy)