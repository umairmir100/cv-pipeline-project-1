# Model architecture 
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import train_dataloader, test_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyModel(nn.Module):
    def __init__(self, input_features_channel):
        super().__init__()
        self.features_extracted = nn.Sequential(
            nn.Conv2d(in_channels=input_features_channel, out_channels=32, kernel_size=(3,3), stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2,2)),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2,2)),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2,2)),
        )

        # classifier
        self.classifier = nn.Sequential(
        # ANN
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),

            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64), #HL2
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64,10)
    )                            # 10 classes for output  
        

    def forward(self, input):
        features_new = self.features_extracted(input)
        out = self.classifier(features_new)

        return out



# init parameters
learning_rate = 0.001
epochs = 100

model = MyModel(1) # input feature channels
model.to(device)

loss_func = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)


# Training loop

for epoch in range(epochs):
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