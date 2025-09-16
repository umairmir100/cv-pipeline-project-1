# Model architecture 
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import train_dataloader, test_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyModel(nn.Module):
    def __init__(self, input_features_channel, num_classes = 10):
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
            nn.Linear(64,num_classes)
    )                            # 10 classes for output  
        

    def forward(self, input):
        features_new = self.features_extracted(input)
        out = self.classifier(features_new)

        return out



