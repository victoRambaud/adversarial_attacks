import torch.nn as nn


class ConvClassifier(nn.Module):

    def __init__(self, ):
        super(ConvClassifier, self).__init__()

        # encoder
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 10)

        self.ReLU = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=False)
        self.avgpool = nn.AvgPool2d(3, stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.ReLU(x)
        x = self.pool(x)
        # print('conv1', x.shape)

        x = self.conv2(x)
        x = self.ReLU(x)
        x = self.pool(x)
        # print('conv2', x.shape)

        x = self.conv3(x)
        x = self.ReLU(x)
        x = self.pool(x)
        # print('conv3', x.shape)
        x = self.avgpool(x)
        # print('avgpool', x.shape)

        x = x.reshape(x.size(0), -1)
        # print('flatten', x.shape)
        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.fc2(x)
        return self.sigmoid(x)
