import torch.nn as nn


class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=2, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        self.drop5 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(in_features=4096, out_features=1024)
        self.relu5 = nn.ReLU()

        self.drop6 = nn.Dropout(0.3)
        self.fc6 = nn.Linear(in_features=1024, out_features=256)
        self.relu6 = nn.ReLU()

        self.fc7 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = x.reshape(x.size(0), -1)
        # print('X after reshape shape: ', x.shape)

        x = self.drop5(x)
        x = self.fc5(x)
        x = self.relu5(x)

        x = self.drop6(x)
        x = self.fc6(x)
        x = self.relu6(x)

        x = self.fc7(x)

        return x


