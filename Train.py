import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from Net import CnnNet
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.cuda.empty_cache()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])

train_path = './data/cifar10/cifar10/train'
test_path = './data/cifar10/cifar10/test'

train_data = ImageFolder(train_path, transform=transform)
test_data = ImageFolder(test_path, transform=transform)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CnnNet()
model.to(device)

lr = 0.001
batch_size = 16
num_epochs = 25
running_loss = 0.0

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

    print(f'Epoch {epoch + 1}, Accuracy: {accuracy:.4f}%')

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': running_loss
}, 'models/CIFAR-10_model.pth')




