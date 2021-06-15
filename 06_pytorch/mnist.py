from torch import optim
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

torch.manual_seed(1337)

# Get datasets
transform = transforms.ToTensor()
train_set = datasets.MNIST(
    root="data/",
    train=True,
    download=True,
    transform=transform
)

test_set = datasets.MNIST(
    root="data/",
    train=False,
    download=True,
    transform=transform
)

# Loaders
batch_size = 128
train_loader = torch.utils.data.DataLoader(
    train_set, shuffle=True, batch_size=batch_size
)

test_loader = torch.utils.data.DataLoader(
    test_set, shuffle=False, batch_size=batch_size
)

# Device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

def accuracy(y, t):
    return torch.sum(y == t) / y.shape[0]

def train(model, train_loader, test_loader, epochs=20):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=0.01,
        momentum=0.9
    )

    # Training
    n_epochs = 20
    for epoch in range(n_epochs):
        train_acc = 0
        test_acc = 0
        # Train
        model.train()
        for x, t in train_loader:
            x, t = x.to(device), t.to(device)
            optimizer.zero_grad()

            z = model(x)
            pred = torch.argmax(z, dim=1)
            J = loss(z, t)

            J.backward()
            optimizer.step()

            train_acc += accuracy(pred, t)

        # Validation
        model.eval()
        for x, t in test_loader:
            x, t = x.to(device), t.to(device)
            z = model(x)
            pred = torch.argmax(z, dim=1)

            test_acc += accuracy(pred, t)

        train_acc = train_acc / len(train_loader) * 100
        test_acc = test_acc / len(test_loader) * 100
        print(f"\tEpoch #{epoch} | Train acc {train_acc:.2f}% | Test acc {test_acc:.2f}%")


# Task 1. - Simple FC network
def get_fc_network(n_inputs, n_hidden=1024, n_outputs=10):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_inputs, n_hidden),
        nn.Sigmoid(),
        nn.Linear(n_hidden, n_outputs)
        # Returns logits / no softmax here
    ).to(device)

# MNIST WxH
n_inputs = 28 * 28
K = 1024
O = 10
fc_network = get_fc_network(n_inputs, n_hidden=K, n_outputs=O)

print("FC network")
train(fc_network, train_loader, test_loader)

# Task 2. - FC with convolutions
class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv = nn.Conv2d(1, 16, 6, 1)
        self.conv2 = nn.Conv2d(16, 32, 6, 1)
        self.pooling = nn.MaxPool2d(4)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.pooling(x)
        # Flatten with start_dim=1 (to keep the batch size)
        x = torch.flatten(x, start_dim=1)

        x = self.fc(x)

        return x

print("\nConvNet")
train(ConvNet().to(device), train_loader, test_loader)
