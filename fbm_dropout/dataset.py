import torch
from torchvision import datasets, transforms

def get_MNIST_dataset(dir: str):
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),])

    trainset = datasets.MNIST(dir, download=True, train=True, transform=transform)
    testset = datasets.MNIST(dir, download=True, train=False, transform=transform)

    return trainset, testset

def get_toy_dataset(n_train: int, n_test: int, noise: float):
    
    X_train = torch.unsqueeze(torch.linspace(-1, 1, n_train), 1)
    Y_train = X_train + noise * torch.normal(torch.zeros(n_train, 1), torch.ones(n_train, 1))

    X_test = torch.unsqueeze(torch.linspace(-1, 1, n_test), 1)
    Y_test = X_test + noise * torch.normal(torch.zeros(n_test, 1), torch.ones(n_test, 1))

    return (X_train, Y_train), (X_test, Y_test)