from torchvision import datasets, transforms

def get_MNIST_dataset():
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),])

    trainset = datasets.MNIST('~/datasets', download=True, train=True, transform=transform)
    testset = datasets.MNIST('~/datasets', download=True, train=False, transform=transform)

    return trainset, testset