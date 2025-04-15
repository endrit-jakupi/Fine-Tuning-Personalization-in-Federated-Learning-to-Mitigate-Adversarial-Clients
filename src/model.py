import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import sys


class Logistic2(torch.nn.Module):
    """ Simple, small fully connected model.
    """

    def __init__(self, nb_classes):
        """ Model parameter constructor.
        """
        super().__init__()
        # Build parameters
        self._f1 = torch.nn.Linear(28 * 28, nb_classes)

    def forward(self, x):
        """ Model's forward pass.
        Args:
            x Input tensor
        Returns:
            Output tensor
        """
        # Forward pass
        x = self._f1(x.view(-1, 28 * 28))
        return x
    

class CNN(torch.nn.Module):
    """ Simple, small CNN model.
    """

    def __init__(self, nb_classes):
        """ Model parameter constructor.
        """
        super().__init__()
        # Build parameters
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)

        self.fc1 = nn.Linear(4*4*64, 1024)
        self.fc2 = nn.Linear(1024, nb_classes)

    def forward(self, x):
        # Forward pass
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))

        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Logistic2(torch.nn.Module):
    """ Simple, small fully connected model.
    """

    def __init__(self, nb_classes):
        """ Model parameter constructor.
        """
        super().__init__()
        # Build parameters
        
        self._f1 = torch.nn.Linear(122, 32)
        self._f2 = torch.nn.Linear(32, 16)
        self._f3 = torch.nn.Linear(16, 2)
        

    def forward(self, x):
        """ Model's forward pass.
        Args:
            x Input tensor
        Returns:
            Output tensor
        """
        # Forward pass
        x = torch.nn.ReLU()(self._f1(x.view(-1, 122)))
        x = torch.nn.ReLU()(self._f2(x))
        x = self._f3(x)
        return x
    
class Logistic_phishing(torch.nn.Module):
    """ Simple, small fully connected model.
    """

    def __init__(self, nb_classes):
        """ Model parameter constructor.
        """
        super().__init__()
        # Build parameters
        
        self._f1 = torch.nn.Linear(68,2)
        

    def forward(self, x):
        """ Model's forward pass.
        Args:
            x Input tensor
        Returns:
            Output tensor
        """
        # Forward pass
        x = self._f1(x)
        return x



class Logistic_MNIST_bin(torch.nn.Module):
    """ Simple, small fully connected model.
    """

    def __init__(self, nb_classes):
        """ Model parameter constructor.
        """
        super().__init__()
        # Build parameters
        
        self._f1 = torch.nn.Linear(28*28,2)
        

    def forward(self, x):
        """ Model's forward pass.
        Args:
            x Input tensor
        Returns:
            Output tensor
        """
        # Forward pass
        x = self._f1(x.view(-1, 28 * 28))
        return x



class CifarNet(nn.Module):
    # From pytorch tutorial
    def __init__(self, nb_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, nb_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x


ResNet = torchvision.models.resnet18
models = { "cnn": CNN , "cifarnet": CifarNet, "simplecnn": SimpleCNN, "resnet": ResNet, "logistic_phishing": Logistic_phishing, "logistic_mnist_bin": Logistic_MNIST_bin}
#models = {"logistic": Logistic , "cnn": CNN , "cifarnet": CifarNet, "simplecnn": SimpleCNN, "resnet": ResNet}


def setup_model(model_name, args, seed = 0):
    torch.manual_seed(seed)
    return models[model_name](args)