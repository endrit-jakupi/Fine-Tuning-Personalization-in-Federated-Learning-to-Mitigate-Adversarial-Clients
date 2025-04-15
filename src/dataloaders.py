import torch
import torchvision.transforms as transforms
import numpy as np
from numpy import genfromtxt
import random
from torchvision import datasets, transforms
from sklearn import datasets as sk_datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from torch.utils.data import Dataset
import sys
from tqdm import tqdm, trange
#from src.svm import get_phishing
from torch.utils.data.sampler import SubsetRandomSampler
import logging
import os


def homogenous_mnist_train_test(config, m, test_m = 2000, batch_size = 16, alpha = None, seed=42, save_folder=None):

    #dataset = datasets.MNIST(root='./data', train=True, download=True, transform = transforms.ToTensor())
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

    np.random.seed(seed)
    torch.manual_seed(seed)
    dataloaders_dict = {}
    for i in trange( config['n']-config['f']):
        idx = np.random.choice(len(dataset), m+test_m, replace=False)
        client_dataset = torch.utils.data.Subset(dataset, idx)
        client_train_dataset, client_test_dataset = torch.utils.data.random_split(client_dataset, [m/(m+test_m), test_m / (m+test_m)])
        client_train_loader = torch.utils.data.DataLoader(client_train_dataset, batch_size=batch_size, shuffle=True)
        client_test_loader = torch.utils.data.DataLoader(client_test_dataset, batch_size=128, shuffle=False)
        dataloaders_dict[i] = [client_train_loader, client_test_loader]
    
    return dataloaders_dict

def homogenous_mnist_binary_train_test_old(config, m, test_m = 2000, batch_size = 16, alpha = None, seed=42, save_folder=None):

    #dataset = datasets.MNIST(root='./data', train=True, download=True, transform = transforms.ToTensor())
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    subset_indices = ((dataset.targets == 0) + (dataset.targets == 1)).nonzero().view(-1) # 12665 indices

    np.random.seed(seed)
    torch.manual_seed(seed)
    dataloaders_dict = {}
    for i in trange( config['n']-config['f']):
        idx = np.random.choice(subset_indices, m+test_m, replace=False)
        client_dataset = torch.utils.data.Subset(dataset, idx)
        client_train_dataset, client_test_dataset = torch.utils.data.random_split(client_dataset, [m/(m+test_m), test_m / (m+test_m)])
        client_train_loader = torch.utils.data.DataLoader(client_train_dataset, batch_size=batch_size, shuffle=True)
        client_test_loader = torch.utils.data.DataLoader(client_test_dataset, batch_size=128, shuffle=False)
        dataloaders_dict[i] = [client_train_loader, client_test_loader]
    
    return dataloaders_dict

def homogenous_mnist_binary_train_test_89(config, m, test_m = 2000, batch_size = 16, alpha = None, seed=42, save_folder=None):

    #dataset = datasets.MNIST(root='./data', train=True, download=True, transform = transforms.ToTensor())
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    #subset_indices = ((dataset.targets == 8) + (dataset.targets == 9)).nonzero().view(-1) # 12665 indices for 0,1
    subset_indices = ((dataset.targets == 8) + (dataset.targets == 9) + ((dataset.targets == 7) + (dataset.targets == 6))).nonzero().view(-1) # 12665 indices for 0,1
    dataset.targets = torch.logical_or(dataset.targets == 8, dataset.targets == 7).long()
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataloaders_dict = {}
    for i in trange( config['n']-config['f']):
        idx = np.random.choice(subset_indices, m+test_m, replace=False)
        
        client_train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=SubsetRandomSampler(idx[:m]))
        client_test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=SubsetRandomSampler(idx[m:]))
        #client_dataset = torch.utils.data.Subset(dataset, idx)
        dataloaders_dict[i] = [client_train_loader, client_test_loader]
        #print('train dist', mnist_label_distribution(client_train_loader))
        #print('test dist', mnist_label_distribution(client_test_loader))

    return dataloaders_dict

def homogenous_mnist_binary_train_test(config, m, test_m = 2000, batch_size = 16, alpha = None, seed=42, save_folder=None):

    #dataset = datasets.MNIST(root='./data', train=True, download=True, transform = transforms.ToTensor())
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    #subset_indices = ((dataset.targets == 8) + (dataset.targets == 9)).nonzero().view(-1) # 12665 indices for 0,1
    #subset_indices = ((dataset.targets  8) + (dataset.targets == 9) + ((dataset.targets == 7) + (dataset.targets == 6))).nonzero().view(-1) # 12665 indices for 0,1
    dataset.targets = (dataset.targets <= 4).long()
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataloaders_dict = {}
    for i in trange( config['n']-config['f']):
        idx = np.random.choice(len(dataset.targets), m+test_m, replace=False)
        
        client_train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=SubsetRandomSampler(idx[:m]))
        client_test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=SubsetRandomSampler(idx[m:]))
        #client_dataset = torch.utils.data.Subset(dataset, idx)
        dataloaders_dict[i] = [client_train_loader, client_test_loader]
        #print('train dist', mnist_label_distribution(client_train_loader))
        #print('test dist', mnist_label_distribution(client_test_loader))

    return dataloaders_dict

def dirichlet_mnist_binary_train_test(config, m, test_m = 2000, batch_size = 16, alpha = 1, seed=42, save_folder=None):
    n = config["n"]
    #dataset = datasets.MNIST(root='./data', train=True, download=True, transform = transforms.ToTensor())
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    #subset_indices = ((dataset.targets == 8) + (dataset.targets == 9)).nonzero().view(-1) # 12665 indices for 0,1
    #subset_indices = ((dataset.targets == 8) + (dataset.targets == 9) + ((dataset.targets == 7) + (dataset.targets == 1))).nonzero().view(-1) # 12665 indices for 0,1
    dataset.targets = (dataset.targets <= 4).long()
    #dataset.targets = torch.logical_or(dataset.targets == 8, dataset.targets == 7).long()

    np.random.seed(seed)
    torch.manual_seed(seed)
    #probabilities = np.random.dirichlet(np.ones(2)*alpha, size = n)
    probabilities = np.ones((config["n"], 2))*0.5
    probabilities[0] = np.array([alpha, 1-alpha])
    for j in range(1,config["n"]):
        probabilities[j] = np.array([1-alpha, alpha])
    dataloaders_dict = {}
    for i in range( config['n']-config['f']):
        p = probabilities[i]
        #while min(p)<0.1:
            #p = np.random.dirichlet(np.ones(2)*alpha, size = 1)[0]
        #print(p)
        #selection_p = [p[x]  for  x in dataset.targets[subset_indices]] 
        selection_p = [p[x]  for  x in dataset.targets] 
        selection_p = selection_p/np.sum(selection_p)

        idx = np.random.choice(len(dataset.targets), m+test_m, replace=False, p=selection_p)
        
        client_train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=SubsetRandomSampler(idx[:m]))
        client_test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=SubsetRandomSampler(idx[m:]))
        #client_dataset = torch.utils.data.Subset(dataset, idx)
        dataloaders_dict[i] = [client_train_loader, client_test_loader]
        #print('train dist', mnist_label_distribution(client_train_loader))
        #print('test dist', mnist_label_distribution(client_test_loader))

    return dataloaders_dict

def dirichlet_mnist_binary_train_test_correct_seeding(config, m, test_m = 2000, batch_size = 16, alpha = 1, seed=42, save_folder=None):
    n = config["n"]
    #dataset = datasets.MNIST(root='./data', train=True, download=True, transform = transforms.ToTensor())
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    #subset_indices = ((dataset.targets == 8) + (dataset.targets == 9)).nonzero().view(-1) # 12665 indices for 0,1
    #subset_indices = ((dataset.targets == 8) + (dataset.targets == 9) + ((dataset.targets == 7) + (dataset.targets == 1))).nonzero().view(-1) # 12665 indices for 0,1
    dataset.targets = (dataset.targets <= 4).long()
    #dataset.targets = torch.logical_or(dataset.targets == 8, dataset.targets == 7).long()

    np.random.seed(0)
    probabilities = np.random.dirichlet(np.ones(2)*alpha, size = n)

    dataloaders_dict = {}
    for i in range( config['n']-config['f']):
        p = probabilities[i]
        selection_p = [p[x]  for  x in dataset.targets] 
        selection_p = selection_p/np.sum(selection_p)
        np.random.seed(seed)
        idx = np.random.choice(len(dataset.targets), m+test_m, replace=False, p=selection_p)
        
        client_train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=SubsetRandomSampler(idx[:m]))
        client_test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=SubsetRandomSampler(idx[m:]))
        #client_dataset = torch.utils.data.Subset(dataset, idx)
        dataloaders_dict[i] = [client_train_loader, client_test_loader]
        #print('train dist', mnist_label_distribution(client_train_loader))
        #print('test dist', mnist_label_distribution(client_test_loader))

    return dataloaders_dict


def gamma_mnist_binary_train_test(config, m, test_m = 2000, batch_size = 16, alpha = 1, seed=42, save_folder=None):
    gamma = alpha

    # beta fraction of the training and test dataset of each client will be uniform
    # the rest will be one class only
    n = config["n"]

    # Using a binary version of MNSIT
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))


    train_dataset.targets = (train_dataset.targets <= 4).long()
    test_dataset.targets = (test_dataset.targets <= 4).long()

    #train_dataset = torch.utils.data.Subset(train_dataset, range(m*(config['n'] - config['f'])))
    #test_dataset = torch.utils.data.Subset(test_dataset, range(test_m*(config['n'] - config['f'])))
    


    n_samples_train = len(train_dataset) # m*(config['n'] - config['f']) #
    n_samples_test =  len(test_dataset) # test_m*(config['n'] - config['f']) #
    n_samples_iid_train = int(gamma * m*(config['n'] - config['f']))
    n_samples_iid_test = int(gamma * test_m*(config['n'] - config['f']))

    #JS: Sample gamma_similarity % of the dataset, and build homogeneous dataset

    homogeneous_train_dataset, _ = torch.utils.data.random_split(train_dataset, [n_samples_iid_train, n_samples_train - n_samples_iid_train])
    homogeneous_test_dataset, _ = torch.utils.data.random_split(test_dataset, [n_samples_iid_test, n_samples_test - n_samples_iid_test])
    
    split_indices_homogeneous_train = np.array_split(homogeneous_train_dataset.indices, config['n']-config['f'])
    split_indices_homogeneous_test = np.array_split(homogeneous_test_dataset.indices, config['n']-config['f'])

    #JS: Rearrange the entire dataset by sorted labels
    labels = range(2)
    ordered_indices_train = []
    ordered_indices_test = []

    indices_dict_train={0:[], 1:[]}
    indices_dict_test={0:[], 1:[]} 

    for label in labels:
        label_indices_train = (train_dataset.targets == label).nonzero().tolist()
        label_indices_train = [item for sublist in label_indices_train for item in sublist]
        indices_dict_train[label] = label_indices_train
        label_indices_test = (test_dataset.targets == label).nonzero().tolist()
        label_indices_test = [item for sublist in label_indices_test for item in sublist]
        indices_dict_test[label] = label_indices_test


    split_indices_heterogeneous_train = []
    split_indices_heterogeneous_test = []

    for i in range(config['n']-config['f']):
        split_indices_heterogeneous_train.append([index for index in indices_dict_train[i%2] if index not in split_indices_homogeneous_train[i]][:int((1-gamma)*m)])
        split_indices_heterogeneous_test.append([index for index in indices_dict_test[i%2] if index not in split_indices_homogeneous_test[i]][:int((1-gamma)*test_m)])

    
    """
    for label in labels:
        label_indices_train = (train_dataset.targets == label).nonzero().tolist()
        label_indices_train = [item for sublist in label_indices_train for item in sublist]

        ordered_indices_train += label_indices_train

        label_indices_test = (test_dataset.targets == label).nonzero().tolist()
        label_indices_test = [item for sublist in label_indices_test for item in sublist]
        ordered_indices_test += label_indices_test
    """
    #JS: split the (sorted) heterogeneous indices equally among the honest workers
    """indices_heterogeneous_train = [index for index in ordered_indices_train if index not in homogeneous_train_dataset.indices] 
    indices_heterogeneous_train = indices_heterogeneous_train[:int((1-gamma)*(config['n'] - config['f'])*m)]
    print(len(indices_heterogeneous_train))
    split_indices_heterogeneous_train = np.array_split(indices_heterogeneous_train, config['n']-config['f'])

    indices_heterogeneous_test = [index for index in ordered_indices_test if index not in homogeneous_test_dataset.indices]
    indices_heterogeneous_test = indices_heterogeneous_test[:int((1-gamma)*(config['n'] - config['f'])*test_m)]
    split_indices_heterogeneous_test = np.array_split(indices_heterogeneous_test, config['n']-config['f'])
    """

    dataloaders_dict = {}

    for worker_id in range(config['n']-config['f']):
        homogeneous_dataset_worker_train = torch.utils.data.Subset(train_dataset, split_indices_homogeneous_train[worker_id])
        heterogeneous_dataset_worker_train = torch.utils.data.Subset(train_dataset, split_indices_heterogeneous_train[worker_id])
        concat_datasets = torch.utils.data.ConcatDataset([homogeneous_dataset_worker_train, heterogeneous_dataset_worker_train])
        client_train_loader = torch.utils.data.DataLoader(concat_datasets, batch_size=batch_size, shuffle=True)

        homogeneous_dataset_worker_test = torch.utils.data.Subset(test_dataset, split_indices_homogeneous_test[worker_id])
        heterogeneous_dataset_worker_test = torch.utils.data.Subset(test_dataset, split_indices_heterogeneous_test[worker_id])
        concat_datasets = torch.utils.data.ConcatDataset([homogeneous_dataset_worker_test, heterogeneous_dataset_worker_test])
        client_test_loader = torch.utils.data.DataLoader(concat_datasets, batch_size=batch_size, shuffle=True)


        #JS: have one dataset iterator per honest worker
        dataloaders_dict[worker_id] = [client_train_loader, client_test_loader]

    return dataloaders_dict



def mnist_label_distribution(dataloader):
    # Returns the histogram of label distribution 
    labels = np.zeros(10)
    for data in dataloader:
        _, targets = data
        for i in range(10):
            labels[i] += torch.sum(targets == i).item()
    return labels





def dirichlet_mnist_train_test(config, m, test_m = 2000, batch_size = 64, alpha = 1, seed=42, save_folder=None):
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform = transforms.ToTensor())
    n = config["n"]
    dataloaders_dict = {}
    np.random.seed(seed)
    #torch.manual_seed(seed)
    np.random.seed(0)
    probabilities = np.random.dirichlet(np.ones(10)*alpha, size = n)

    np.random.seed(seed)
    for i in trange( config['n']-config['f']):
        p = probabilities[i]
        selection_p = [p[x]  for  x in dataset.targets] 
        selection_p = selection_p/np.sum(selection_p)
        idx = np.random.choice(len(dataset), m+test_m, replace=False,  p=selection_p)
        client_dataset = torch.utils.data.Subset(dataset, idx)
        client_train_dataset, client_test_dataset = torch.utils.data.random_split(client_dataset, [m/(m+test_m), test_m / (m+test_m)])
        client_train_loader, client_test_loader = torch.utils.data.DataLoader(client_train_dataset, batch_size=batch_size, shuffle=True), torch.utils.data.DataLoader(client_test_dataset, batch_size=128, shuffle=True)
        dataloaders_dict[i] = [client_train_loader, client_test_loader]

    return dataloaders_dict

def extreme_mnist_homogenous_test(config, m, test_m = 1000, alpha = None):
    # Just for testing purposes
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform = transforms.ToTensor())
    n = config["n"]
    dataloaders_dict = {}
    for i in trange(config['n']):
        random_class = np.random.randint(0, 10)
        p = np.zeros(10)
        p[random_class] = 1
        selection_p = [p[x]  for  x in dataset.targets] 
        selection_p = selection_p/np.sum(selection_p)
        train_idx = np.random.choice(len(dataset), m, replace=False,  p=selection_p)
        test_idx = np.random.choice(len(dataset), test_m, replace=False)
        client_train_dataset = torch.utils.data.Subset(dataset, train_idx)
        client_test_dataset = torch.utils.data.Subset(dataset, test_idx)
        client_train_loader =  torch.utils.data.DataLoader(client_train_dataset, batch_size=16, shuffle=True)
        client_test_loader = torch.utils.data.DataLoader(client_test_dataset, batch_size=64, shuffle=True)
        dataloaders_dict[i] = [client_train_loader, client_test_loader]
    
    return dataloaders_dict
        
def homogenous_cifar10_train_test_(config, m, test_m = 1000, alpha = None):

    dataset = datasets.CIFAR10(root='./data/cifar10', train=True,
                                                    download=True, transform = transforms.ToTensor())
    
    dataloaders_dict = {}
    i = 0 # Only build main client test dataset with test_m datapoints
    train_idx = np.random.choice(len(dataset), m, replace=False)
    test_idx = np.random.choice(len(dataset), test_m, replace=False)
    client_train_dataset = torch.utils.data.Subset(dataset, train_idx)
    client_test_dataset = torch.utils.data.Subset(dataset, test_idx)
    client_train_loader =  torch.utils.data.DataLoader(client_train_dataset, batch_size=16, shuffle=True)
    client_test_loader = torch.utils.data.DataLoader(client_test_dataset, batch_size=64, shuffle=True)
    dataloaders_dict[i] = [client_train_loader, client_test_loader]
    test_m = 4 # Test set for other clients is not used
    for i in trange(1, config['n']):
        train_idx = np.random.choice(len(dataset), m, replace=False)
        test_idx = np.random.choice(len(dataset), test_m, replace=False)
        client_train_dataset = torch.utils.data.Subset(dataset, train_idx)
        client_test_dataset = torch.utils.data.Subset(dataset, test_idx)
        client_train_loader =  torch.utils.data.DataLoader(client_train_dataset, batch_size=16, shuffle=True)
        client_test_loader = torch.utils.data.DataLoader(client_test_dataset, batch_size=64, shuffle=True)
        dataloaders_dict[i] = [client_train_loader, client_test_loader]
    
    return dataloaders_dict

class SimpleDataset_obs(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

        
class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.toarray(), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



def dirichlet_cifar10_binary_train_test(config, m, test_m = 2000, batch_size = 16, alpha = 1, seed=42, save_folder=None):
    n = config["n"]
    #dataset = datasets.MNIST(root='./data', train=True, download=True, transform = transforms.ToTensor())
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    #subset_indices = ((dataset.targets == 8) + (dataset.targets == 9)).nonzero().view(-1) # 12665 indices for 0,1
    #subset_indices = ((dataset.targets == 8) + (dataset.targets == 9) + ((dataset.targets == 7) + (dataset.targets == 1))).nonzero().view(-1) # 12665 indices for 0,1
    dataset.targets = (dataset.targets <= 4).long()
    #dataset.targets = torch.logical_or(dataset.targets == 8, dataset.targets == 7).long()

    np.random.seed(0)
    probabilities = np.random.dirichlet(np.ones(2)*alpha, size = n)

    dataloaders_dict = {}
    for i in range( config['n']-config['f']):
        p = probabilities[i]
        selection_p = [p[x]  for  x in dataset.targets] 
        selection_p = selection_p/np.sum(selection_p)
        np.random.seed(seed)
        idx = np.random.choice(len(dataset.targets), m+test_m, replace=False, p=selection_p)
        
        client_train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=SubsetRandomSampler(idx[:m]))
        client_test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=SubsetRandomSampler(idx[m:]))
        #client_dataset = torch.utils.data.Subset(dataset, idx)
        dataloaders_dict[i] = [client_train_loader, client_test_loader]
        #print('train dist', mnist_label_distribution(client_train_loader))
        #print('test dist', mnist_label_distribution(client_test_loader))

    return dataloaders_dict


def homogenous_cifar10_train_test(config, m, test_m=1000, batch_size = 64, alpha = None, seed = 42):
    dm = torch.tensor([0.4914, 0.4822, 0.4465])
    ds = torch.tensor([0.2023, 0.1994, 0.2010])

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dm, ds)
        ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Split dataset into train and test sets
    #train_dataset, test_dataset = train_test_split(dataset, test_size=test_m, random_state=42)

    #dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

    dataloaders_dict = {}
    for i in trange(config['n']):
        idx =  np.random.choice(len(dataset), m+test_m, replace=False)
        client_dataset = torch.utils.data.Subset(dataset, idx)
        client_train_dataset, client_test_dataset = torch.utils.data.random_split(client_dataset, [m/(m+test_m), test_m / (m+test_m)])
        client_train_loader = torch.utils.data.DataLoader(client_train_dataset, batch_size=batch_size, shuffle=True)
        client_test_loader = torch.utils.data.DataLoader(client_test_dataset, batch_size=128, shuffle=True)
        dataloaders_dict[i] = [client_train_loader, client_test_loader]
        
        """train_idx = np.random.choice(len(dataset), m, replace=False)
        test_idx = np.random.choice(len(dataset), test_m, replace=False)
        client_train_dataset = torch.utils.data.Subset(dataset, train_idx)
        client_test_dataset = torch.utils.data.Subset(dataset, test_idx)
        client_train_loader = torch.utils.data.DataLoader(client_train_dataset, batch_size=batch_size, shuffle=True)
        client_test_loader = torch.utils.data.DataLoader(client_test_dataset, batch_size=128, shuffle=True)
        dataloaders_dict[i] = [client_train_loader, client_test_loader]
        """
    return dataloaders_dict



def homogenous_phishing_train_test(config, m, test_m = 2000, batch_size = 16, alpha = None, seed=42, save_folder=None):
    #m = config['m']
    #test_m = config['test_m']
    X, y = load_svmlight_file("./data/phishing.txt", n_features = 68)

    #X, y = load_svmlight_file("./data/mushrooms")
    y = ((y+1)/2).astype(int) # Convert labels to 0,1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=config['seed'])
    train_dataset = SimpleDataset(X_train, y_train)
    test_dataset = SimpleDataset(X_test, y_test)
    #assert m < len(y_train)/config['n'], "m is too big"
    dataloaders_dict = {}
    for i in trange( config['n']):
        train_idx = np.random.choice(len(train_dataset), m, replace=False)
        test_idx = np.random.choice(len(test_dataset), test_m, replace=False)
        client_train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
        client_test_dataset = torch.utils.data.Subset(test_dataset, test_idx)
        client_train_loader =  torch.utils.data.DataLoader(client_train_dataset, batch_size=batch_size, shuffle=True)
        client_test_loader = torch.utils.data.DataLoader(client_test_dataset, batch_size=64, shuffle=True)
        dataloaders_dict[i] = [client_train_loader, client_test_loader]
    
    return dataloaders_dict

def dirichlet_phishing_train_test(config, m, test_m = 2000, batch_size = 16, alpha = 1, seed=42, save_folder=None):

    #m = config['m']
    #test_m = config['test_m']
    X, y = load_svmlight_file("./data/phishing.txt", n_features = 68)

    #X, y = load_svmlight_file("./data/mushrooms")
    y = ((y+1)/2).astype(int) # Convert labels to 0,1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    train_dataset = SimpleDataset(X_train, y_train)
    test_dataset = SimpleDataset(X_test, y_test)

    #assert m < len(y_train)/config['n'], "m is too big"
    dataloaders_dict = {}
    np.random.seed(0)
    probabilities = np.random.dirichlet(np.ones(2)*alpha, size = config["n"])
    if save_folder is not None:
        logging.basicConfig(filename=os.path.join(save_folder, "log.log"), level=logging.INFO)
        logging.info(probabilities)
        
    np.random.seed(seed)
    for i in trange(config['n']-config['f']):
        p = probabilities[i]
        #print(p)
        train_selection_p = [p[0] if j==0.0 else p[1]  for  j in y_train ]
        train_selection_p = train_selection_p/np.sum(train_selection_p)
        test_selection_p = [p[0] if j==0.0 else p[1]  for  j in y_test]
        test_selection_p = test_selection_p/np.sum(test_selection_p)
        train_idx = np.random.choice(len(train_dataset), m, replace=False,  p=train_selection_p) # TODO change the seed here instead
        test_idx = np.random.choice(len(test_dataset), test_m, replace=False, p=test_selection_p)
        client_train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
        client_test_dataset = torch.utils.data.Subset(test_dataset, test_idx)
        client_train_loader =  torch.utils.data.DataLoader(client_train_dataset, batch_size=batch_size, shuffle=True)
        client_test_loader = torch.utils.data.DataLoader(client_test_dataset, batch_size=128, shuffle=True)
        dataloaders_dict[i] = [client_train_loader, client_test_loader]
    
    return dataloaders_dict

def dirichlet_phishing_train_test_extreme(config, m, test_m = 2000, batch_size = 16, alpha = 1, seed=42, save_folder=None):
    #m = config['m']
    #test_m = config['test_m']
    X, y = load_svmlight_file("./data/phishing.txt", n_features = 68)

    #X, y = load_svmlight_file("./data/mushrooms")
    y = ((y+1)/2).astype(int) # Convert labels to 0,1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    train_dataset = SimpleDataset(X_train, y_train)
    test_dataset = SimpleDataset(X_test, y_test)

    #assert m < len(y_train)/config['n'], "m is too big"
    dataloaders_dict = {}
    np.random.seed(0)
    #probabilities = np.random.dirichlet(np.ones(2)*alpha, size = config["n"])
    probabilities = np.ones((config["n"], 2))*0.5
    probabilities[0] = np.array([alpha, 1-alpha])
    for j in range(1,config["n"],2):
        probabilities[j] = np.array([1-alpha, alpha])

    np.random.seed(seed)
    for i in trange(config['n']-config['f']):
        p = probabilities[i]
        #print(p)
        train_selection_p = [p[0] if j==0.0 else p[1]  for  j in y_train ]
        train_selection_p = train_selection_p/np.sum(train_selection_p)
        test_selection_p = [p[0] if j==0.0 else p[1]  for  j in y_test]
        test_selection_p = test_selection_p/np.sum(test_selection_p)
        train_idx = np.random.choice(len(train_dataset), m, replace=False,  p=train_selection_p) # TODO change the seed here instead
        test_idx = np.random.choice(len(test_dataset), test_m, replace=False, p=test_selection_p)
        client_train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
        client_test_dataset = torch.utils.data.Subset(test_dataset, test_idx)
        client_train_loader =  torch.utils.data.DataLoader(client_train_dataset, batch_size=batch_size, shuffle=True)
        client_test_loader = torch.utils.data.DataLoader(client_test_dataset, batch_size=128, shuffle=True)
        dataloaders_dict[i] = [client_train_loader, client_test_loader]
    
    return dataloaders_dict


def gamma_phishing_train_test(config, m, test_m = 2000, batch_size = 16, alpha = 1, seed=42, save_folder=None):
    #m = config['m']
    #test_m = config['test_m']
    gamma = alpha
    X, y = load_svmlight_file("./data/phishing.txt", n_features = 68)

    #X, y = load_svmlight_file("./data/mushrooms")
    y = ((y+1)/2).astype(int) # Convert labels to 0,1
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seed) # 7728, 3317 for 0.3 // 5527, 5528 for 0.5
    
    train_dataset = SimpleDataset(X_train, y_train)
    test_dataset = SimpleDataset(X_test, y_test)



    n_samples_train = len(train_dataset) # m*(config['n'] - config['f']) #
    n_samples_test =  len(test_dataset) # test_m*(config['n'] - config['f']) #
    n_samples_iid_train = int(gamma * m*(config['n'] - config['f']))
    n_samples_iid_test = int(gamma * test_m*(config['n'] - config['f']))

    #JS: Sample gamma_similarity % of the dataset, and build homogeneous dataset

    homogeneous_train_dataset, _ = torch.utils.data.random_split(train_dataset, [n_samples_iid_train, n_samples_train - n_samples_iid_train])
    homogeneous_test_dataset, _ = torch.utils.data.random_split(test_dataset, [n_samples_iid_test, n_samples_test - n_samples_iid_test])
    
    
    split_indices_homogeneous_train = np.array_split(homogeneous_train_dataset.indices, config['n']-config['f'])
    split_indices_homogeneous_test = np.array_split(homogeneous_test_dataset.indices, config['n']-config['f'])


    labels = range(2)

    indices_dict_train={0:[], 1:[]}
    indices_dict_test={0:[], 1:[]} 

    for label in labels:
        label_indices_train = (train_dataset.y == label).nonzero().tolist()
        label_indices_train = [item for sublist in label_indices_train for item in sublist]
        indices_dict_train[label] = label_indices_train
        label_indices_test = (test_dataset.y == label).nonzero().tolist()
        label_indices_test = [item for sublist in label_indices_test for item in sublist]
        indices_dict_test[label] = label_indices_test


    split_indices_heterogeneous_train = []
    split_indices_heterogeneous_test = []

    for i in range(config['n']-config['f']):
        split_indices_heterogeneous_train.append([index for index in indices_dict_train[i%2] if index not in split_indices_homogeneous_train[i]][:int((1-gamma)*m)])
        split_indices_heterogeneous_test.append([index for index in indices_dict_test[i%2] if index not in split_indices_homogeneous_test[i]][:int((1-gamma)*test_m)])



    dataloaders_dict = {}

    for worker_id in range(config['n']-config['f']):
        homogeneous_dataset_worker_train = torch.utils.data.Subset(train_dataset, split_indices_homogeneous_train[worker_id])
        heterogeneous_dataset_worker_train = torch.utils.data.Subset(train_dataset, split_indices_heterogeneous_train[worker_id])
        concat_datasets = torch.utils.data.ConcatDataset([homogeneous_dataset_worker_train, heterogeneous_dataset_worker_train])
        client_train_loader = torch.utils.data.DataLoader(concat_datasets, batch_size=batch_size, shuffle=True)

        homogeneous_dataset_worker_test = torch.utils.data.Subset(test_dataset, split_indices_homogeneous_test[worker_id])
        heterogeneous_dataset_worker_test = torch.utils.data.Subset(test_dataset, split_indices_heterogeneous_test[worker_id])
        concat_datasets = torch.utils.data.ConcatDataset([homogeneous_dataset_worker_test, heterogeneous_dataset_worker_test])
        client_test_loader = torch.utils.data.DataLoader(concat_datasets, batch_size=batch_size, shuffle=True)


        #JS: have one dataset iterator per honest worker
        dataloaders_dict[worker_id] = [client_train_loader, client_test_loader]
    
    return dataloaders_dict



def dirichlet_phishing_train_test_old(config, m, test_m = 2000, batch_size = 16, alpha = 1, seed=42, save_folder=None):
    #m = config['m']
    #test_m = config['test_m']
    X, y = load_svmlight_file("./data/phishing.txt", n_features = 68)

    #X, y = load_svmlight_file("./data/mushrooms")
    y = ((y+1)/2).astype(int) # Convert labels to 0,1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=config['seed'])
    train_dataset = SimpleDataset(X_train, y_train)
    test_dataset = SimpleDataset(X_test, y_test)

    #assert m < len(y_train)/config['n'], "m is too big"
    dataloaders_dict = {}
    np.random.seed(config["seed"])
    probabilities = np.random.dirichlet(np.ones(2)*alpha, size = config["n"])

    for i in trange(config['n']-config['f']):
        p = probabilities[i]
        train_selection_p = [p[0] if j==0.0 else p[1]  for  j in y_train ]
        train_selection_p = train_selection_p/np.sum(train_selection_p)
        test_selection_p = [p[0] if j==0.0 else p[1]  for  j in y_test]
        test_selection_p = test_selection_p/np.sum(test_selection_p)
        train_idx = np.random.choice(len(train_dataset), m, replace=False,  p=train_selection_p)
        test_idx = np.random.choice(len(test_dataset), test_m, replace=False, p=test_selection_p)
        client_train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
        client_test_dataset = torch.utils.data.Subset(test_dataset, test_idx)
        client_train_loader =  torch.utils.data.DataLoader(client_train_dataset, batch_size=batch_size, shuffle=True)
        client_test_loader = torch.utils.data.DataLoader(client_test_dataset, batch_size=128, shuffle=True)
        dataloaders_dict[i] = [client_train_loader, client_test_loader]
    
    return dataloaders_dict




heterogneity = { "homogeneous" : homogenous_mnist_train_test, "dirichlet_mnist" : dirichlet_mnist_train_test,
                 "extreme_v2" : extreme_mnist_homogenous_test , "homogeneous_phishing" : homogenous_phishing_train_test,
                 "dirichlet_phishing" : dirichlet_phishing_train_test, "homogeneous_cifar10" : homogenous_cifar10_train_test,
                 "homogeneous_mnist_bin" : homogenous_mnist_binary_train_test, "dirichlet_mnist_bin" : dirichlet_mnist_binary_train_test, 
                 "gamma_mnist_bin" : gamma_mnist_binary_train_test, "dirichlet_mnist_bin_correct_seeding" : dirichlet_mnist_binary_train_test_correct_seeding,
                 "gamma_phishing_train_test" : gamma_phishing_train_test, "dirichlet_phishing_extreme": dirichlet_phishing_train_test_extreme}


if __name__ == '__main__':
    config = {'n': 20, 'm': 64, 'f':0 }
    dls  = gamma_mnist_binary_train_test(config, m = 100, test_m = 200, batch_size = 16, gamma = 0.2, seed=42, save_folder=None)

    for i in range(3):
        print(mnist_label_distribution(dls[i][0]))
        print(mnist_label_distribution(dls[i][1]))
