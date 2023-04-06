#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import numpy as np


def data_partition(MNIST_training_dataset):

    dataset_dict = {i: [] for i in range(10)}
    training_dataset_dict = {i: [] for i in range(10)}
    global_dataset_dict = {i: [] for i in range(10)}

    for i in range(MNIST_training_dataset.__len__()):
        #print(self.dataset.__getitem__(i)[1])
        sample = MNIST_training_dataset.__getitem__(i)
        dataset_dict[sample[1]].append(sample)

    # shuffle 

    for i in range(10):
        num = len(dataset_dict[i])
        random_index = np.random.choice(num, size = num, replace = False)
        print(random_index, len(random_index))
        dataset_dict[i] = np.array(dataset_dict[i])[random_index]

        training_dataset_dict[i] = dataset_dict[i][:int(num * 0.9)]
        global_dataset_dict[i] = dataset_dict[i][int(num * 0.9):]

    return training_dataset_dict, global_dataset_dict



class MNIST4FedMD10Clients(Dataset):

    def __init__(self, proxy_dataset = True, client_index = None, dataset_path = "./MNISTwithProxyDataset.pth", transform = None):

        # The transform has already been done.

        if proxy_dataset == True and client_index != None:
            raise Exception("client_index shoud be None when the proxy_dataset is True")

        dicts =  torch.load(dataset_path)

        if proxy_dataset:
            client_index = None
            proxy_dataset = dicts["proxy_dataset_dict"]
            self.dataset = []
            #self.dataset = [global_dataset[i] for i in range(len(global_dataset))]
            for i in range(10):
                for j in range(len(proxy_dataset[i])):
                    self.dataset.append(proxy_dataset[i][j])
        else:
            #num = len(dicts["training_dataset_dict"][client_index])
            self.dataset = dicts["training_dataset_dict"][client_index]

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        return self.dataset[idx]



class MNISTProxyDataset(Dataset):

    def __init__(self, dataset_path = "./MNISTwithProxyDataset.pth"):

        dicts =  torch.load(dataset_path)
        proxy_dataset = dicts["proxy_dataset_dict"]
        self.dataset = []
        for i in range(10):
            for j in range(len(proxy_dataset[i])):
                self.dataset.append(proxy_dataset[i][j])

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        return self.dataset[idx]



class MNISTLocalDataset(Dataset):

    def __init__(self, client_index = None, dataset_path = "./MNISTwithProxyDataset.pth", transform = None):

        dicts =  torch.load(dataset_path)

        self.dataset = dicts["training_dataset_dict"][client_index]

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        return self.dataset[idx]






if __name__ == '__main__':
    # transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    #     ])

    dataset = MNIST4FedMD10Clients(global_dataset = True, client_index = 3)
    print(dataset.__len__(),dataset.__getitem__(3)[0].size())

    # sumnum = 0
    # for i in range(10):
    #     num = len(training_dataset_dict[i])
    #     print(len(training_dataset_dict[i]))
    #     #print(len(data_dict[4]))
    #     sumnum += num
    # print("sumnum",sumnum)

    # sumnum = 0
    # for i in range(10):
    #     num = len(global_dataset_dict[i])
    #     print(len(global_dataset_dict[i]))
    #     #print(len(data_dict[4]))
    #     sumnum += num
    # print("sumnum",sumnum)



