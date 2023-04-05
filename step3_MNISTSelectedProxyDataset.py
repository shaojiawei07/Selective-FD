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
from torchvision import datasets, transforms

# selected_samples_voting_dict can be regarded as the soft label


# # Allow multi-target label
# class MNIST10Clients1ClassSelectedProxyDataset_hardlabel(Dataset):

#     def __init__(self, 
#                  client_index = 0,
#                  dataset_path = "./MNIST_KuLSIF_estimator/MNIST1classSelectedProxy_voting_vector.pth",
#                  transform = None):

#         dicts =  torch.load(dataset_path)
#         selected_samples_dict = dicts["selected_samples_dict"]
#         selected_samples_voting_dict = dicts["selected_samples_voting_dict"]

#         input_list = []
#         label_list = []
#         ensemble_label_list = []

#         for class_index in range(10):
#             for j in range(selected_samples_dict[class_index].shape[0]):
#                 input_list.append(torch.reshape(torch.tensor(selected_samples_dict[class_index][j]), (1,28,28)))
#                 label_list.append(class_index)
#                 ensemble_label_list.append(selected_samples_voting_dict[class_index][j])

#         self.input_list = input_list
#         self.label_list = label_list
#         self.ensemble_label_list = torch.tensor(ensemble_label_list)


#     def __len__(self):
#         return len(self.input_list)


#     def __getitem__(self, idx):
#         return self.input_list[idx], self.ensemble_label_list[idx] #, self.label_list[idx]


class MNIST10Clients1ClassSelectedProxyDataset_hardlabel(Dataset):

    def __init__(self, 
                 client_index = 0,
                 dataset_path = "./MNIST_KuLSIF_estimator/MNIST1classSelectedProxyQuartile.pth",
                 transform = None):

        dicts =  torch.load(dataset_path)
        selected_samples_dict = dicts["selected_samples_dict"]
        selected_samples_voting_dict = dicts["selected_samples_voting_dict"]

        input_list = []
        label_list = []
        ensemble_label_list = []

        for class_index in range(10):
            for j in range(selected_samples_dict[class_index].shape[0]):

                max_index = np.argwhere(selected_samples_voting_dict[class_index][j] == np.amax(selected_samples_voting_dict[class_index][j]))
                if max_index.size != 1:
                    continue

                voting_vector = np.zeros_like(selected_samples_voting_dict[class_index][j])
                voting_vector[max_index] = 1

                input_list.append(torch.reshape(torch.tensor(selected_samples_dict[class_index][j]), (1,28,28)))
                label_list.append(class_index)
                ensemble_label_list.append(voting_vector)

        self.input_list = input_list
        self.label_list = label_list
        self.ensemble_label_list = torch.tensor(np.array(ensemble_label_list))


    def __len__(self):
        return len(self.input_list)


    def __getitem__(self, idx):
        return self.input_list[idx], self.ensemble_label_list[idx] #, self.label_list[idx]


class MNIST10Clients1ClassSelectedProxyDataset_softlabel(Dataset):

    def __init__(self, 
                 client_index = 0,
                 dataset_path = "./MNIST_KuLSIF_estimator/MNIST1classSelectedProxyQuartile.pth",
                 transform = None):

        dicts =  torch.load(dataset_path)
        selected_samples_dict = dicts["selected_samples_dict"]
        selected_samples_voting_dict = dicts["selected_samples_voting_dict"]

        input_list = []
        label_list = []
        ensemble_label_list = []

        for class_index in range(10):
            for j in range(selected_samples_dict[class_index].shape[0]):

                input_list.append(torch.reshape(torch.tensor(selected_samples_dict[class_index][j]), (1,28,28)))
                label_list.append(class_index)
                ensemble_label_list.append(selected_samples_voting_dict[class_index][j])

        self.input_list = input_list
        self.label_list = label_list
        self.ensemble_label_list = torch.tensor(np.array(ensemble_label_list))


    def __len__(self):
        return len(self.input_list)


    def __getitem__(self, idx):
        return self.input_list[idx], self.ensemble_label_list[idx] #, self.label_list[idx]


class MNIST10Clients2ClassSelectedProxyDataset_hardlabel(Dataset):

    def __init__(self, 
                 client_index = 0,
                 dataset_path = "./MNIST_KuLSIF_estimator/MNIST2classSelectedProxyQuartile.pth",
                 transform = None):

        dicts =  torch.load(dataset_path)
        selected_samples_dict = dicts["selected_samples_dict"]
        selected_samples_voting_dict = dicts["selected_samples_voting_dict"]

        input_list = []
        label_list = []
        ensemble_label_list = []

        for class_index in range(10):
            for j in range(selected_samples_dict[class_index].shape[0]):

                max_index = np.argwhere(selected_samples_voting_dict[class_index][j] == np.amax(selected_samples_voting_dict[class_index][j]))
                if max_index.size != 1:
                    continue

                voting_vector = np.zeros_like(selected_samples_voting_dict[class_index][j])
                voting_vector[max_index] = 1

                input_list.append(torch.reshape(torch.tensor(selected_samples_dict[class_index][j]), (1,28,28)))
                label_list.append(class_index)
                ensemble_label_list.append(voting_vector)

        self.input_list = input_list
        self.label_list = label_list
        self.ensemble_label_list = torch.tensor(np.array(ensemble_label_list))


    def __len__(self):
        return len(self.input_list)


    def __getitem__(self, idx):
        return self.input_list[idx], self.ensemble_label_list[idx] #, self.label_list[idx]


class MNIST10Clients2ClassSelectedProxyDataset_softlabel(Dataset):

    def __init__(self, 
                 client_index = 0,
                 dataset_path = "./MNIST_KuLSIF_estimator/MNIST2classSelectedProxyQuartile.pth",
                 transform = None):

        dicts =  torch.load(dataset_path)
        selected_samples_dict = dicts["selected_samples_dict"]
        selected_samples_voting_dict = dicts["selected_samples_voting_dict"]

        input_list = []
        label_list = []
        ensemble_label_list = []

        for class_index in range(10):
            for j in range(selected_samples_dict[class_index].shape[0]):

                input_list.append(torch.reshape(torch.tensor(selected_samples_dict[class_index][j]), (1,28,28)))
                label_list.append(class_index)
                ensemble_label_list.append(selected_samples_voting_dict[class_index][j])

        self.input_list = input_list
        self.label_list = label_list
        self.ensemble_label_list = torch.tensor(np.array(ensemble_label_list))


    def __len__(self):
        return len(self.input_list)


    def __getitem__(self, idx):
        return self.input_list[idx], self.ensemble_label_list[idx]


if __name__ == '__main__':


    dataset = MNIST10Clients1ClassSelectedProxyDataset_softlabel()
    print(dataset.__len__())
    # print(dataset.__getitem__(-1))
    # data , _ = dataset.__getitem__(-1)
    # print(data.size())

