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

# This is the first dataset, which draws a proxy dataset from the 
# original MNIST dataset.


# def data_partition(MNIST_training_dataset):

#     dataset_dict = {i: [] for i in range(10)}
#     training_dataset_dict = {i: [] for i in range(10)}
#     global_dataset_dict = {i: [] for i in range(10)}



#     for i in range(MNIST_training_dataset.__len__()):
#         #print(self.dataset.__getitem__(i)[1])
#         sample = MNIST_training_dataset.__getitem__(i)
#         dataset_dict[sample[1]].append(sample)

#     # shuffle 

#     for i in range(10):
#         num = len(dataset_dict[i])
#         random_index = np.random.choice(num, size = num, replace = False)
#         print(random_index, len(random_index))
#         dataset_dict[i] = np.array(dataset_dict[i])[random_index]

#         training_dataset_dict[i] = dataset_dict[i][:int(num * 0.9)]
#         global_dataset_dict[i] = dataset_dict[i][int(num * 0.9):]

#     return training_dataset_dict, global_dataset_dict



class MNIST4KuLSIFandFedMD10Clients1Class(Dataset):

    def __init__(self, proxy_dataset = True, client_index = None, dataset_path = "./MNISTwithProxyDataset.pth", transform = None):

        # The transform has already been done.

        if proxy_dataset == True and client_index != None:
            raise Exception("client_index shoud be None when the proxy_dataset is True")

        dicts =  torch.load(dataset_path)

        #label_list = []

        self.client_index = client_index

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
        data, label = self.dataset[idx] #, self.client_index
        return data, torch.tensor(label)


# class MNIST4FedMD10Clients1ClasswithSelectedProxyDataset(Dataset):

#     def __init__(self, 
#                  client_index = 0, 
#                  dataset_path = "./MNISTwithProxyDataset.pth", 
#                  proxy_dataset_path = "../MNIST_KuLSIF_estimator/MNIST_1class_selected_proxy_samples_by_mean.pth",
#                  transform = None):

#         # The transform has already been done.

#         dicts =  torch.load(dataset_path)

#         dataset = dicts["training_dataset_dict"][client_index]

#         #print(dataset1[1:3])

#         input_list = []
#         label_list = []

#         for i in range(len(dataset)):
#             input_sample, label = dataset[i]
#             input_list.append(torch.reshape(input_sample, (1,28,28)))
#             label_list.append(label)

#         #print(len(input_list))
#         #print(input_list[0].size())

#         proxy_dataset_dict = torch.load(proxy_dataset_path)

#         for k in range(10):

#             for class_index in range(10):
#                 #print(proxy_dataset_dict[class_index].shape[0])
#                 for j in range(proxy_dataset_dict[class_index].shape[0]):
#                     input_list.append(torch.reshape(torch.tensor(proxy_dataset_dict[class_index][j]), (1,28,28)))
#                     label_list.append(class_index)

#         self.input_list = input_list
#         self.label_list = label_list




#     def __len__(self):
#         return len(self.input_list)


#     def __getitem__(self, idx):
#         return self.input_list[idx], self.label_list[idx]






class MNIST4KuLSIFandFedMD10Clients2Class(Dataset):

    def __init__(self, proxy_dataset = True, client_index = None, dataset_path = "./MNISTwithProxyDataset.pth", transform = None):

        # The transform has already been done.

        if proxy_dataset == True and client_index != None:
            raise Exception("client_index shoud be None when the proxy_dataset is True")

        dicts =  torch.load(dataset_path)

        if proxy_dataset:
            client_index = None
            proxy_dataset = dicts["proxy_dataset_dict"]
            self.dataset = []
            for i in range(10):
                for j in range(len(proxy_dataset[i])):
                    self.dataset.append(proxy_dataset[i][j])
        else:

            num1 = len(dicts["training_dataset_dict"][client_index])
            dataset1 = dicts["training_dataset_dict"][client_index][int(num1 * 0.5):] 

            self.dataset1 = dataset1

            num2 = len(dicts["training_dataset_dict"][(client_index + 1) %10])
            dataset2 = dicts["training_dataset_dict"][(client_index + 1) %10][:int(num2 * 0.5)]

            self.dataset2 = dataset2

            self.dataset = np.append(dataset1,dataset2, axis = 0)
            np.random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        return self.dataset[idx]


if __name__ == '__main__':


    #dataset = MNIST4KuLSIFandFedMD10Clients1ClasswithProxyDataset()
    dataset = MNIST4KuLSIFandFedMD10Clients1Class(proxy_dataset = False, client_index = 5)
    print(dataset.__len__())
    print(dataset.__getitem__(-1))
    data , _ = dataset.__getitem__(-1)
    print(data.size())

    #dataset = MNIST4FedMD2Clients(global_dataset = False, client_index = 0)
    # dataset = MNIST4KuLSIFandFedMD10Clients2Class(proxy_dataset = False, client_index = 9)
    # dataset = MNIST4KuLSIFandFedMD10Clients2Class(proxy_dataset = True, client_index = None)
    # #dataset = MNIST4FedMD5Clients(global_dataset = False, client_index = 1)
    # print(dataset.__len__(),dataset.__getitem__(3)[0].size())
    # print(torch.max(dataset.__getitem__(3)[0]),torch.min(dataset.__getitem__(3)[0]))
    # print(dataset.__getitem__(3)[1],dataset.__getitem__(5000)[1])





