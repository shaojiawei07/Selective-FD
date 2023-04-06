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
from KuLSIF import KuLSIF_density_ratio_estimation, KuLSIF_estimator
import sklearn.metrics as metrics
import os



class MNISTLocalDataset(Dataset):

    def __init__(self, client_index = None, dataset_path = "./MNISTTrainDataset.pth", transform = None):

        dicts =  torch.load(dataset_path)

        self.dataset = dicts["training_dataset_dict"][client_index]

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        return self.dataset[idx]



class MNISTProxyDataset(Dataset):

    def __init__(self, dataset_path = "./MNISTTrainDataset.pth"):

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



class MNISTSelectedProxyDataset(Dataset):

    def __init__(self, dataset_path = "./MNISTTrainDataset.pth",
                 selected_dataset_path = "./MNISTSelectedProxyDataset.pth"): # dataset_path = "./MNIST_KuLSIF_estimator/MNIST1classSelectedProxyQuartile.pth"


        #torch.save({},"./MNISTSelectedProxyDataset.pth")

        if os.path.exists(selected_dataset_path):
            print("\n The SelectedProxyDataset exists \n")
            ...
        else:
            print("\n Prepare the SelectedProxyDataset \n")
            construct_SelectedProxyDataset(dataset_path, selected_dataset_path)

        dicts =  torch.load(selected_dataset_path)
        selected_samples_dict = dicts["selected_samples_dict"]
        selected_samples_voting_dict = dicts["selected_samples_voting_dict"]

        input_list = []
        #label_list = []
        ensemble_label_list = []

        for class_index in range(10):
            for j in range(selected_samples_dict[class_index].shape[0]):

                input_list.append(torch.reshape(torch.tensor(selected_samples_dict[class_index][j]), (1,28,28)))
                #label_list.append(class_index)
                ensemble_label_list.append(selected_samples_voting_dict[class_index][j])

        self.input_list = input_list
        #self.label_list = label_list
        self.ensemble_label_list = torch.tensor(np.array(ensemble_label_list))


    def __len__(self):
        return len(self.input_list)


    def __getitem__(self, idx):
        return self.input_list[idx], self.ensemble_label_list[idx]



def construct_SelectedProxyDataset(dataset_path, selected_dataset_path):

    density_ratio_estimator_dict1 = {}

    for i in range(10):
        density_ratio_estimator_dict1[i] = bulid_estimator_1class_per_client(client_index = i, 
                                                                             Gaussian_kernel_width = 5, 
                                                                             dataset_path = dataset_path)

    selected_samples_dict, selected_samples_voting_dict = MNIST1ClassSelectProxySamples(density_ratio_estimator_dict1, 
                                                                                        dataset_path = dataset_path)

    torch.save({"selected_samples_dict":selected_samples_dict,"selected_samples_voting_dict":selected_samples_voting_dict},
                selected_dataset_path)

    #print("path exists?", os.path.exists("./MNISTSelectedProxyDataset.pth"))


def bulid_estimator_1class_per_client(client_index, Gaussian_kernel_width = 5, dataset_path = "./MNISTTrainDataset.pth"):

    known_dataset = MNISTLocalDataset(client_index = client_index, dataset_path = dataset_path)

    known_dataset = known_dataset.dataset
    train_dataset = known_dataset[:int(len(known_dataset) * 0.05)]
    train_dataset = known_samples_collection(train_dataset)
    KuLSIF_estimator = KuLSIF_density_ratio_estimation(Gaussian_kernel_width = 5, 
                                                       known_samples = train_dataset, 
                                                       auxiliary_samples = uniform_samples_collection(sample_num = 250), 
                                                       lamda = 250 ** (-0.5)) # kernel with = 4 or 5 is ok. 20 and 30 are too large.


    eval_dataset = known_dataset[int(len(known_dataset) * 0.05):]
    eval_dataset = known_samples_collection(eval_dataset)
    ratio_eval = KuLSIF_estimator.ratio_estimator(eval_dataset)

    eval_mean, eval_median, eval_std = np.mean(ratio_eval), np.median(ratio_eval), np.std(ratio_eval)
    eval_first_quartile = np.quantile(ratio_eval, q = 0.25)

    KuLSIF_estimator.eval_mean = eval_mean
    KuLSIF_estimator.eval_median = eval_median
    KuLSIF_estimator.eval_std = eval_std
    KuLSIF_estimator.eval_first_quartile = eval_first_quartile

    return KuLSIF_estimator


def MNIST1ClassSelectProxySamples(density_ratio_estimator_dict, dataset_path):

    #density_ratio_estimator_dict = torch.load("./MNIST_KuLSIF_estimator/MNISTDStrongNonIIDEsti.pth")

    proxy_dataset = MNISTProxyDataset(dataset_path = dataset_path)
    test_data_dict = test_samples_collection(proxy_dataset)

    binary_classification_result_dict = {}

    selected_samples_dict = {}
    selected_samples_voting_dict = {}

    for j in range(10):

        class_index = j

        print(class_index)

        for i in range(10):

            estimated_density_ratio = density_ratio_estimator_dict[i].ratio_estimator(test_data_dict[class_index])

            binary_classification_result_dict[i] = estimated_density_ratio > density_ratio_estimator_dict[i].eval_first_quartile #.eval_median 


        aggregation = [binary_result for binary_result in binary_classification_result_dict.values()]
        aggregation = np.array(aggregation)
        #print(aggregation)
        #print(aggregation.shape) # (10, 593(sample num))
        aggregation = np.swapaxes(aggregation, axis1 = 0, axis2 = 1) # (593, 10)

        aggregation_sum = np.sum(aggregation, axis = 1)

        selected_list = np.nonzero(aggregation_sum)

        selected_samples_dict[class_index] = test_data_dict[class_index][selected_list]

        selected_samples_voting_dict[class_index] =  aggregation[selected_list] / np.reshape(aggregation_sum[selected_list], (-1,1))

    
    return selected_samples_dict, selected_samples_voting_dict



def test_samples_collection(dataset, label_num = 10):
    sample_dict = {i:[] for i in range(label_num)}

    for i in range(len(dataset.dataset)):
        image, label = dataset.dataset[i]
        image_vector = torch.reshape(image,(-1,))
        sample_dict[label].append(image_vector.numpy())

    for i in range(label_num):
        sample_dict[i] = np.array(sample_dict[i])

    return sample_dict


def known_samples_collection(dataset):

    # reshape the given inputs to vectors
    # convert [tensor] to array

    local_training_dataset_list = []

    for i in range(len(dataset)):
        image, label = dataset[i]
        image_vector = torch.reshape(image,(-1,))
        local_training_dataset_list.append(image_vector.numpy())

    local_training_samples = np.array(local_training_dataset_list)
    print("local training dataset size:",local_training_samples.shape)

    return local_training_samples


def uniform_samples_collection(max_value = 2.82157,
                               min_value = -0.4242,
                               dim = 784,
                               sample_num = 1000):

    auxiliary_samples = np.random.uniform(low = min_value, high = max_value, size = (sample_num, dim))

    #print("auxiliary dataset:",auxiliary_samples.shape)

    return auxiliary_samples







if __name__ == '__main__':

    ...

    #dataset = MNISTSelectedProxyDataset() #MNIST4FedMD10Clients(global_dataset = True, client_index = 3)
    #print(dataset.__len__(),dataset.__getitem__(3)[0].size())




