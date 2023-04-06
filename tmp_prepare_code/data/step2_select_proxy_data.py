#from utils.MNIST4KuLSIF import MNIST4KuLSIFandFedMD10Clients1Class, MNIST4KuLSIFandFedMD10Clients2Class
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
from KuLSIF import KuLSIF_density_ratio_estimation, KuLSIF_estimator
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

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


def MNIST1ClassSelectProxySamples(density_ratio_estimator_dict):

    #density_ratio_estimator_dict = torch.load("./MNIST_KuLSIF_estimator/MNISTDStrongNonIIDEsti.pth")

    proxy_dataset = MNIST4KuLSIFandFedMD10Clients1Class(proxy_dataset = True, dataset_path = "./utils/MNISTwithProxyDataset.pth")
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

    
    return selected_samples_dict, selected_samples_voting_dict #{"selected_samples_dict":selected_samples_dict,"selected_samples_voting_dict":selected_samples_voting_dict}

    # torch.save({"selected_samples_dict":selected_samples_dict,"selected_samples_voting_dict":selected_samples_voting_dict},
    #            "MNIST_KuLSIF_estimator/MNIST1classSelectedProxyQuartile.pth")


def MNIST2ClassSelectProxySamples():

    density_ratio_estimator_dict = torch.load("./MNIST_KuLSIF_estimator/MNISTDWeakNonIIDEsti.pth")

    #density_ratio_estimator_dict[client_index]["KuLSIF_estimator1"]

    proxy_dataset = MNIST4KuLSIFandFedMD10Clients1Class(proxy_dataset = True, dataset_path = "./utils/MNISTwithProxyDataset.pth")
    test_data_dict = test_samples_collection(proxy_dataset)

    #binary_classification_result_dict = {}

    selected_samples_dict = {}
    selected_samples_voting_dict = {}

    for j in range(10):

        class_index = j

        print(class_index)

        sample_num = len(test_data_dict[class_index])

        #print(len(test_data_dict[class_index]))

        aggregation = np.zeros((sample_num, 10))

        for i in range(10):


            KuLSIF_estimator1 = density_ratio_estimator_dict[i]["KuLSIF_estimator1"]
            KuLSIF_estimator2 = density_ratio_estimator_dict[i]["KuLSIF_estimator2"]

            estimated_density_ratio1 = KuLSIF_estimator1.ratio_estimator(test_data_dict[class_index])
            binary_classification_result1 = estimated_density_ratio1 > KuLSIF_estimator1.eval_first_quartile #.eval_median #+ density_ratio_estimator_dict[i].eval_std
            aggregation[:,i] += binary_classification_result1

            estimated_density_ratio2 = KuLSIF_estimator2.ratio_estimator(test_data_dict[class_index])
            binary_classification_result2 = estimated_density_ratio2 > KuLSIF_estimator2.eval_first_quartile #.eval_median #+ density_ratio_estimator_dict[i].eval_std
            aggregation[:,(i+1)%10] += binary_classification_result2



        aggregation_sum = np.sum(aggregation, axis = 1)

        #print(aggregation_sum)

        selected_list = np.nonzero(aggregation_sum)

        selected_samples_dict[class_index] = test_data_dict[class_index][selected_list]

        selected_samples_voting_dict[class_index] =  aggregation[selected_list] / np.reshape(aggregation_sum[selected_list], (-1,1))

    torch.save({"selected_samples_dict":selected_samples_dict,"selected_samples_voting_dict":selected_samples_voting_dict},
               "MNIST_KuLSIF_estimator/MNIST2classSelectedProxyQuartile.pth")







if __name__ == '__main__':

    # torch.save({"selected_samples_dict":1},
    #            "./MNIST_KuLSIF_estimator/MNIST2classSelectedProxyMedian.pth")

    MNIST1ClassSelectProxySamples()
    MNIST2ClassSelectProxySamples()

    ...

