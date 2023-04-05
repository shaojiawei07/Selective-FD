from utils.MNIST4KuLSIF import MNIST4KuLSIFandFedMD10Clients1Class, MNIST4KuLSIFandFedMD10Clients2Class
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



def bulid_estimator_1class_per_client(client_index, Gaussian_kernel_width = 5):

    known_dataset = MNIST4KuLSIFandFedMD10Clients1Class(proxy_dataset = False, 
                                                        client_index = client_index, 
                                                        dataset_path = "./utils/MNISTwithProxyDataset.pth")

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


def bulid_estimator_2class_per_client(client_index, Gaussian_kernel_width = 5):


    inlier_dataset = MNIST4KuLSIFandFedMD10Clients2Class(proxy_dataset = False,
                                        client_index = client_index,
                                        dataset_path = "./utils/MNISTwithProxyDataset.pth")

    dataset1 = inlier_dataset.dataset1
    dataset2 = inlier_dataset.dataset2

    train_dataset1 = dataset1[:int(len(dataset1) * 0.1)]
    train_dataset2 = dataset2[:int(len(dataset2) * 0.1)]

    # train_dataset = np.append(inlier_set1[:int(len(inlier_set1) * 0.05)],
    #                           inlier_set2[:int(len(inlier_set2) * 0.05)], 
    #                           axis = 0)

    train_dataset1 = known_samples_collection(train_dataset1)
    train_dataset2 = known_samples_collection(train_dataset2)

    eval_dataset1 = dataset1[int(len(dataset1) * 0.1):]
    eval_dataset2 = dataset2[int(len(dataset2) * 0.1):]
    eval_dataset1 = known_samples_collection(eval_dataset1)
    eval_dataset2 = known_samples_collection(eval_dataset2)

    # 1st estimator
    KuLSIF_estimator1 = KuLSIF_density_ratio_estimation(Gaussian_kernel_width = 5, 
                                                       known_samples = train_dataset1, 
                                                       auxiliary_samples = uniform_samples_collection(sample_num = 250), 
                                                       lamda = 250 ** (-0.5)) # kernel with = 4 or 5 is ok. 20 and 30 are too large.
    ratio_eval1 = KuLSIF_estimator1.ratio_estimator(eval_dataset1)
    eval_mean1, eval_median1, eval_std1 = np.mean(ratio_eval1), np.median(ratio_eval1), np.std(ratio_eval1)
    eval_first_quartile1 = np.quantile(ratio_eval1, q = 0.25)

    KuLSIF_estimator1.eval_mean = eval_mean1
    KuLSIF_estimator1.eval_median = eval_median1
    KuLSIF_estimator1.eval_std = eval_std1
    KuLSIF_estimator1.eval_first_quartile = eval_first_quartile1


    # 2nd estimator
    KuLSIF_estimator2 = KuLSIF_density_ratio_estimation(Gaussian_kernel_width = 5, 
                                                       known_samples = train_dataset2, 
                                                       auxiliary_samples = uniform_samples_collection(sample_num = 250), 
                                                       lamda = 250 ** (-0.5)) # kernel with = 4 or 5 is ok. 20 and 30 are too large.
    ratio_eval2 = KuLSIF_estimator2.ratio_estimator(eval_dataset2)
    eval_mean2, eval_median2, eval_std2 = np.mean(ratio_eval2), np.median(ratio_eval2), np.std(ratio_eval2)
    eval_first_quartile2 = np.quantile(ratio_eval2, q = 0.25)

    KuLSIF_estimator2.eval_mean = eval_mean2
    KuLSIF_estimator2.eval_median = eval_median2
    KuLSIF_estimator2.eval_std = eval_std2
    KuLSIF_estimator2.eval_first_quartile = eval_first_quartile2

    return KuLSIF_estimator1, KuLSIF_estimator2




if __name__ == '__main__':

    np.random.seed(0)

    density_ratio_estimator_dict1 = {}
    density_ratio_estimator_dict2 = {}

    # torch.save(density_ratio_estimator_dict1, "./MNIST_KuLSIF_estimator/MNISTDStrongNonIIDEsti.pth")
    # torch.save(density_ratio_estimator_dict2, "./MNIST_KuLSIF_estimator/MNISTDWeakNonIIDEsti.pth")

    # the client number is 10
    for i in range(10):
        density_ratio_estimator_dict1[i] = bulid_estimator_1class_per_client(client_index = i, Gaussian_kernel_width = 5)
        KuLSIF_estimator1, KuLSIF_estimator2 = bulid_estimator_2class_per_client(client_index = i, Gaussian_kernel_width = 5)
        density_ratio_estimator_dict2[i] = {"KuLSIF_estimator1":KuLSIF_estimator1,"KuLSIF_estimator2":KuLSIF_estimator2}



    torch.save(density_ratio_estimator_dict1, "./MNIST_KuLSIF_estimator/MNISTDStrongNonIIDEsti.pth")
    torch.save(density_ratio_estimator_dict2, "./MNIST_KuLSIF_estimator/MNISTDWeakNonIIDEsti.pth")



