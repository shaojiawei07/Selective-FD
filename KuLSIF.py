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

    

class KuLSIF_density_ratio_estimation:

    # estimate the density r(x) = p(x) / q(x)
    # p(x) is the known distribution (known samples)
    # q(x) is the auxiliary distribution
    # Particularly, supp Q should contain supp P.
    # Here, we assume X is in a compact set/space,
    # and q(x) could be a uniform distribution in 
    # this space.

    def __init__(self, 
                 Gaussian_kernel_width,
                 known_samples,
                 auxiliary_samples,
                 lamda):

        self.n = auxiliary_samples.shape[0] # number of auxiliary_samples
        self.m = known_samples.shape[0] # number of known samples
        self.dim = auxiliary_samples.shape[1]
        self.Gaussian_kernel_width = Gaussian_kernel_width
        self.known_samples = known_samples
        self.auxiliary_samples = auxiliary_samples
        self.lamda = lamda
        self.K11 = self._compute_K11_matrix()
        self.K12 = self._compute_K12_matrix()
        self.alpha_vector = self._compute_alpha_vector()
        self.K11 = None 
        self.K12 = None

    def _compute_K11_matrix(self): 
        #return a squared matrix with size aux_num * aux_num

        Gaussian_kernel_width = self.Gaussian_kernel_width
        auxiliary_samples = self.auxiliary_samples

        auxiliary_samples1 = np.expand_dims(auxiliary_samples, axis = 0) # 1 * aux_num * dim
        auxiliary_samples2 = np.expand_dims(auxiliary_samples, axis = 1) # aux_num * 1 * dim

        distance_matrix = auxiliary_samples1 - auxiliary_samples2
        distance_matrix = np.linalg.norm(distance_matrix, ord = 2, axis = 2)
        K11 = np.exp(-distance_matrix ** 2 / Gaussian_kernel_width ** 2 / 2)

        return K11

    def _compute_K12_matrix(self): 
        #return a matrix with size aux_num * sample_num

        Gaussian_kernel_width = self.Gaussian_kernel_width
        known_samples = self.known_samples
        auxiliary_samples = self.auxiliary_samples

        known_samples = np.expand_dims(known_samples, axis = 0) # 1 * sample_num * dim
        auxiliary_samples = np.expand_dims(auxiliary_samples, axis = 1) # aux_num * 1 * dim

        distance_matrix = auxiliary_samples - known_samples
        distance_matrix = np.linalg.norm(distance_matrix, ord = 2, axis = 2)
        K12 = np.exp(-distance_matrix ** 2 / Gaussian_kernel_width ** 2 / 2)

        return K12

    def _compute_alpha_vector(self):

        K11 = self.K11 
        K12 = self.K12
        LHS_matrix = K11 / self.n + self.lamda * np.eye(self.n)
        try:
            inverse_LHS_matrix = np.linalg.inv(LHS_matrix)
        except:
            inverse_LHS_matrix = np.linalg.pinv(LHS_matrix)

        one_vector = np.ones((self.m,1))

        RHS_vector = - K12.dot(one_vector) / (self.n * self.m * self.lamda)

        #print(RHS_vector,RHS_vector.shape,"RHS_vector")

        alpha_vector = np.dot(LHS_matrix,RHS_vector)


        return alpha_vector

    def ratio_estimator(self, test_samples):
        # test_samples (num_test_samples * dim)
        # aux_num is self.n
        # num is self.m

        test_samples = np.expand_dims(test_samples, axis = 1) # (num_test_samples, 1, dim)
        auxiliary_samples = np.expand_dims(self.auxiliary_samples, axis = 0) # (1, aux_num, dim)
        distance_matrix1 = test_samples - auxiliary_samples # (num_test_sample, aux_num, dim)
        distance_matrix1 = np.linalg.norm(distance_matrix1, ord = 2 ,axis = 2) # (num_test_sample, aux_num)
        distance_matrix1 = np.exp(-distance_matrix1 ** 2 / self.Gaussian_kernel_width ** 2 / 2) # (num_test_sample, aux_num)
        alpha_vector = np.reshape(self.alpha_vector, (self.n,))
        negative_term = np.dot(distance_matrix1,alpha_vector)


        known_samples = np.expand_dims(self.known_samples, axis = 0) # (1, num, dim)
        distance_matrix2 = test_samples - known_samples # (num_test_sample, num, dim)
        distance_matrix2 = np.linalg.norm(distance_matrix2, ord = 2 ,axis = 2) # (num_test_sample, num)
        distance_matrix2 = np.exp(-distance_matrix2 ** 2 / self.Gaussian_kernel_width ** 2 / 2) # (num_test_sample, num)
        positive_term = np.mean(distance_matrix2, axis = 1) / self.lamda

        return negative_term + positive_term

    def save_parameters(self, path):
        param_dict = {}
        param_dict["Gaussian_kernel_width"] = self.Gaussian_kernel_width
        param_dict["known_samples"] = self.known_samples
        param_dict["auxiliary_samples"] = self.auxiliary_samples
        param_dict["lamda"] = self.lamda
        param_dict["alpha_vector"] = self.alpha_vector

        torch.save(param_dict, path)




class KuLSIF_estimator(KuLSIF_density_ratio_estimation):
    def __init__(self,
                 Gaussian_kernel_width,
                 known_samples,
                 auxiliary_samples,
                 lamda,
                 alpha_vector):

        self.n = auxiliary_samples.shape[0] # number of auxiliary_samples
        self.m = known_samples.shape[0] # number of known samples
        self.dim = auxiliary_samples.shape[1]
        self.Gaussian_kernel_width = Gaussian_kernel_width
        self.known_samples = known_samples
        self.auxiliary_samples = auxiliary_samples
        self.lamda = lamda
        self.alpha_vector = alpha_vector 




if __name__ == '__main__':

    ...







