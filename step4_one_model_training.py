import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.nn.init as init
import copy
from utils.MNIST4KuLSIF import MNIST4KuLSIFandFedMD10Clients1Class, MNIST4KuLSIFandFedMD10Clients2Class #, MNISTProxyDataset4FedMD10Clients1Class
from step3_MNISTSelectedProxyDataset import * #MNIST10Clients1ClassSelectedProxyDataset_hardlabel, MNIST10Clients2ClassSelectedProxyDataset_hardlabel
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

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=5e-2)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--channel_noise', type=float, default=0.3162)
#parser.add_argument('--intermediate_dim', type=int, default=64)
parser.add_argument('--beta', type=float, default=1e-3)
parser.add_argument('--threshold', type=float, default=1e-2)
parser.add_argument('--test', type=int, default=0)
parser.add_argument('--weights', type=str)
parser.add_argument('--local_step', type=int, default = 1)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)


# def train(args, model, device, train_loader, optimizer, epoch):

#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)

#         optimizer.zero_grad()
#         output = model(data)

#         if len(target.size()) > 1:
#             loss = - torch.mean(torch.sum(output * target, dim = 1)) #F.nll_loss does not support multi-target loss
#         else:
        
#             loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()


def train_with_local_data(args, model, device, local_dataset, optimizer, epoch):

    model.train()

    data, target = sample_a_mini_batch_from_local_dataset(local_dataset, args.batch_size)
    data, target = data.to(device), target.to(device)
    #proxy_data, proxy_target = sample_a_mini_batch_from_proxy_dataset(proxy_dataset, args.batch_size)
    #proxy_data, proxy_target = proxy_data.to(device), proxy_target.to(device)

    output1 = model(data)
    #output2 = model(proxy_data)

    loss1 = F.nll_loss(output1, target)
    #loss2 = - torch.mean(torch.sum(output2 * proxy_target, dim = 1))

    loss = loss1 #1e-1 * loss1 + loss2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_with_proxy_data(args, model, device, proxy_dataset, optimizer, epoch):

    model.train()

    #data, target = sample_a_mini_batch_from_local_dataset(local_dataset, args.batch_size)
    #data, target = data.to(device), target.to(device)
    proxy_data, proxy_target = sample_a_mini_batch_from_proxy_dataset(proxy_dataset, args.batch_size)
    proxy_data, proxy_target = proxy_data.to(device), proxy_target.to(device)

    #output1 = model(data)
    output2 = model(proxy_data)

    #loss1 = F.nll_loss(output1, target)
    loss2 = - torch.mean(torch.sum(output2 * proxy_target, dim = 1))

    loss = loss2 #1e-1 * loss1 + loss2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# def train(args, model, device, local_dataset, proxy_dataset, optimizer, epoch):

#     model.train()

#     data, target = sample_a_mini_batch_from_local_dataset(local_dataset, args.batch_size)
#     data, target = data.to(device), target.to(device)
#     proxy_data, proxy_target = sample_a_mini_batch_from_proxy_dataset(proxy_dataset, args.batch_size)
#     proxy_data, proxy_target = proxy_data.to(device), proxy_target.to(device)

#     output1 = model(data)
#     output2 = model(proxy_data)

#     loss1 = F.nll_loss(output1, target)
#     loss2 = - torch.mean(torch.sum(output2 * proxy_target, dim = 1))

#     loss = 1e-1 * loss1 + loss2

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #data = torch.reshape(data, (-1, 28 * 28))
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    return 100. * correct / len(test_loader.dataset)#, pruned_number


def sample_a_mini_batch_from_local_dataset(local_dataset, batch_size):
    mini_batch_data = []
    mini_batch_target = []
    num = local_dataset.__len__()
    random_index = np.random.choice(num, size = batch_size, replace = False)
    for i in range(len(random_index)):
        data = local_dataset.__getitem__(random_index[i])
        mini_batch_data.append(data[0])
        mini_batch_target.append(data[1])
    return torch.stack(mini_batch_data, dim = 0), torch.LongTensor(mini_batch_target)

def sample_a_mini_batch_from_proxy_dataset(proxy_dataset, batch_size):
    mini_batch_data = []
    mini_batch_target = []
    num = proxy_dataset.__len__()
    random_index = np.random.choice(num, size = batch_size, replace = False)
    for i in range(len(random_index)):
        data, ensemble_label = proxy_dataset.__getitem__(random_index[i])
        mini_batch_data.append(data)
        mini_batch_target.append(ensemble_label)
        #print(mini_batch_target)
    return torch.stack(mini_batch_data, dim = 0), torch.stack(mini_batch_target, dim = 0) #torch.Tensor(mini_batch_target)



def main_train():
    kwargs = {'num_workers': 1, 'pin_memory': True}

    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, download = True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay = 5e-5)
    
    #clean_test_acc = 0
    #poisoned_test_acc = 0
    #pruned_dim = 0
    #saved_model = {}


    local_train_dataset =  MNIST4KuLSIFandFedMD10Clients2Class(client_index = 5,
                                                               dataset_path = "./utils/MNISTwithProxyDataset.pth",
                                                               proxy_dataset = False,)

    proxy_dataset = MNIST10Clients2ClassSelectedProxyDataset_softlabel()
    #proxy_dataset = MNIST10Clients2ClassSelectedProxyDataset_hardlabel()


    train_loader1 = torch.utils.data.DataLoader(
                local_train_dataset,
                batch_size=args.batch_size, shuffle=True, **kwargs)

    train_loader2 = torch.utils.data.DataLoader(
                proxy_dataset,
                batch_size=args.batch_size, shuffle=True, **kwargs)

    #train(args, model, device, train_loader1, optimizer, epoch = 1)
    for epoch in range(1, args.epochs + 1):

        

        #train(args, model, device, local_train_dataset, proxy_dataset, optimizer, epoch)

        for i in range(args.local_step):

            train_with_local_data(args, model, device, local_train_dataset, optimizer, epoch)

        for i in range(args.local_step * 5):

            train_with_proxy_data(args, model, device, proxy_dataset, optimizer, epoch)
        
        if epoch % 5 == 0:

            acc = test(args, model, device, test_loader)

            print('\nepoch:',epoch,acc)
            #print(acc)


if __name__ == '__main__':
    seed_torch(0)

    main_train()

