#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
#from utils.MNIST4FedMD10Clients import MNIST4FedMD10Clients
from utils.MNIST4KuLSIF import MNIST4KuLSIFandFedMD10Clients1Class, MNIST4KuLSIFandFedMD10Clients2Class #, MNISTProxyDataset4FedMD10Clients1Class
from step3_MNISTSelectedProxyDataset import * #MNIST10Clients1ClassSelectedProxyDataset_hardlabel, MNIST10Clients2ClassSelectedProxyDataset_hardlabel
import copy
import numpy as np
import datetime


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20000, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--local_step', type=int, default=1,
                    help='learning rate (default: 5)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-mps', action='store_true', default=False,
                    help='disables macOS GPU training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--tau', type=float, default=1,
                    help='temperature coefficient')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
args = parser.parse_args()


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
        #print(x.size())
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        #x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1), F.softmax(x / args.tau, dim = 1), x


def sample_a_mini_batch(local_dataset, batch_size):
    mini_batch_data = []
    mini_batch_target = []
    num = local_dataset.__len__()
    random_index = np.random.choice(num, size = batch_size, replace = False)
    for i in range(len(random_index)):
        data = local_dataset.__getitem__(random_index[i])
        mini_batch_data.append(data[0])
        mini_batch_target.append(data[1])
    return torch.stack(mini_batch_data, dim = 0), torch.LongTensor(mini_batch_target) #torch.cat(mini_batch_target)


def generate_softlabel(args, model_list, device, global_dataset, batch_size):
    #model.eval()
    mini_batch_data = []
    mini_batch_ensemble_label = []
    num = global_dataset.__len__()
    random_index = np.random.choice(num, size = batch_size, replace = False)

    for i in range(len(random_index)):
        data, ensemble_label = global_dataset.__getitem__(random_index[i])
        mini_batch_data.append(data)
        mini_batch_ensemble_label.append(ensemble_label)

    mini_batch_data = torch.stack(mini_batch_data, dim = 0)
    mini_batch_ensemble_label = torch.stack(mini_batch_ensemble_label, dim = 0)


    with torch.no_grad():
        mini_batch_softlabel = 0.0
        for j in range(10):
            model_list[j].eval()
            _, softlabel, _ = model_list[j](mini_batch_data.to(device))

            mini_batch_softlabel += softlabel * torch.reshape(mini_batch_ensemble_label[:,j], (-1,1) ).to(device) #/ 10.0

    return mini_batch_data, mini_batch_softlabel




def local_train_global_dataset(args, model, device, mini_batch_data, mini_batch_softlabel, optimizer, scheduler, local_step = 1):
    model.train()
    for _ in range(local_step):

        data, target = mini_batch_data.to(device), mini_batch_softlabel.to(device)
        optimizer.zero_grad()
        _, _, output = model(data)
        #loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output, target) #cross_entropy() supports unnormalized input logtits
        loss.backward()
        optimizer.step()
    scheduler.step()



def local_train(args, model, device, local_dataset, optimizer, scheduler, local_step = 1):
    model.train()
    #print("local_step", local_step)
    for _ in range(local_step):
        #print("local")
        data, target = sample_a_mini_batch(local_dataset, args.batch_size)
        #print(target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _, _ = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step()




def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _, _ = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    return 100. * correct / len(test_loader.dataset)


def main():
    # Training settings
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    accuracy_record = []

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device",device)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    test_dataset = datasets.MNIST('./data', train=False,
                       transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    #global_dataset = MNIST4FedMD10Clients(proxy_dataset = True, dataset_path = "./utils/MNISTwithProxyDataset.pth")

    global_dataset = MNIST10Clients1ClassSelectedProxyDataset_softlabel() #MNIST4FedMD10Clients(proxy_dataset = True, dataset_path = "./utils/MNISTwithProxyDataset.pth")

    FL_training_dataset_list = {}
    FL_local_model_list = {}
    FL_optimizer_list = {}
    FL_sceduler_list = {}
    for i in range(10):
        # dataset preparation
        train_dataset = MNIST4KuLSIFandFedMD10Clients1Class(client_index = i,
                                                            dataset_path = "./utils/MNISTwithProxyDataset.pth",
                                                            proxy_dataset = False,)

        #MNIST4FedMD10Clients(proxy_dataset = False, client_index = i, dataset_path = "./utils/MNISTwithProxyDataset.pth")
        FL_training_dataset_list[i] = train_dataset
        # model preparation
        FL_local_model_list[i] = Net().to(device)
        #optimizer preparation
        FL_optimizer_list[i] = optim.SGD(FL_local_model_list[i].parameters(), lr=args.lr)
        # scheduler preparation
        FL_sceduler_list[i] = StepLR(FL_optimizer_list[i], step_size=5000, gamma=args.gamma)

    #model_aggregation(FL_local_model_list)

    acc_max = 0

    datetime_str = datetime.datetime.today().strftime('_%Y-%m-%d_%H-%M-%S_')

    for epoch in range(1, args.epochs + 1):
        
        for i in range(10):
            local_train(args, 
                        model = FL_local_model_list[i], 
                        device = device, 
                        local_dataset = FL_training_dataset_list[i], 
                        optimizer = FL_optimizer_list[i], 
                        scheduler = FL_sceduler_list[i], 
                        local_step = args.local_step)

        mini_batch_data, mini_batch_softlabel = generate_softlabel(args, FL_local_model_list, device, global_dataset, batch_size = int(0.1 * global_dataset.__len__()))
        #print("epoch",epoch , mini_batch_softlabel[-1])

        for i in range(10):
            local_train_global_dataset(args, 
                                       model = FL_local_model_list[i], 
                                       device= device, 
                                       mini_batch_data = mini_batch_data, 
                                       mini_batch_softlabel = mini_batch_softlabel, 
                                       optimizer = FL_optimizer_list[i], 
                                       scheduler = FL_sceduler_list[i], 
                                       local_step = args.local_step * 10)

        if epoch % 50 == 0:
            print("epoch",epoch)
            accuracy_list = []
            for j in range(10):
                accuracy_list.append(test(FL_local_model_list[j], device, test_loader))
            #accuracy = test(FL_local_model_list[0], device, test_loader)
            accuracy = np.mean(accuracy_list) #torch.mean(torch.cat(accuracy_list))
            print("accuracy",accuracy,"accuracy_list", accuracy_list)
            if accuracy > acc_max:
                acc_max = accuracy
            print("acc_max", acc_max)
            accuracy_record.append(accuracy)
            #torch.save({"accuracy":accuracy_record, "args":args},"./FedMD_results/softlabel_result_local" + str(args.local_step) + "_tau_" + str(args.tau) + datetime_str +".pth")
            #test(FL_local_model_list[1], device, test_loader)
    print("max accuracy", acc_max)

    # if args.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()




