from scipy.integrate import odeint
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from esn import spectral_norm_scaling


def count_parameters(model):
    """Return total number of parameters and 
    trainable parameters of a PyTorch model.
    """
    params = []
    trainable_params = []
    for p in model.parameters():
        params.append(p.numel())
        if p.requires_grad:
            trainable_params.append(p.numel())
    pytorch_total_params = sum(params)
    pytorch_total_trainableparams = sum(trainable_params)
    print('Total params:', pytorch_total_params)
    print('Total trainable params:', pytorch_total_trainableparams)


def n_params(model):
    """Return total number of parameters of the 
    LinearRegression model of Scikit-Learn.
    """
    return (sum([a.size for a in model.coef_]) +  
            sum([a.size for a in model.intercept_]))


# ########## Torch Dataset for FordA ############## #
import torch.utils.data as data
import torch.nn.functional as F

class datasetforRC(data.Dataset):
    """
    This class assumes mydata to have the form:
            [ (x1,y1), (x2,y2) ]
    where xi are inputs, and yi are targets.
    """
        
    def __init__(self, mydata):
        self.mydata = mydata
            
    def __getitem__(self, idx):
        sample = self.mydata[idx]
        idx_inp, idx_targ = sample[0], sample[1]
        idx_inp, idx_targ = torch.Tensor(idx_inp), torch.Tensor([idx_targ])
        # reshape time series for torch (batch, inplength, inpdim)
        idx_inp = idx_inp.reshape(idx_inp.shape[0], 1)
        # one-hot encoding gives problems with scikit-learn LogisticRegression of RC models
        return idx_inp, idx_targ
        
    def __len__(self):
        return len(self.mydata)


class FordA_dataset(data.Dataset):
    """
    This class assumes mydata to have the form:
            [ (x1,y1), (x2,y2) ]
    where xi are inputs, and yi are targets.
    """
        
    def __init__(self, mydata):
        self.mydata = mydata
            
    def __getitem__(self, idx):
        sample = self.mydata[idx]
        idx_inp, idx_targ = sample[0], sample[1]
        idx_inp, idx_targ = torch.Tensor(idx_inp), torch.Tensor([idx_targ])
        # reshape time series for torch (batch, inplength, inpdim)
        idx_inp = idx_inp.reshape(idx_inp.shape[0], 1)
        # one-hot encoding targets
        idx_targ = F.one_hot(idx_targ.type(torch.int64), num_classes=2).float()
        # reshape target for torch (batch, classes)
        idx_targ = idx_targ.reshape(idx_targ.shape[1])
        return idx_inp, idx_targ
        
    def __len__(self):
        return len(self.mydata)


class Adiac_dataset(data.Dataset):
    """
    This class assumes mydata to have the form:
            [ (x1,y1), (x2,y2) ]
    where xi are inputs, and yi are targets.
    """
    
    def __init__(self, mydata):
        self.mydata = mydata
        
    def __getitem__(self, idx):
        sample = self.mydata[idx]
        idx_inp, idx_targ = sample[0], sample[1]
        idx_inp, idx_targ = torch.Tensor(idx_inp), torch.Tensor([idx_targ])
        # reshape time series for torch (batch, inplength, inpdim)
        idx_inp = idx_inp.reshape(idx_inp.shape[0], 1)
        # one-hot encoding targets
        idx_targ = F.one_hot(idx_targ.type(torch.int64), num_classes=37).float()
        # reshape target for torch (batch, classes)
        idx_targ = idx_targ.reshape(idx_targ.shape[1])
        return idx_inp, idx_targ
    
    def __len__(self):
        return len(self.mydata)
# ################################################# #


class LSTM(nn.Module):
    def __init__(self, n_inp, n_hid, n_out):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_inp, n_hid, batch_first=True)
        self.readout = torch.nn.Linear(n_hid, n_out)

    def forward(self, x):
        out, h = self.lstm(x)
        out = self.readout(out[:, -1])
        return out


class coRNNCell(nn.Module):
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon, no_friction=False, device='cpu'):
        super(coRNNCell, self).__init__()
        self.dt = dt
        gamma_min, gamma_max = gamma
        eps_min, eps_max = epsilon
        self.gamma = torch.rand(n_hid, requires_grad=False, device=device) * (gamma_max - gamma_min) + gamma_min
        self.epsilon = torch.rand(n_hid, requires_grad=False, device=device) * (eps_max - eps_min) + eps_min
        if no_friction:
            self.i2h = nn.Linear(n_inp + n_hid, n_hid)
        else:
            self.i2h = nn.Linear(n_inp + n_hid + n_hid, n_hid)
        self.no_friction = no_friction

    def forward(self,x,hy,hz):
        if self.no_friction:
            i2h_inp = torch.cat((x, hy), 1)
        else:
            i2h_inp = torch.cat((x, hz, hy), 1)
        hz = hz + self.dt * (torch.tanh(self.i2h(i2h_inp))
                             - self.gamma * hy - self.epsilon * hz)
        hy = hy + self.dt * hz

        return hy, hz

class coRNN(nn.Module):
    """
    Batch-first (B, L, I)
    """
    def __init__(self, n_inp, n_hid, n_out, dt, gamma, epsilon, device='cpu',
                 no_friction=False):
        super(coRNN, self).__init__()
        self.n_hid = n_hid
        self.cell = coRNNCell(n_inp,n_hid,dt,gamma,epsilon, no_friction=no_friction, device=device)
        self.readout = nn.Linear(n_hid, n_out)
        self.device = device

    def forward(self, x):
        ## initialize hidden states
        hy = torch.zeros(x.size(0), self.n_hid).to(self.device)
        hz = torch.zeros(x.size(0), self.n_hid).to(self.device)

        for t in range(x.size(1)):
            hy, hz = self.cell(x[:, t],hy,hz)
        output = self.readout(hy)

        return output


class coESN(nn.Module):
    """
    Batch-first (B, L, I)
    """
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon, rho, input_scaling, device='cpu',
                 fading=False):
        super().__init__()
        self.n_hid = n_hid
        self.device = device
        self.fading = fading
        self.dt = dt
        if isinstance(gamma, tuple):
            gamma_min, gamma_max = gamma
            self.gamma = torch.rand(n_hid, requires_grad=False, device=device) * (gamma_max - gamma_min) + gamma_min
        else:
            self.gamma = gamma
        if isinstance(epsilon, tuple):
            eps_min, eps_max = epsilon
            self.epsilon = torch.rand(n_hid, requires_grad=False, device=device) * (eps_max - eps_min) + eps_min
        else:
            self.epsilon = epsilon

        h2h = 2 * (2 * torch.rand(n_hid, n_hid) - 1)
        h2h = spectral_norm_scaling(h2h, rho)
        self.h2h = nn.Parameter(h2h, requires_grad=False)

        x2h = torch.rand(n_inp, n_hid) * input_scaling
        self.x2h = nn.Parameter(x2h, requires_grad=False)
        bias = (torch.rand(n_hid) * 2 - 1) * input_scaling
        self.bias = nn.Parameter(bias, requires_grad=False)

    def cell(self, x, hy, hz):
        hz = hz + self.dt * (torch.tanh(
            torch.matmul(x, self.x2h)  + torch.matmul(hy, self.h2h) + self.bias)
                             - self.gamma * hy - self.epsilon * hz)
        if self.fading:
            hz = hz - self.dt * hz

        hy = hy + self.dt * hz
        if self.fading:
            hy = hy - self.dt * hy
        return hy, hz

    def forward(self, x):
        ## initialize hidden states
        hy = torch.zeros(x.size(0),self.n_hid).to(self.device)
        hz = torch.zeros(x.size(0),self.n_hid).to(self.device)
        all_states = []
        for t in range(x.size(1)):
            hy, hz = self.cell(x[:, t],hy,hz)
            all_states.append(hy)

        return torch.stack(all_states, dim=1), [hy]  # list to be compatible with ESN implementation


def get_cifar_data(bs_train,bs_test, whole_train=False):
    train_dataset = torchvision.datasets.CIFAR10(root='data/',
                                                 train=True,
                                                 transform=transforms.ToTensor(),
                                                 download=True)

    test_dataset = torchvision.datasets.CIFAR10(root='data/',
                                                train=False,
                                                transform=transforms.ToTensor())


    if whole_train:
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [50000,0])
    else:
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [47000,3000])

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=bs_train,
                                               shuffle=True,
                                               drop_last=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=bs_test,
                                               shuffle=False,
                                               drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=bs_test,
                                              shuffle=False,
                                              drop_last=True)

    return train_loader, valid_loader, test_loader

def get_lorenz(N, F, num_batch=128, lag=25, washout=200):
    # https://en.wikipedia.org/wiki/Lorenz_96_model
    def L96(x, t):
        """Lorenz 96 model with constant forcing"""
        # Setting up vector
        d = np.zeros(N)
        # Loops over indices (with operations and Python underflow indexing handling edge cases)
        for i in range(N):
            d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
        return d

    dt = 0.01
    t = np.arange(0.0, 20+(lag*dt)+(washout*dt), dt)
    dataset = []
    for i in range(num_batch):
        x0 = np.random.rand(N) + F - 0.5 # [F-0.5, F+0.5]
        x = odeint(L96, x0, t)
        dataset.append(x)
    dataset = np.stack(dataset, axis=0)  # (num_batch, 2000, 5)
    dataset = torch.from_numpy(dataset).float()
    return dataset


def get_mnist_data(bs_train,bs_test, whole_train=False):
    train_dataset = torchvision.datasets.MNIST(root='data/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='data/',
                                              train=False,
                                              transform=transforms.ToTensor())

    if whole_train:
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [60000,0])
    else:
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [57000,3000])

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=bs_train,
                                               shuffle=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                              batch_size=bs_test,
                                              shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=bs_test,
                                              shuffle=False)

    return train_loader, valid_loader, test_loader


@torch.no_grad()
def check(m):
    xi = torch.max(torch.abs(1 - m.epsilon * m.dt))
    eta = torch.max(torch.abs(1 - m.gamma * m.dt**2))
    sigma = torch.norm(m.h2h)
    print(xi, eta, sigma, torch.max(m.epsilon), torch.max(m.gamma))

    if (xi - eta) / (m.dt ** 2) <= xi - torch.max(m.gamma):
        if sigma <= (xi - eta) / (m.dt ** 2) and xi < 1 / (1 + m.dt):
            return True
        if (xi - eta) / (m.dt ** 2) < sigma and sigma <= xi - torch.max(m.gamma) and sigma < (1 - xi - eta) / m.dt**2:
            return True
        if sigma >= xi - torch.max(m.gamma) and sigma <= (1 - eta - m.dt * torch.max(m.gamma)) / (m.dt * (1 + m.dt)):
            return True
    else:
        if sigma <= xi - torch.max(m.gamma) and xi < 1 / (1 + m.dt):
            return True
        if xi - torch.max(m.gamma) < sigma and sigma <= (xi - eta) / (m.dt ** 2) and sigma < ((1 - xi) / m.dt) - torch.max(m.gamma):
            return True
        if sigma >= (xi - eta) / m.dt**2 and sigma < (1 - eta - m.dt * torch.max(m.gamma)) / (m.dt * (1 + m.dt)):
            return True
    return False


def get_FordA_data(bs_train,bs_test, whole_train=False, RC=True):

    def fromtxt_to_numpy(filename='FordA_TRAIN.txt', valid_len=1320):
        # read the txt file
        forddata = np.genfromtxt(filename, dtype='float64')
        # create a list of lists with each line of the txt file
        l = []
        for i in forddata:
            el = list(i)
            while len(el) < 3:
                el.append('a')
            l.append(el)
        # create a numpy array from the list of lists
        arr = np.array(l)
        if valid_len is None:
            test_targets = (arr[:,0]+1)/2
            test_series = arr[:,1:]
            return test_series, test_targets
        else:
            if valid_len == 0:
                train_targets = (arr[:,0]+1)/2
                train_series = arr[:,1:]
                val_targets = arr[0:0,0] # empty
                val_series = arr[0:0,1:] # empty
            elif valid_len > 0 :
                train_targets = (arr[:-valid_len,0]+1)/2
                train_series = arr[:-valid_len,1:]
                val_targets = (arr[-valid_len:,0]+1)/2
                val_series = arr[-valid_len:,1:]
            return train_series, train_targets, val_series, val_targets

    # Generate list of input-output pairs
    def inp_out_pairs(data_x, data_y):
        mydata = []
        for i in range(len(data_y)):
            sample = (data_x[i,:], data_y[i])
            mydata.append(sample)
        return mydata

    # generate torch datasets
    if whole_train:
        valid_len = 0
    else:
        valid_len = 1320
    train_series, train_targets, val_series, val_targets = fromtxt_to_numpy(filename='FordA_TRAIN.txt', valid_len=valid_len)
    mytraindata, myvaldata = inp_out_pairs(train_series, train_targets), inp_out_pairs(val_series, val_targets)
    if RC:
        mytraindata, myvaldata = datasetforRC(mytraindata), datasetforRC(myvaldata)
        test_series, test_targets = fromtxt_to_numpy(filename='FordA_TEST.txt', valid_len=None)
        mytestdata = inp_out_pairs(test_series, test_targets)
        mytestdata = datasetforRC(mytestdata)
    else:
        mytraindata, myvaldata = FordA_dataset(mytraindata), FordA_dataset(myvaldata)
        test_series, test_targets = fromtxt_to_numpy(filename='FordA_TEST.txt', valid_len=None)
        mytestdata = inp_out_pairs(test_series, test_targets)
        mytestdata = FordA_dataset(mytestdata)


    # generate torch dataloaders
    mytrainloader = data.DataLoader(mytraindata, 
                    batch_size=bs_train, shuffle=True, drop_last=True)
    myvaloader = data.DataLoader(myvaldata, 
                        batch_size=bs_test, shuffle=False, drop_last=True)
    mytestloader = data.DataLoader(mytestdata, 
                batch_size=bs_test, shuffle=False, drop_last=True)
    return mytrainloader, myvaloader, mytestloader


def get_Adiac_data(bs_train,bs_test, whole_train=False, RC=True):

    def fromtxt_to_numpy(filename='Adiac_TRAIN.txt', valid_len=120):
        # read the txt file
        adiacdata = np.genfromtxt(filename, dtype='float64')
        # create a list of lists with each line of the txt file
        l = []
        for i in adiacdata:
            el = list(i)
            while len(el) < 3:
                el.append('a')
            l.append(el)
        # create a numpy array from the list of lists
        arr = np.array(l)
        if valid_len is None:
            test_targets = arr[:,0]-1
            test_series = arr[:,1:]
            return test_series, test_targets
        else:
            if valid_len == 0:
                train_targets = arr[:,0]-1
                train_series = arr[:,1:]
                val_targets = arr[0:0,0] # empty
                val_series = arr[0:0,1:] # empty
            elif valid_len > 0 :
                train_targets = arr[:-valid_len,0]-1
                train_series = arr[:-valid_len,1:]
                val_targets = arr[-valid_len:,0]-1
                val_series = arr[-valid_len:,1:]
            return train_series, train_targets, val_series, val_targets

    # Generate list of input-output pairs
    def inp_out_pairs(data_x, data_y):
        mydata = []
        for i in range(len(data_y)):
            sample = (data_x[i,:], data_y[i])
            mydata.append(sample)
        return mydata

    # generate torch datasets
    if whole_train:
        valid_len = 0
    else:
        valid_len = 120
    train_series, train_targets, val_series, val_targets = fromtxt_to_numpy(filename='Adiac_TRAIN.txt', valid_len=valid_len)
    mytraindata, myvaldata = inp_out_pairs(train_series, train_targets), inp_out_pairs(val_series, val_targets)
    if RC:
        mytraindata, myvaldata = datasetforRC(mytraindata), datasetforRC(myvaldata)
        test_series, test_targets = fromtxt_to_numpy(filename='Adiac_TEST.txt', valid_len=None)
        mytestdata = inp_out_pairs(test_series, test_targets)
        mytestdata = datasetforRC(mytestdata)
    else:
        mytraindata, myvaldata = Adiac_dataset(mytraindata), Adiac_dataset(myvaldata)
        test_series, test_targets = fromtxt_to_numpy(filename='Adiac_TEST.txt', valid_len=None)
        mytestdata = inp_out_pairs(test_series, test_targets)
        mytestdata = Adiac_dataset(mytestdata)


    # generate torch dataloaders
    mytrainloader = data.DataLoader(mytraindata, 
                    batch_size=bs_train, shuffle=True, drop_last=True)
    myvaloader = data.DataLoader(myvaldata, 
                        batch_size=bs_test, shuffle=False, drop_last=True)
    mytestloader = data.DataLoader(mytestdata, 
                batch_size=bs_test, shuffle=False, drop_last=True)
    return mytrainloader, myvaloader, mytestloader