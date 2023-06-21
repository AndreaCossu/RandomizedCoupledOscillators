from scipy.integrate import odeint
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from esn import spectral_norm_scaling


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


def get_cifar_data(bs_train,bs_test):
    train_dataset = torchvision.datasets.CIFAR10(root='data/',
                                                 train=True,
                                                 transform=transforms.ToTensor(),
                                                 download=True)

    test_dataset = torchvision.datasets.CIFAR10(root='data/',
                                                train=False,
                                                transform=transforms.ToTensor())

    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [47000,3000])

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

def get_mnist_data(bs_train,bs_test):
    train_dataset = torchvision.datasets.MNIST(root='data/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='data/',
                                              train=False,
                                              transform=transforms.ToTensor())

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