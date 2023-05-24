from torch import nn
import torch
from torch.autograd import Variable
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
            i2h_inp = torch.cat((x, hy),1)
        else:
            i2h_inp = torch.cat((x, hz, hy),1)
        hz = hz + self.dt * (torch.tanh(self.i2h(i2h_inp))
                                   - self.gamma * hy - self.epsilon * hz)
        hy = hy + self.dt * hz

        return hy, hz

class coRNN(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, dt, gamma, epsilon, device='cpu',
                 no_friction=False):
        super(coRNN, self).__init__()
        self.n_hid = n_hid
        self.cell = coRNNCell(n_inp,n_hid,dt,gamma,epsilon, no_friction=no_friction, device=device)
        self.readout = nn.Linear(n_hid, n_out)
        self.device = device

    def forward(self, x):
        ## initialize hidden states
        hy = Variable(torch.zeros(x.size(1),self.n_hid)).to(self.device)
        hz = Variable(torch.zeros(x.size(1),self.n_hid)).to(self.device)

        for t in range(x.size(0)):
            hy, hz = self.cell(x[t],hy,hz)
        output = self.readout(hy)

        return output


class coESN(nn.Module):
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon, rho, input_scaling, device='cpu', leaky=False):
        super().__init__()
        self.n_hid = n_hid
        self.device = device
        self.leaky = leaky
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
        if self.leaky:
            hz = hz  - self.dt * hz

        hy = hy + self.dt * hz
        if self.leaky:
            hy = hy  - self.dt * hy
        return hy, hz
    def forward(self, x):
        ## initialize hidden states
        hy = Variable(torch.zeros(x.size(1),self.n_hid)).to(self.device)
        hz = Variable(torch.zeros(x.size(1),self.n_hid)).to(self.device)
        all_states = []
        for t in range(x.size(0)):
            hy, hz = self.cell(x[t],hy,hz)
            all_states.append(hy)

        return all_states, [hy]  # list to be compatible with ESN implementation

