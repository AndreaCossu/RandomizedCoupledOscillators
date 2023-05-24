from torch import nn
import torch
from torch.autograd import Variable

class coRNNCell(nn.Module):
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon, no_friction=False):
        super(coRNNCell, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
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
        self.cell = coRNNCell(n_inp,n_hid,dt,gamma,epsilon, no_friction=no_friction)
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
