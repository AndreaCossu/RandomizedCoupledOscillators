# https://en.wikipedia.org/wiki/Lorenz_96_model
import torch
from scipy.integrate import odeint
import numpy as np


def get_lorenz(N, F, num_batch=128):
    def L96(x, t):
        """Lorenz 96 model with constant forcing"""
        # Setting up vector
        d = np.zeros(N)
        # Loops over indices (with operations and Python underflow indexing handling edge cases)
        for i in range(N):
            d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
        return d

    lag = 25
    washout = 200
    dt = 0.01
    t = np.arange(0.0, 20+(lag*dt)+(washout*dt), dt) # 2000 points
    dataset = []
    for i in range(num_batch):
        x0 = np.random.rand(N) + F - 0.5 # [F-0.5, F+0.5]
        x = odeint(L96, x0, t)
        dataset.append(x)
    dataset = np.stack(dataset, axis=0)  # (num_batch, 2000, 5)
    dataset = torch.from_numpy(dataset).permute(1, 0, 2).float() # (2000, num_batch, 5)
    return dataset


if __name__ == '__main__':
    N, F = 5, 8
    dataset = get_lorenz(N, F)