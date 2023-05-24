from torch import nn, optim
import torch
import model
import torch.nn.utils
import utils
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--n_hid', type=int, default=128,
                    help='hidden size of recurrent net')
parser.add_argument('--T', type=int, default=2000,
                    help='length of sequences')
parser.add_argument('--max_steps', type=int, default=50000,
                    help='max learning steps')
parser.add_argument('--log_interval', type=int, default=100,
                    help='log interval')
parser.add_argument('--batch', type=int, default=50,
                    help='batch size')
parser.add_argument('--batch_test', type=int, default=1000,
                    help='size of test set')
parser.add_argument('--lr', type=float, default=2e-2,
                    help='learning rate')
parser.add_argument('--dt',type=float, default=6e-2,
                    help='step size <dt> of the coRNN')
parser.add_argument('--gamma',type=float, default=66,
                    help='y controle parameter <gamma> of the coRNN')
parser.add_argument('--epsilon',type=float, default = 15,
                    help='z controle parameter <epsilon> of the coRNN')
parser.add_argument('--cpu', action="store_true")
parser.add_argument('--no_friction', action="store_true")

args = parser.parse_args()

n_inp = 2
n_out = 1
device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")

model = model.coRNN(n_inp, args.n_hid, n_out, args.dt, args.gamma, args.epsilon,
                    device=device, no_friction=args.no_friction).to(device)


objective = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def test():
    model.eval()
    with torch.no_grad():
        data, label = utils.get_batch(args.T, args.batch_test)
        label = label.unsqueeze(1)
        out = model(data.to(device))
        loss = objective(out, label.to(device))

    return loss.item()

def train():
    test_mse = []
    for i in tqdm(range(args.max_steps)):
        data, label = utils.get_batch(args.T,args.batch)
        label = label.unsqueeze(1)

        optimizer.zero_grad()
        out = model(data.to(device))
        loss = objective(out, label.to(device))
        loss.backward()
        optimizer.step()

        if(i%100==0 and i!=0):
            mse_error = test()
            print('Test MSE: {:.6f}'.format(mse_error))
            test_mse.append(mse_error)
            model.train()

            plt.figure()
            plt.plot(range(len(test_mse)), test_mse)
            if args.no_friction:
                plt.savefig('result/adding_no_friction.png')
            else:
                plt.savefig('result/adding.png')
            plt.close()

if __name__ == '__main__':
    train()
