import numpy as np
import torch.nn.utils
import argparse
from tqdm import tqdm
from utils import get_mackey_glass, LSTM, coRNN


parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--n_hid', type=int, default=100,
                    help='hidden size of recurrent net')
parser.add_argument('--batch', type=int, default=32,
                    help='batch size')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=30,
                    help='max epochs')
parser.add_argument('--window_size', type=int, default=200,
                    help='size of the input window for rnn')
parser.add_argument('--dt', type=float, default=0.076,
                    help='step size <dt> of the coRNN')
parser.add_argument('--gamma', type=float, default=0.4,
                    help='y controle parameter <gamma> of the coRNN')
parser.add_argument('--epsilon', type=float, default=8.0,
                    help='z controle parameter <epsilon> of the coRNN')
parser.add_argument('--gamma_range', type=float, default=2.7,
                    help='y controle parameter <gamma> of the coRNN')
parser.add_argument('--epsilon_range', type=float, default=4.7,
                    help='z controle parameter <epsilon> of the coRNN')
parser.add_argument('--cpu', action="store_true")
parser.add_argument('--check', action="store_true")
parser.add_argument('--no_friction', action="store_true")
parser.add_argument('--lstm', action="store_true")
parser.add_argument('--use_test', action="store_true")


args = parser.parse_args()
print(args)

main_folder = 'result'

device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
print("Using device ", device)
n_inp = 1
n_out = 1


gamma = (args.gamma - args.gamma_range / 2., args.gamma + args.gamma_range / 2.)
epsilon = (args.epsilon - args.epsilon_range / 2., args.epsilon + args.epsilon_range / 2.)

if args.lstm:
    model = LSTM(n_inp, args.n_hid, n_out).to(device)
else:
    model = coRNN(n_inp, args.n_hid, n_out,args.dt,gamma,epsilon,
                  no_friction=args.no_friction, device=device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.MSELoss()

(train_dataset, train_target), (valid_dataset, valid_target), (test_dataset, test_target) = get_mackey_glass(washout=0,
                                                                                                             window_size=args.window_size)
train_dataset = torch.utils.data.TensorDataset(train_dataset, train_target)
valid_dataset = torch.utils.data.TensorDataset(valid_dataset, valid_target)
test_dataset = torch.utils.data.TensorDataset(test_dataset, test_target)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, drop_last=False)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch, shuffle=False, drop_last=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, drop_last=False)


@torch.no_grad()
def test(dataloader):
    model.eval()
    predictions, target = [], []
    for x, y in dataloader:
        out = model(x.to(device))
        predictions.append(out.cpu())
        target.append(y)

    predictions = torch.cat(predictions, dim=0).numpy()
    target = torch.cat(target, dim=0).numpy()
    mse = np.mean(np.square(predictions - target))
    rmse = np.sqrt(mse)
    norm = np.sqrt(np.square(target).mean())
    nrmse = rmse / (norm + 1e-9)
    return nrmse

train_losses = []
val_losses = []
test_losses = []
for epoch in range(args.epochs):
    model.train()
    train_loss = 0.
    for x, y in tqdm(train_loader):
        optimizer.zero_grad()
        out = model(x.to(device))
        loss = criterion(out, y.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss = train_loss / float(len(train_loader))
    train_losses.append(train_loss)
    valid_nmse = test(valid_loader)
    val_losses.append(valid_nmse)
    test_nmse = test(test_loader) if args.use_test else 0.0
    test_losses.append(test_nmse)

if args.lstm:
    f = open(f'{main_folder}/mackey_log_lstm.txt', 'a')
elif args.no_friction: # hcoRNN
    f = open(f'{main_folder}/mackey_log_hcornn.txt', 'a')
else: # cornn
    f = open(f'{main_folder}/mackey_log_cornn.txt', 'a')
ar = ''
for k, v in vars(args).items():
    ar += f'{str(k)}: {str(v)}, '
ar += f'train_list: {[str(round(el, 5)) for el in train_losses]}, valid_list: {[str(round(el, 5)) for el in val_losses]}, test_list: {[str(round(el, 5)) for el in test_losses]} train: {str(round(min(train_losses), 5))} valid: {str(round(min(val_losses), 5))}, test: {str(round(min(test_losses), 5))}'
f.write(ar + '\n')
f.write('**************\n\n\n')
f.close()