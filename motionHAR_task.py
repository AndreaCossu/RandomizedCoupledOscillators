from torch import nn, optim
import torch
import torch.nn.utils
from utils import get_motion_data, coRNN, coESN, check, LSTM
from pathlib import Path
import argparse
from tqdm import tqdm
from esn import DeepReservoir
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--n_hid', type=int, default=256,
                    help='hidden size of recurrent net')
parser.add_argument('--epochs', type=int, default=120,
                    help='max epochs')
parser.add_argument('--batch', type=int, default=120,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0054,
                    help='learning rate')
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
parser.add_argument('--esn', action="store_true")
parser.add_argument('--inp_scaling', type=float, default=1.,
                    help='ESN input scaling')
parser.add_argument('--rho', type=float, default=0.99,
                    help='ESN spectral radius')
parser.add_argument('--leaky', type=float, default=1.0,
                    help='ESN spectral radius')
parser.add_argument('--lstm', action="store_true")
parser.add_argument('--use_test', action="store_true")

args = parser.parse_args()
print(args)

n_inp = 12
n_out = 6
bs_test = 500

main_folder = 'result'

device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
gamma = (args.gamma - args.gamma_range / 2., args.gamma + args.gamma_range / 2.)
epsilon = (args.epsilon - args.epsilon_range / 2., args.epsilon + args.epsilon_range / 2.)

if args.lstm:
  model = LSTM(n_inp, args.n_hid, n_out).to(device)
elif args.esn and not args.no_friction:
    model = DeepReservoir(n_inp, tot_units=args.n_hid, spectral_radius=args.rho,
                          input_scaling=args.inp_scaling,
                          connectivity_recurrent=args.n_hid,
                          connectivity_input=args.n_hid, leaky=args.leaky).to(device)
elif args.esn and args.no_friction:
    model = coESN(n_inp, args.n_hid, args.dt, gamma, epsilon, args.rho,
                  args.inp_scaling, device=device).to(device)
    if args.check:
        check_passed = check(model)
        print("Check: ", check_passed)
        if not check_passed:
            raise ValueError("Check not passed.")
else:
    model = coRNN(n_inp, args.n_hid, n_out,args.dt,gamma,epsilon,
                  no_friction=args.no_friction, device=device).to(device)

train_loader, valid_loader, test_loader = get_motion_data(args.batch, bs_test)

objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def test(data_loader):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        i = -1
        for x, labels in tqdm(data_loader):
            x, labels = x.to(device), labels.to(device)
            i += 1

            output = model(x)
            test_loss += objective(output, labels).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()
    test_loss /= i+1
    accuracy = 100. * correct / len(data_loader.dataset)

    return accuracy.item()

@torch.no_grad()
def test_esn(data_loader, classifier, scaler):
    activations, ys = [], []
    for x, labels in tqdm(data_loader):
        x = x.to(device)
        output = model(x)[-1][0]
        activations.append(output.cpu())
        ys.append(labels)
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)

if args.esn:
    activations, ys = [], []
    for x, labels in tqdm(train_loader):
        x = x.to(device)
        output = model(x)[-1][0]
        activations.append(output.cpu())
        ys.append(labels)
    activations = torch.cat(activations, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    classifier = LogisticRegression(max_iter=1000).fit(activations, ys)
    valid_acc = test_esn(valid_loader, classifier, scaler)
    test_acc = test_esn(test_loader, classifier, scaler) if args.use_test else 0.0
else:
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        model.train()
        for x, labels in tqdm(train_loader):
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = objective(output, labels)
            loss.backward()
            optimizer.step()

        valid_acc = test(valid_loader)
        test_acc = test(test_loader) if args.use_test else 0.0

        Path(main_folder).mkdir(parents=True, exist_ok=True)
        if args.no_friction:
            f = open(f'{main_folder}/motionHAR_log_no_friction.txt', 'a')
        else:
            f = open(f'{main_folder}/motionHAR_log.txt', 'a')

        f.write('valid accuracy: ' + str(round(valid_acc, 2)) + '\n')
        f.write('test accuracy: ' + str(round(test_acc, 2)) + '\n')
        f.close()
        print(f"Valid accuracy: ", valid_acc)
        print(f"Test accuracy: ", test_acc)

        if (epoch+1) % 100 == 0:
            args.lr /= 10.
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

if args.lstm:
    f = open(f'{main_folder}/motionHAR_log_lstm.txt', 'a')
elif args.no_friction and (not args.esn): # coRNN without friction
    f = open(f'{main_folder}/motionHAR_log_no_friction.txt', 'a')
elif args.esn and args.no_friction: # coESN
    f = open(f'{main_folder}/motionHAR_log_coESN.txt', 'a')
elif args.esn: # ESN
    f = open(f'{main_folder}/motionHAR_log_esn.txt', 'a')
else: # original coRNN
    f = open(f'{main_folder}/motionHAR_log.txt', 'a')
ar = ''
for k, v in vars(args).items():
    ar += f'{str(k)}: {str(v)}, '
ar += f'valid: {str(round(valid_acc, 2))}, test: {str(round(test_acc, 2))}'
f.write(ar + '\n')
f.write('**************\n\n\n')
f.close()
