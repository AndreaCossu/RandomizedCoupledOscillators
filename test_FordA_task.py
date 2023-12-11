from torch import nn, optim
import torch
import torch.nn.utils
from utils import coRNN, coESN, check, LSTM, get_FordA_data, TrainedRON
from pathlib import Path
import argparse
from tqdm import tqdm
from esn import DeepReservoir
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import numpy as np

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--n_hid', type=int, default=100,
                    help='hidden size of recurrent net')
parser.add_argument('--epochs', type=int, default=120, 
                    help='max epochs')
parser.add_argument('--batch', type=int, default=120, 
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0021,
                    help='learning rate')
parser.add_argument('--dt', type=float, default=0.042,
                    help='step size <dt> of the coRNN')
parser.add_argument('--gamma', type=float, default=2.7,
                    help='y controle parameter <gamma> of the coRNN')
parser.add_argument('--epsilon', type=float, default=4.7,
                    help='z controle parameter <epsilon> of the coRNN')
parser.add_argument('--gamma_range', type=float, default=2.7,
                    help='y controle parameter <gamma> of the coRNN')
parser.add_argument('--epsilon_range', type=float, default=4.7,
                    help='z controle parameter <epsilon> of the coRNN')
parser.add_argument('--cpu', action="store_true")
parser.add_argument('--check', action="store_true")
parser.add_argument('--no_friction', action="store_true", help="remove friction term inside non-linearity")
parser.add_argument('--esn', action="store_true")
parser.add_argument('--inp_scaling', type=float, default=1.,
                    help='ESN input scaling')
parser.add_argument('--rho', type=float, default=0.99,
                    help='ESN spectral radius')
parser.add_argument('--leaky', type=float, default=1.0,
                    help='ESN spectral radius')
parser.add_argument('--use_test', action="store_true")
parser.add_argument('--test_trials', type=int, default=5,
                    help='number of trials to compute mean and std on test')
parser.add_argument('--lstm', action="store_true", help="use LSTM")
parser.add_argument('--trained_ron', action="store_true", help="use LSTM")


main_folder = 'result'
args = parser.parse_args()
print(args)

device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")

def test(data_loader):
    model.eval()
    correct = 0
    test_loss = 0
    nanflag = False
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)

            output = model(x)
            test_loss += objective(output, y).item()
            pred = output.data.max(1, keepdim=True)[1]
            ytarg = y.data.max(1, keepdim=True)[1] # added
            #correct += pred.eq(y.data.view_as(pred)).sum()
            correct += pred.eq(ytarg.data.view_as(pred)).sum() # changed
    test_loss /= i+1
    print(test_loss)
    accuracy = 100. * correct / len(data_loader.dataset)
    if torch.isnan(torch.Tensor([test_loss])) or test_loss>100000:
        nanflag = True
    return accuracy.item(), nanflag

@torch.no_grad()
def test_esn(data_loader, classifier, scaler):
    activations, ys = [], []
    for x, y in tqdm(data_loader):
        x = x.to(device)
        output = model(x)[-1][0]
        activations.append(output.cpu())
        ys.append(y)
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)

n_inp = 1
n_out = 2 # classes
bs_test = 120
gamma = (args.gamma - args.gamma_range / 2., args.gamma + args.gamma_range / 2.)
epsilon = (args.epsilon - args.epsilon_range / 2., args.epsilon + args.epsilon_range / 2.)

max_test_accs = []
if args.test_trials > 1:
    main_folder = 'result'
    if args.esn:
        train_loader, valid_loader, test_loader = get_FordA_data(args.batch,bs_test, whole_train=True)
    else:
        train_loader, valid_loader, test_loader = get_FordA_data(args.batch,bs_test, whole_train=True, RC=False)
else:
    if args.esn:
        train_loader, valid_loader, test_loader = get_FordA_data(args.batch,bs_test)
    else:
        train_loader, valid_loader, test_loader = get_FordA_data(args.batch,bs_test, RC=False)
        

for trial in range(args.test_trials):
    accs = []


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

    elif args.trained_ron:
        model = TrainedRON(n_inp, args.n_hid, n_out, args.dt, gamma, epsilon, args.rho,
                           args.inp_scaling, device=device).to(device)
    else:
        model = coRNN(n_inp, args.n_hid, n_out,args.dt,gamma,epsilon,
                    no_friction=args.no_friction, device=device).to(device)

    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.esn:
        activations, ys = [], []
        for x, y in tqdm(train_loader):
            x = x.to(device)
            output = model(x)[-1][0]
            activations.append(output.cpu())
            ys.append(y)
        activations = torch.cat(activations, dim=0).numpy()
        ys = torch.cat(ys, dim=0).numpy()
        scaler = preprocessing.StandardScaler().fit(activations)
        activations = scaler.transform(activations)
        classifier = LogisticRegression(max_iter=1000).fit(activations, ys)
        valid_acc = test_esn(valid_loader, classifier, scaler) if args.test_trials<=1 else 0.0
        test_acc = test_esn(test_loader, classifier, scaler) if args.use_test else 0.0
        accs.append(test_acc)
    else:
        for epoch in range(args.epochs):
            print(f"Epoch {epoch}")
            model.train()
            for x, y in tqdm(train_loader):
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                output = model(x)
                loss = objective(output, y)
                loss.backward()
                optimizer.step()

            if args.test_trials<=1:
                valid_acc, nanflag = test(valid_loader)
            else:
                valid_acc, nanflag = 0.0, False
            if args.use_test:
                test_acc, nanflag = test(test_loader) 
            else:
                test_acc = 0.0
            accs.append(test_acc)
            if nanflag:
                break # I don't want to run many epochs if I get nan values early...
            Path(main_folder).mkdir(parents=True, exist_ok=True)
            if args.lstm:
                f = open(f'{main_folder}/FordA_log_lstm.txt', 'a')
            elif args.trained_ron:
                f = open(f'{main_folder}/FordA_log_trained_ron.txt', 'a')
            elif args.no_friction:
                f = open(f'{main_folder}/FordA_log_no_friction.txt', 'a')
            else:
                f = open(f'{main_folder}/FordA_log.txt', 'a')

            f.write('valid accuracy: ' + str(round(valid_acc, 4)) + '\n')
            f.write('test accuracy: ' + str(round(test_acc, 4)) + '\n')
            f.close()
            print(f"Valid accuracy: ", valid_acc)
            print(f"Test accuracy: ", test_acc)

            if (epoch+1) % 100 == 0:
                scaled_lr = args.lr / 10.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = scaled_lr

    if args.lstm:
        f = open(f'{main_folder}/FordA_log_lstm.txt', 'a') 
    elif args.no_friction and (not args.esn): # coRNN without friction
        f = open(f'{main_folder}/FordA_log_no_friction.txt', 'a')
    elif args.esn and args.no_friction: # coESN
        f = open(f'{main_folder}/FordA_log_coESN.txt', 'a')
    elif args.esn: # ESN
        f = open(f'{main_folder}/FordA_log_esn.txt', 'a')
    elif args.trained_ron:
        f = open(f'{main_folder}/FordA_log_trained_ron.txt', 'a')
    else: # original coRNN
        f = open(f'{main_folder}/FordA_log.txt', 'a')
    ar = ''
    for k, v in vars(args).items():
        ar += f'{str(k)}: {str(v)}, '
    ar += f'valid: {str(round(valid_acc, 4))}, test: {str(round(test_acc, 4))}'
    f.write(ar + '\n')
    f.write('**************\n\n\n')
    f.close()

    max_test_accs.append(max(accs))

mean_test = np.mean(np.array(max_test_accs))
std_test = np.std(np.array(max_test_accs))

if args.lstm:
    f = open(f'{main_folder}/FordA_log_lstm.txt', 'a')
elif args.no_friction and (not args.esn): # coRNN without friction
    f = open(f'{main_folder}/FordA_log_no_friction.txt', 'a')
elif args.trained_ron:
    f = open(f'{main_folder}/FordA_log_trained_ron.txt', 'a')
elif args.esn and args.no_friction: # coESN
    f = open(f'{main_folder}/FordA_log_coESN.txt', 'a')
elif args.esn: # ESN
    f = open(f'{main_folder}/FordA_log_esn.txt', 'a')
else: # original coRNN
    f = open(f'{main_folder}/FordA_log.txt', 'a')
ar = f'List of maximum test accuracies: {str(max_test_accs)}'
f.write(ar + '\n')
ar = f'Mean test accuracy: {str(round(mean_test, 4))}, Std test accuracy: {str(round(std_test, 4))}'
f.write(ar + '\n')
f.write('**************\n\n\n')
f.close()