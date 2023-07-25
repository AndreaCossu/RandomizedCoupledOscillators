import torch
import argparse
from utils import load_har, coRNN, check, coESN
from esn import DeepReservoir
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--root', type=str, default='data/har',
                    help='root directory of the dataset')
parser.add_argument('--n_hid', type=int, default=64,
                    help='hidden size of recurrent net')
parser.add_argument('--epochs', type=int, default=250,
                    help='max epochs')
parser.add_argument('--batch', type=int, default=64,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.017,
                    help='learning rate')
parser.add_argument('--dt', type=float, default=0.1,
                    help='step size <dt> of the coRNN')
parser.add_argument('--gamma', type=float, default=0.2,
                    help='y controle parameter <gamma> of the coRNN')
parser.add_argument('--epsilon', type=float, default=6.4,
                    help='z controle parameter <epsilon> of the coRNN')
parser.add_argument('--gamma_range', type=float, default=1,
                    help='y controle parameter <gamma> of the coRNN')
parser.add_argument('--epsilon_range', type=float, default=1,
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
parser.add_argument('--use_test', action="store_true")


args = parser.parse_args()
print(args)

n_inp = 9
n_out = 2

main_folder = 'result'

device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
gamma = (args.gamma - args.gamma_range / 2., args.gamma + args.gamma_range / 2.)
epsilon = (args.epsilon - args.epsilon_range / 2., args.epsilon + args.epsilon_range / 2.)

if args.esn and not args.no_friction:
    model = DeepReservoir(n_inp, tot_units=args.n_hid, spectral_radius=args.rho,
                          input_scaling=args.inp_scaling,
                          connectivity_recurrent=args.n_hid,
                          connectivity_input=args.n_hid, leaky=args.leaky).to(device)
elif args.no_friction and args.esn:
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

train_dataset, val_dataset, test_dataset = load_har(args.root)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch,
                                           shuffle=True, drop_last=False)
valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch,
                                           shuffle=False, drop_last=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch,
                                          shuffle=False, drop_last=False)

objective = torch.nn.CrossEntropyLoss()

def binary_accuracy(preds, y):
    preds = torch.argmax(preds, dim=1)
    correct = (preds == y).float()  # convert into float for division
    acc = correct.sum() / float(len(correct))
    return acc.item()

@torch.no_grad()
def test(dataloader):
    epoch_acc = 0
    model.eval()
    for i, (data, labels) in enumerate(dataloader):
        predictions = model(data.to(device))
        epoch_acc += binary_accuracy(predictions.cpu(), labels)
    return epoch_acc / float(len(dataloader))

@torch.no_grad()
def test_esn(dataloader, scaler, classifier):
    outputs, labels = [], []
    for data, l in dataloader:
        output = model(data.to(device))[-1][0]
        outputs.append(output.cpu())
        labels.append(l)
    outputs = torch.cat(outputs, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    activations = scaler.transform(outputs)
    return classifier.score(activations, labels)


train_accs, val_accs, test_accs = [], [], []
if args.esn:
    outputs, labels = [], []
    for data, l in tqdm(train_loader):
        output = model(data.to(device))[-1][0]
        labels.append(l)
        outputs.append(output.cpu())
    outputs = torch.cat(outputs, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    scaler = preprocessing.StandardScaler().fit(outputs)
    activations = scaler.transform(outputs)
    classifier = LogisticRegression(max_iter=1000).fit(activations, labels)
    acc = classifier.score(activations, labels)
    eval_acc = test_esn(valid_loader, scaler, classifier)
    test_acc = test_esn(test_loader, scaler, classifier) if args.use_test else 0.0

    train_accs.append(acc)
    val_accs.append(eval_acc)
    test_accs.append(test_acc)
else:
    for epoch in range(args.epochs):
        epoch_acc = 0.
        model.train()
        for data, labels in tqdm(train_loader):
            optimizer.zero_grad()
            output = model(data.to(device))
            loss = objective(output, labels.to(device))
            loss.backward()
            optimizer.step()
            epoch_acc += binary_accuracy(output.cpu(), labels)
        epoch_acc /= float(len(train_loader))

        eval_acc = test(valid_loader)
        test_acc = test(test_loader) if args.use_test else 0.0

        train_accs.append(epoch_acc)
        val_accs.append(eval_acc)
        test_accs.append(test_acc)


if args.no_friction and args.esn: # coESN
    f = open(f'{main_folder}/har_log_coESN.txt', 'a')
elif args.no_friction: # coRNN
    f = open(f'{main_folder}/har_log_hcornn.txt', 'a')
elif args.esn: # ESN
    f = open(f'{main_folder}/har_log_esn.txt', 'a')
else:
    f = open(f'{main_folder}/har_log_cornn.txt', 'a')

ar = ''
for k, v in vars(args).items():
    ar += f'{str(k)}: {str(v)}, '
ar += f'train_list: {[str(round(el, 5)) for el in train_accs]}, valid_list: {[str(round(el, 5)) for el in val_accs]}, test_list: {[str(round(el, 5)) for el in test_accs]} train: {str(round(max(train_accs), 5))} valid: {str(round(max(val_accs), 5))}, test: {str(round(max(test_accs), 5))}'
f.write(ar + '\n')
f.write('**************\n\n\n')
f.close()