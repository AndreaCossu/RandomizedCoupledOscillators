import numpy as np
import torch.nn.utils
import argparse
from esn import DeepReservoir
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from utils import get_mackey_glass, coESN, check


parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--n_hid', type=int, default=256,
                    help='hidden size of recurrent net')
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
parser.add_argument('--inp_scaling', type=float, default=1.,
                    help='ESN input scaling')
parser.add_argument('--rho', type=float, default=0.99,
                    help='ESN spectral radius')
parser.add_argument('--leaky', type=float, default=1.0,
                    help='ESN spectral radius')
parser.add_argument('--alpha', type=float, default=0.0)
parser.add_argument('--lag', type=int, default=1)
parser.add_argument('--use_test', action="store_true")


args = parser.parse_args()
print(args)

main_folder = 'result'

device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
print("Using device ", device)
n_inp = 1
n_out = 1
washout = 200
lag = args.lag

gamma = (args.gamma - args.gamma_range / 2., args.gamma + args.gamma_range / 2.)
epsilon = (args.epsilon - args.epsilon_range / 2., args.epsilon + args.epsilon_range / 2.)

if not args.no_friction:
    model = DeepReservoir(n_inp, tot_units=args.n_hid, spectral_radius=args.rho,
                          input_scaling=args.inp_scaling,
                          connectivity_recurrent=args.n_hid,
                          connectivity_input=args.n_hid, leaky=args.leaky).to(device)
else:

    model = coESN(n_inp, args.n_hid, args.dt, gamma, epsilon, args.rho,
                  args.inp_scaling, device=device).to(device)
    if args.check:
        check_passed = check(model)
        print("Check: ", check_passed)
        if not check_passed:
            raise ValueError("Check not passed.")

(train_dataset, train_target), (valid_dataset, valid_target), (test_dataset, test_target) = get_mackey_glass(args.lag)

@torch.no_grad()
def test_esn(dataset, target, classifier, scaler):
    dataset = dataset.reshape(1, -1, 1).to(device)
    target = target.reshape(-1, 1).numpy()
    activations = model(dataset)[0].cpu().numpy()
    activations = activations[:, washout:]
    activations = activations.reshape(-1, args.n_hid)
    activations = scaler.transform(activations)
    predictions = classifier.predict(activations)
    mse = np.mean(np.square(predictions - target))
    rmse = np.sqrt(mse)
    norm = np.sqrt(np.square(target).mean())
    nrmse = rmse / (norm + 1e-9)
    return nrmse

dataset = train_dataset.reshape(1, -1, 1).to(device)
target = train_target.reshape(-1, 1).numpy()
activations = model(dataset)[0].cpu().numpy()
activations = activations[:, washout:]
activations = activations.reshape(-1, args.n_hid)
scaler = preprocessing.StandardScaler().fit(activations)
activations = scaler.transform(activations)
classifier = Ridge(alpha=args.alpha, max_iter=1000).fit(activations, target)
valid_nmse = test_esn(valid_dataset, valid_target, classifier, scaler)
test_nmse = test_esn(test_dataset, test_target, classifier, scaler) if args.use_test else 0.0

if args.no_friction: # coESN
    f = open(f'{main_folder}/mackey_log_coESN.txt', 'a')
else: # ESN
    f = open(f'{main_folder}/mackey_log_esn.txt', 'a')
ar = ''
for k, v in vars(args).items():
    ar += f'{str(k)}: {str(v)}, '
ar += f'valid: {str(round(valid_nmse, 5))}, test: {str(round(test_nmse, 5))}'
f.write(ar + '\n')
f.write('**************\n\n\n')
f.close()