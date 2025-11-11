import torch
import torch.nn as nn
import argparse
from DeepGS_model import deepGS
from DNNGP_model import DNNGP
from Train import train


parser = argparse.ArgumentParser(description='K-fold Cross-Validation Training')
parser.add_argument('--folds', type=int, default=10, help='Number of k-fold splits (default: 10)')
parser.add_argument('--epochs', type=int, default=400, help='Number of training epochs (default: 400)')
parser.add_argument('--batch_size', type=int, default=17, help='Batch size (default: 17)')
parser.add_argument('--PCs', type=int, default=1691, help='Number of principal components (default: 1691)')
parser.add_argument('--model_type', type=str, default='DNNGP', choices=['DNNGP', 'DeepGS'], 
                    help='Model type: DNNGP or DeepGS (default: DNNGP)')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate (default: 0.00001)')
parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay (default: 0.00001)')
parser.add_argument('--input_file', type=str, default='./Output/DNNGP_input.pt', 
                    help='Input data file path (default: ./Output/DNNGP_input.pt)')

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

input_data = torch.load(args.input_file)

folds = args.folds
epochs = args.epochs
batch_size = args.batch_size
loss_fn = nn.L1Loss()
PCs = args.PCs
model_type = args.model_type

if model_type == "DNNGP":
    network = DNNGP(input_size = PCs)
elif model_type == "DeepGS":
    network = deepGS(input_size = PCs)
else:
    raise ValueError(f"Invalid model type: {model_type}")

network = nn.DataParallel(network)
network = network.to(device)

optimizer = torch.optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

train(network=network, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs, input_data=input_data, 
        batch_size=batch_size, folds=folds, device=device, PCs=PCs)
