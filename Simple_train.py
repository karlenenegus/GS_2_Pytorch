import torch
import torch.nn as nn
import argparse
from DeepGS_model import deepGS
from Train import train
from DNNGP_model import DNNGP

parser = argparse.ArgumentParser(description='Simple Training with Validation Split')
parser.add_argument('--input_file', type=str, default='./Output/DNNGP_input.pt', 
                    help='Input data file path (default: ./Output/DNNGP_input.pt)')
parser.add_argument('--validation_split', type=float, default=0.10, 
                    help='Validation split ratio (default: 0.10)')
parser.add_argument('--model_type', type=str, default='DNNGP', choices=['DNNGP', 'DeepGS'], 
                    help='Model type: DNNGP or DeepGS (default: DNNGP)')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs (default: 20)')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size (default: 20)')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay (default: 0.00001)')
parser.add_argument('--PCs', type=int, default=None, 
                    help='Number of principal components (default: None, uses input_data.shape[2] - 1)')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load input data
input_data = torch.load(args.input_file)
validation_split = args.validation_split

# Determine input size
if args.PCs is not None:
    input_size = args.PCs
else:
    input_size = input_data.shape[2] - 1

# Set training parameters
# Choose model type: DNNGP or deepGS
if args.model_type == "DNNGP":
    model = DNNGP(input_size=input_size)
elif args.model_type == "DeepGS":
    model = deepGS(input_size=input_size)
else:
    raise ValueError(f"Invalid model type: {args.model_type}")

model = nn.DataParallel(model)
network = model.to(device)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
epochs = args.epochs
batch_size = args.batch_size

train(network=network, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs, input_data=input_data, 
      batch_size=batch_size, validation_split=validation_split, device=device, PCs=args.PCs)