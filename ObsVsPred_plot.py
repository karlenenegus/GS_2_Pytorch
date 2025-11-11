import numpy as np
import argparse
import os
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Generate Observed vs Predicted Plots')
parser.add_argument('--results_file', type=str, default='./Output/Validation_output.npy', 
                    help='Path to validation output results file (default: ./Output/Validation_output.npy)')
parser.add_argument('--metadata_file', type=str, default='./Output/metadata.npy', 
                    help='Path to metadata file (default: ./Output/metadata.npy)')
parser.add_argument('--output_dir', type=str, default='./Output', 
                    help='Directory for output plots (default: ./Output)')
parser.add_argument('--title', type=str, default='Grain Length', 
                    help='Title for the plots (default: Grain Length)')
parser.add_argument('--folds', type=int, default=None, 
                    help='Number of folds (default: None, auto-detect from results)')
args = parser.parse_args()

plt.ioff()

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

results = np.load(args.results_file, allow_pickle=True)
metadata = np.load(args.metadata_file, allow_pickle=True)
results = results.item()
metadata = metadata.item()

epoch_num = metadata.get('epoch_num', metadata.get('epochs', 1))
batch_size = metadata['batch_size']
folds_meta = metadata.get('folds', args.folds)

# Auto-detect number of folds if not specified
if args.folds is None:
    # Try to detect from results structure
    if isinstance(results, dict):
        args.folds = len([k for k in results.keys() if isinstance(k, int)])
    else:
        args.folds = 10  # Default fallback
else:
    args.folds = args.folds

pcc = []
for i in range(args.folds):
    if i in results:
        pcc.append(pearsonr(results[i]["targets"], results[i]["outputs"])[0])
    else:
        print(f"Warning: Fold {i} not found in results")

plt_idx = np.where(abs(pcc - np.max(pcc)) == min(abs(pcc - np.max(pcc))))[0][0]

f, ax = plt.subplots()
plt.scatter(results[plt_idx]["targets"], results[plt_idx]["outputs"])
plt.text(.01, .99, f'r = {round(pcc[plt_idx], 4)} \nMAE val. loss = {round(results[plt_idx]["loss"], 4)}', ha='left', va='top', transform=ax.transAxes)
plt.text(.01, .89, f'epochs = {epoch_num} \nbatch size = {batch_size} \nfolds = {folds_meta}', ha='left', va='top', transform=ax.transAxes)
plt.ylabel("Predicted")
plt.xlabel("Observed")
plt.title(args.title)
plt.savefig(os.path.join(args.output_dir, 'obs_pred_max.png'))

plt_idx = np.where(abs(pcc - np.median(pcc)) == min(abs(pcc - np.median(pcc))))[0][0]

f, ax = plt.subplots()
plt.scatter(results[plt_idx]["targets"], results[plt_idx]["outputs"])
plt.text(.01, .99, f'r = {round(pcc[plt_idx], 4)} \nMAE val. loss = {round(results[plt_idx]["loss"], 4)}', ha='left', va='top', transform=ax.transAxes)
plt.text(.01, .89, f'epochs = {epoch_num} \nbatch size = {batch_size} \nfolds = {folds_meta}', ha='left', va='top', transform=ax.transAxes)
plt.ylabel("Predicted")
plt.xlabel("Observed")
plt.title(args.title)
plt.savefig(os.path.join(args.output_dir, 'obs_pred_median.png'))

plt_idx = np.where(abs(pcc - np.mean(pcc)) == min(abs(pcc - np.mean(pcc))))[0][0]

f, ax = plt.subplots()
plt.scatter(results[plt_idx]["targets"], results[plt_idx]["outputs"])
plt.text(.01, .99, f'K-fold mean r = {round(np.mean(pcc), 4)} \nr = {round(pcc[plt_idx], 4)} \nMAE val. loss = {round(results[plt_idx]["loss"], 4)}', ha='left', va='top', transform=ax.transAxes)
plt.text(.01, .86, f'epochs = {epoch_num} \nbatch size = {batch_size} \nfolds = {folds_meta}', ha='left', va='top', transform=ax.transAxes)
plt.ylabel("Predicted")
plt.xlabel("Observed")
plt.title(args.title)
plt.savefig(os.path.join(args.output_dir, 'obs_pred_mean.png'))

############### Loss plot 

loss_curve_file = os.path.join(os.path.dirname(args.results_file), 'Loss_curve.npy')
if not os.path.exists(loss_curve_file):
    loss_curve_file = os.path.join(args.output_dir, 'Loss_curve.npy')

if os.path.exists(loss_curve_file):
    Loss_curve = np.load(loss_curve_file, allow_pickle=True)
    Loss_curve = Loss_curve.item()
    
    test = np.asarray(Loss_curve[plt_idx]['testing'])
    train = np.asarray(Loss_curve[plt_idx]['training'])
    epoch_num_loss = metadata.get('epoch_num', metadata.get('epochs', len(test)))
    batch_size_loss = metadata['batch_size']
    folds_loss = metadata.get('folds', args.folds)
    epoch = list(range(0, epoch_num_loss, 1))
    
    f, ax = plt.subplots()
    plt.plot(epoch, test, c = "red", label="Validation")
    plt.plot(epoch, train, c = "black", label="Training")
    plt.xlabel("Epoch")
    plt.ylabel("MAE Loss")
    plt.legend(loc="upper right")
    plt.text(.01, .99, f'epochs = {epoch_num_loss} \nbatch size = {batch_size_loss} \nfolds = {folds_loss}', ha='left', va='top', transform=ax.transAxes)
    
    plt.savefig(os.path.join(args.output_dir, 'Loss.png'))
    print(f"Loss plot saved to {os.path.join(args.output_dir, 'Loss.png')}")
else:
    print(f"Warning: Loss curve file not found at {loss_curve_file}. Skipping loss plot.")

print(f"Plots saved to {args.output_dir}")