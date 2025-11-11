import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from DNNGP_model import DNNGP
from sklearn.model_selection import KFold
from Train import reset_weights
from Train import pheno_preprocessing
from Train import PCA_preprocessing
import time
import copy

class Early_Stopping():
    def __init__(self, patience = 5, min_delta=0, restore_best_weight = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weight
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""
    def __call__(self, model, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.status = f"Stopped on {self.counter}"
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model.state_dict())
                return True
        self.status = f"{self.counter}/{self.patience}"
        return False

device = "cuda" if torch.cuda.is_available() else "cpu"

input_data = torch.load('./Output/DNNGP_input.pt')

folds = 10
epochs = 1000
batch_size = 25
loss_fn = nn.L1Loss()
PCs = 1691
num_iterations = 1  # Number of k-fold iterations to run (set to 100 for full experiment)

network = DNNGP(input_size = PCs)
network = nn.DataParallel(network)
network = network.to(device)

optimizer = torch.optim.SGD(network.parameters(), lr=0.0001, momentum=0.5, weight_decay=0.00001)

startTime = time.time()
kwargs = {'num_workers': 1, 'pin_memory': True} if (device == "cuda") else {}
kfold = KFold(n_splits=folds, shuffle=True)

for j in range(num_iterations):
    results = {}
    loss_curve = {}
    for fold, (train_index, test_index) in enumerate(kfold.split(input_data)):
        print(f'Starting Fold: {fold+1}')
        
        network.apply(reset_weights)
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_index)
        
        # Create a copy to avoid modifying the original input_data
        input_data_processed = pheno_preprocessing(input_data.clone(), train_index, test_index)

        if isinstance(PCs, (int, float)):
            PCA_data = PCA_preprocessing(input_data = input_data_processed, n_components = PCs, train_index=train_index, test_index=test_index)
            trainloader = torch.utils.data.DataLoader(PCA_data, batch_size = batch_size, sampler = train_subsampler, **kwargs)
            testloader = torch.utils.data.DataLoader(PCA_data, batch_size = len(test_index), sampler = test_subsampler, **kwargs)
        
        else:
            trainloader = torch.utils.data.DataLoader(input_data_processed, batch_size = batch_size, sampler = train_subsampler, **kwargs)
            testloader = torch.utils.data.DataLoader(input_data_processed, batch_size = len(test_index), sampler = test_subsampler, **kwargs)
        
        steps = len(list(trainloader))
        done = False

        es = Early_Stopping(patience=10)
        
        train_loss_values = []
        test_loss_values = []
        epoch_count = []
        output_array = []
        input_array = []
        target_array = []
        loss_array = []

        for epoch in range(epochs):
            for i, data1 in enumerate(trainloader, 0):
                X_train, y_train = data1[:, :, 0:-1].to(device), data1[:, :, -1].to(device)
                network = network.train()
                pred_train = network(X_train)
                loss = loss_fn(pred_train.squeeze(1), y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            with torch.inference_mode():
                for i, data2 in enumerate(testloader, 0):
                    inputs, targets = data2[:, :, 0:-1].to(device), data2[:, :, -1].to(device)
                    network = network.eval()
                    pred_val = network(inputs)
                    vloss = loss_fn(pred_val.squeeze(1), targets.type(torch.float))
                    if es(network, vloss): 
                        done = True
                    output_array.append(pred_val.cpu().numpy().squeeze(1).squeeze(1))
                    input_array.append(inputs.cpu().numpy().squeeze(1))
                    target_array.append(targets.cpu().numpy().squeeze(1))
                    loss_array.append(vloss.cpu().item())
            
            train_loss_values.append(loss.cpu().detach().numpy())
            test_loss_values.append(vloss.cpu().detach().numpy())
            
            if epoch % 5 == 0:
                epoch_count.append(epoch)
                print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Val. Loss: {vloss} | EStop: {es.status}")
            if (epoch > 1000) or done:
                break
        # Get results from the last epoch (or best epoch if early stopping occurred)
        final_epoch_idx = max(0, len(output_array) - 1)
        output_array1 = output_array[final_epoch_idx]
        target_array1 = target_array[final_epoch_idx]
        input_array1 = input_array[final_epoch_idx]
        loss_array1 = loss_array[final_epoch_idx]
        metadata = {'batch_size': batch_size, 'folds': folds, 'PCs': PCs}
        
        loss_curve[fold] = {"testing": test_loss_values, "training": train_loss_values}
        results[fold] = {"outputs": output_array1, "targets": target_array1, "loss": loss_array1, "input": input_array1, "PCA_input": PCA_data, "Epochs": epoch}
        save_path = f'./Output/Model-fold/model-fold-{j}-{fold+1}.pth'
        torch.save(network.state_dict(), save_path)

    loc_results = f"./Output/Validation_output{j}.npy"
    loc_results = "".join(loc_results)
    np.save(loc_results, results)

    loc_loss = f"./Output/Loss_curve{j}.npy"
    np.save(loc_loss, loss_curve)

    loc_meta = f"./Output/metadata{j}.npy"   
    np.save(loc_meta, metadata)

    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))