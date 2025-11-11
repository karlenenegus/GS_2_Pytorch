import torch
import time
import copy
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import random

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

def basic_train(network, loss_fn, optimizer, epochs, trainloader, testloader, batch_size, device, folds, PCs):
    
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
            y_pred = network(X_train)
            loss = loss_fn(y_pred.squeeze(1), y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        with torch.inference_mode():
            for i, data2 in enumerate(testloader, 0):
                inputs, targets = data2[:, :, 0:-1].to(device), data2[:, :, -1].to(device)
                network = network.eval()
                outputs = network(inputs)
                loss_value = loss_fn(outputs.squeeze(1), targets.type(torch.float))
                
                output_array.append(outputs.cpu().numpy().squeeze(1).squeeze(1))
                input_array.append(inputs.cpu().numpy().squeeze(1))
                target_array.append(targets.cpu().numpy().squeeze(1))
                loss_array.append(loss_value.cpu().item())
        
        train_loss_values.append(loss.cpu().detach().numpy())
        test_loss_values.append(loss_value.cpu().detach().numpy())
        
        if epoch % 5 == 0:
                epoch_count.append(epoch)
                print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {loss_value} ")
        
    output_array1 = output_array[epochs-1]
    target_array1 = target_array[epochs-1]
    input_array1 = input_array[epochs-1]
    loss_array1 = loss_array[epochs-1]
    metadata = {'batch_size': batch_size, 'epoch_num': epochs, 'folds': folds, 'PCs': PCs}
    
    return output_array1, target_array1, input_array1, loss_array1, train_loss_values, test_loss_values, metadata

def PCA_preprocessing(input_data, n_components, train_index, test_index):
    
    X_train_preprocessing = input_data[train_index, :, :-1]
    X_train_preprocessing = X_train_preprocessing.squeeze(1)
    xscaler = StandardScaler()
    xscaler.fit(X_train_preprocessing)
    X_train_scaled = xscaler.transform(X_train_preprocessing)
    
    pca = PCA(n_components = n_components).fit(X_train_scaled)
    X_train_pcs = pca.transform(X_train_scaled)
    
    X_valid_preprocessing = input_data[test_index, :, :-1]
    X_valid_preprocessing = X_valid_preprocessing.squeeze(1)
    X_valid_scaled = xscaler.transform(X_valid_preprocessing)
    
    X_valid_pcs = pca.transform(X_valid_scaled)
    
    PCA_data = np.empty((input_data.squeeze(1).shape[0], X_train_pcs.shape[1]))
    PCA_data[train_index] = X_train_pcs
    PCA_data[test_index] = X_valid_pcs
    PCA_data = pd.DataFrame(PCA_data)
    PCA_data["trait"] = input_data[:,:, -1].squeeze(1)
    PCA_data = torch.tensor(PCA_data.to_numpy(), dtype=torch.float).unsqueeze(1)
    
    return PCA_data

def pheno_preprocessing(input_data, train_index, test_index):
    pheno_data = np.empty((input_data.squeeze(1).shape[0], 1))
    y_train_preprocessing = input_data[train_index, :, -1]
    #y_train_preprocessing = y_train_preprocessing.squeeze(1)
    yscaler = StandardScaler()
    yscaler.fit(y_train_preprocessing)
    y_train_scaled = yscaler.transform(y_train_preprocessing)
    pheno_data[train_index] = y_train_scaled
    
    y_test_preprocessing = input_data[test_index, :, -1]
    #y_test_preprocessing = y_test_preprocessing.squeeze(1)
    y_test_scaled = yscaler.transform(y_test_preprocessing)
    pheno_data[test_index] = y_test_scaled
    input_data[:, :, -1] = torch.tensor(pheno_data, dtype=torch.float)
    return input_data

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

def train(network, loss_fn, optimizer, epochs, input_data, batch_size, device, folds=None, validation_split=None, PCs=None):
    startTime = time.time()
    kwargs = {'num_workers': 1, 'pin_memory': True} if (device == "cuda") else {}
        
    if isinstance(folds, int):
        kfold = KFold(n_splits=folds, shuffle=True)

        results = {}
        loss_curve = {}
        for fold, (train_index, test_index) in enumerate(kfold.split(input_data)):
            print(f'Starting Fold: {fold+1}')
            
            network.apply(reset_weights)
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_index)
            
            input_data = pheno_preprocessing(input_data, train_index, test_index)
        
            if isinstance(PCs, (int, float)):
                PCA_data = PCA_preprocessing(input_data = input_data, n_components = PCs, train_index=train_index, test_index=test_index)
                trainloader = torch.utils.data.DataLoader(PCA_data, batch_size = batch_size, sampler = train_subsampler, **kwargs)
                testloader = torch.utils.data.DataLoader(PCA_data, batch_size = len(test_index), sampler = test_subsampler, **kwargs)
            else:
                PCA_data = None
                trainloader = torch.utils.data.DataLoader(input_data, batch_size = batch_size, sampler = train_subsampler, **kwargs)
                testloader = torch.utils.data.DataLoader(input_data, batch_size = len(test_index), sampler = test_subsampler, **kwargs)
            
            output_array1, target_array1, input_array1, loss_array1, train_loss_values, test_loss_values, metadata = basic_train(network, loss_fn, optimizer, epochs, trainloader, testloader, batch_size, device, folds, PCs)

            loss_curve[fold] = {"testing": test_loss_values, "training": train_loss_values}
            results[fold] = {"outputs": output_array1, "targets": target_array1, "loss": loss_array1, "input": input_array1, "PCA_input": PCA_data}
            save_path = f'./Output/Model-fold/model-fold-{fold+1}.pth'
            torch.save(network.state_dict(), save_path)
        
    elif folds == None:
        train_split = int((1-validation_split)* len(input_data))
        all_index = range(len(input_data))
        train_index = random.sample(all_index, train_split)
        test_index  = list(set(all_index) - set(train_index))

        input_data = pheno_preprocessing(input_data, train_index, test_index)
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_index)

        if isinstance(PCs, (int, float)):
            PCA_data = PCA_preprocessing(input_data = input_data, n_components = PCs, train_index = train_index, test_index=test_index)
            trainloader = torch.utils.data.DataLoader(PCA_data, batch_size = batch_size, sampler = train_subsampler, **kwargs)
            testloader = torch.utils.data.DataLoader(PCA_data, batch_size = len(test_index), sampler = test_subsampler, **kwargs)
        
        else:
            trainloader = torch.utils.data.DataLoader(input_data, batch_size = batch_size, sampler = train_subsampler, **kwargs)
            testloader = torch.utils.data.DataLoader(input_data, batch_size = len(test_index), sampler = test_subsampler, **kwargs)
        
        network.apply(reset_weights)
        
        output_array1, target_array1, input_array1, loss_array1, train_loss_values, test_loss_values, metadata = basic_train(network, loss_fn, optimizer, epochs, trainloader, testloader, batch_size, device, folds, PCs)

        loss_curve = {"testing": test_loss_values, "training": train_loss_values}
        results = {"outputs": output_array1, "targets": target_array1, "loss": loss_array1, "input": input_array1}
        save_path = f'./Output/Model-fold/model.pth'
        torch.save(network.state_dict(), save_path)        
    else:
        print("Error in fold number")       
    
    np.save('./Output/Validation_output.npy', results)
    np.save('./Output/Loss_curve.npy', loss_curve)
    np.save('./Output/metadata.npy', metadata)
    
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))