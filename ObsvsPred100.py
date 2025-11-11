import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

avg_mean = []
for i in range(1, 79):
	results = np.load(f'./Output/Validation_output{i}.npy', allow_pickle=True)
	metadata = np.load(f'./Output/metadata{i}.npy', allow_pickle=True)
	results = results.item()
	metadata = metadata.item()
	pcc = []
	epoch_all = []
	for j in range(10):
		pcc.append(pearsonr(results[j]["targets"], results[j]["outputs"])[0])
		epoch_all.append(results[j]["Epochs"])
	avg_mean.append(np.mean(pcc))
	print(np.mean(pcc))

avg_mean_all = np.mean(avg_mean)
print(avg_mean_all)
