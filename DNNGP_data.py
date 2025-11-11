import pandas as pd
import numpy as np
import torch
import argparse
import os

parser = argparse.ArgumentParser(description='DNNGP Data Preprocessing')
parser.add_argument('--data_dir', type=str, default='./Data', 
                    help='Directory containing input data files (default: ./Data)')
parser.add_argument('--output_dir', type=str, default='./Output', 
                    help='Directory for output files (default: ./Output)')
parser.add_argument('--snp_start_col', type=int, default=20000, 
                    help='Starting column index for SNP data subset (default: 20000)')
parser.add_argument('--snp_end_col', type=int, default=26280, 
                    help='Ending column index for SNP data subset (default: 26280)')
parser.add_argument('--min_allele_freq', type=float, default=0.05, 
                    help='Minimum allele frequency threshold (default: 0.05)')
parser.add_argument('--max_allele_freq', type=float, default=0.95, 
                    help='Maximum allele frequency threshold (default: 0.95)')
parser.add_argument('--max_geno_freq', type=float, default=0.95, 
                    help='Maximum genotype frequency threshold (default: 0.95)')
args = parser.parse_args()


# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Data - Iranian Phenotypes
## Data - days to maturity and days to heading
iranian_dth = pd.read_excel(os.path.join(args.data_dir, 'PHENOTYPIC DATA IRANIAN', 'Iranian DTH - DTM - D H.xlsx'))
# Fix column names
iranian_dth.columns = ['GID', 'DTM(Heat)', 'DTH(Heat)', 'DTM(Drought)', 'DTH(Drought)']
iranian_dth = iranian_dth.tail(-1)
iranian_dth = iranian_dth.sort_values('GID')
# Replace missing data with NA
pd.set_option('future.no_silent_downcasting', True)
iranian_dth = iranian_dth.replace("\W", np.nan, regex=True).dropna(axis=0, thresh=2)
# Drop any duplicated rows
iranian_dth = iranian_dth.where(iranian_dth.duplicated(keep=False)==False).dropna(axis=0, thresh = 2)

## Data - plant height
iranian_pht = pd.read_excel(os.path.join(args.data_dir, 'PHENOTYPIC DATA IRANIAN', 'Iranian PHT-D.xlsx'))
# Fix column names
iranian_pht.columns = ['GID', 'PHT(Drought)']
iranian_pht = iranian_pht.tail(-1)
iranian_pht = iranian_pht.sort_values('GID')
# Replace missing data with NA
iranian_pht = iranian_pht.replace("\W", np.nan, regex=True).dropna(axis=0, thresh=2)
# Drop any duplicated rows
iranian_pht = iranian_pht.where(iranian_pht.duplicated(keep=False)==False).dropna(axis=0, thresh = 2)

iranian_qt = pd.read_excel(os.path.join(args.data_dir, 'PHENOTYPIC DATA IRANIAN', 'Iranian QUALITY.xlsx'))
# Names are good; just sort by genotype ID
iranian_qt = iranian_qt.sort_values('GID')
# Replace missing data with NA
iranian_qt = iranian_qt.replace("\W", np.nan, regex=True).dropna(axis=0, thresh=2)
# Drop any duplicated rows
iranian_qt = iranian_qt.where(iranian_qt.duplicated(keep=False)==False).dropna(axis=0, thresh = 2)

#Join all the phenotype files
iranian_pheno = pd.merge(iranian_dth, iranian_pht, on='GID', how='outer')
iranian_pheno = pd.merge(iranian_pheno, iranian_qt, on='GID', how = 'outer')

###############################################################################
# SNP Data
iranian_snps_complete = pd.read_csv(os.path.join(args.data_dir, "Iranian_Samples.csv"), dtype='O')

# Subset for memory constraints
iranian_snps = iranian_snps_complete.iloc[:, :]

# Split file into data types
snp_meta = iranian_snps.iloc[7:,:17]
geno_meta = iranian_snps.iloc[:7,17:]
snp_data = iranian_snps.iloc[7:-1, 17:]

# Replace nans
snp_data = snp_data.replace("-", float("nan"))

#Remove suffix from GID names
GID_names = snp_data.columns.to_series().str.split(".", expand=True)[0].to_frame().T
#Filter SNP positions with least missing data; keep best 50%
nan_counts = snp_data.isna().sum().to_frame().T
nan_counts_rows = snp_data.T.isna().sum().to_frame()
nan_to_remove = nan_counts_rows < nan_counts_rows.quantile(q=0.5, axis = 0)

#Merge snp data back together
snp_data = pd.concat([GID_names, snp_data, nan_counts], ignore_index=True)

last_row = snp_data.index[-1]
first_row = snp_data.index[0]
sorted_genotypes = snp_data.sort_values(axis=1, by=[first_row, last_row])

#For duplicates: keep GID with least missing data
duplicate_genotypes = sorted_genotypes.T.duplicated(subset = sorted_genotypes.index[0])
unduplicated_genotypes = sorted_genotypes.loc[:, (duplicate_genotypes==False)]

# Fix index match original snp_data and snp_meta index
unduplicated_genotypes.index = (unduplicated_genotypes.index + 6)
# Replace column names with simplified GIDs
unduplicated_genotypes.columns = unduplicated_genotypes.iloc[0, :]
unduplicated_genotypes = unduplicated_genotypes.iloc[1:-1, :]

SNP = unduplicated_genotypes[nan_to_remove[0]]

############################################################
#Join Phenotypes and SNPs
iranian_pheno.index = iranian_pheno.iloc[:,0].astype("Int64").astype("str")
SNP.columns.name = "GID"
##Selecting only one phenotype column for input data
input_data = SNP.T.merge(iranian_pheno.loc[: ,'length'], how='inner', on = 'GID')
#Write to file
input_data.to_csv(os.path.join(args.output_dir, "Iranian_data.csv"))

###################################################
# Prep data for use with NN
input_data1 = pd.read_csv(os.path.join(args.output_dir, "Iranian_data.csv"))
input_data1b = input_data1.iloc[:, args.snp_start_col:args.snp_end_col]
snps = input_data1b.iloc[:, :-1]

geno_freq = []
for col in snps:
    geno_freq.append(max(snps[col].value_counts()/snps[col].value_counts().sum()))
    
snp_index = (np.array(geno_freq) < args.max_geno_freq)
snps = snps.loc[:, snp_index]

allele_freq = []
for col in snps:
    if (1.0 in snps[col].value_counts().index) & (2.0 in snps[col].value_counts().index):
        allele1 = snps[col].value_counts().get(1)
        allele2 = snps[col].value_counts().get(2)
        total_alleles = (snps[col].value_counts().sum())*2
        freq = (allele1 + 2*allele2)/total_alleles
        allele_freq.append(freq)
    elif 1.0 in snps[col].value_counts().index:
        allele1 = snps[col].value_counts().get(1)
        total_alleles = (snps[col].value_counts().sum())*2
        freq = (allele1)/total_alleles
        allele_freq.append(freq)
    elif (2.0 in snps[col].value_counts().index):
        allele2 = snps[col].value_counts().get(2)
        total_alleles = (snps[col].value_counts().sum())*2
        freq = (2*allele2)/total_alleles
        allele_freq.append(freq)
    else: 
        allele_freq.append(0)
    snps[col] = snps[col].replace(to_replace=np.nan, value = np.nanmean(snps[col]))

snp_index = (np.array(allele_freq) > args.min_allele_freq) & (np.array(allele_freq) < args.max_allele_freq)
new_snps = snps.loc[:, snp_index]

pheno = input_data1b.iloc[:, -1]
pheno = pheno.replace('nan', np.nan)
pheno = pd.DataFrame(pheno)
pheno = pheno.rename(columns={'length': "GL"})

new_dataset = new_snps.join(pheno)
new_dataset = new_dataset[((new_dataset['GL'] < 9) & (new_dataset['GL'].notnull()))]

new_dataset = torch.tensor(new_dataset.values, dtype=torch.float)
new_dataset = new_dataset.unsqueeze(1)

torch.save(new_dataset, os.path.join(args.output_dir, 'DNNGP_input.pt'))
print(f"Data preprocessing complete. Output saved to {os.path.join(args.output_dir, 'DNNGP_input.pt')}")
