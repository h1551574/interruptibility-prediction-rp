# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 11:40:41 2023

@author: alerr
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from seaborn import objects as so
from datetime import datetime



#%%

path = "replication-package/Data/Interruption Data (ANONYMIZED)/Aggregated Interruption Data.csv"
data = pd.read_csv(path)

#%% Correlation: Interruptibility vs. Interruption Lag
x = data['interruption_lag_in_seconds']
y = data[' interruptibility']

pearsonr(x, y)

#%% Correlation: Interruptibility vs. Disturbance
x = data[' disturbance']
y = data[' interruptibility']
pearsonr(x, y)

#%% Correlation: Interruptibility vs. Mental Load
x = data[' mental_workload']
y = data[' interruptibility']
pearsonr(x, y)

#%% prediction interval (interruptibility x interruption lag)

# Code based on:
# https://github.com/jtleek/replication_paper/blob/gh-pages/code/replication_analysis.Rmd
# Patil, P., Peng, R. D., & Leek, J. T. (2016).
# What Should Researchers Expect When They Replicate Studies?
# A Statistical View of Replicability in Psychological Science.
# Perspectives on Psychological Science, 11(4), 539–544.
# https://doi.org/10.1177/1745691616646366


x = data[' interruption_lag']
y = data[' interruptibility']
r_rep = pearsonr(x, y)

r_original_l = 0.382 # lab
fish_orig = np.arctanh(r_original_l)
n_orig_l = 72 # lab
n_rep = 82

se_total = np.sqrt(1/(n_orig_l-3) + 1/(n_rep-3))
low = np.tanh(fish_orig - se_total * 1.96)
high = np.tanh(fish_orig + se_total * 1.96)

print("(interruptibility x interruption lag):\n")
print("Lab:\n")
print("Original Effect Size (Lab): " + str(r_original_l))
print("PI Low: "+ str(low))
print("PI High: " + str(high))
print("Replicated Effect Size: " + str(r_rep.statistic))

r_original_f = 0.282 # field

fish_orig = np.arctanh(r_original_f)
n_orig_f = 139 # field
n_rep = 82

se_total = np.sqrt(1/(n_orig_f-3) + 1/(n_rep-3))
low = np.tanh(fish_orig - se_total * 1.96)
high = np.tanh(fish_orig + se_total * 1.96)

print("\n")


print("Field:\n")
print("Original Effect Size: (Field)" + str(r_original_f))
print("PI Low: "+ str(low))
print("PI High: " + str(high))
print("Replicated Effect Size: " + str(r_rep.statistic))

#%% prediction interval (interruptibility x mental load)

# Code based on:
# https://github.com/jtleek/replication_paper/blob/gh-pages/code/replication_analysis.Rmd
# Patil, P., Peng, R. D., & Leek, J. T. (2016).
# What Should Researchers Expect When They Replicate Studies?
# A Statistical View of Replicability in Psychological Science.
# Perspectives on Psychological Science, 11(4), 539–544.
# https://doi.org/10.1177/1745691616646366


x = data[' mental_workload']
y = data[' interruptibility']
r_rep = pearsonr(x, y)

r_original_l = 0.815 # lab
fish_orig = np.arctanh(r_original_l)
n_orig_l = 72 # lab
n_rep = 82

se_total = np.sqrt(1/(n_orig_l-3) + 1/(n_rep-3))
low = np.tanh(fish_orig - se_total * 1.96)
high = np.tanh(fish_orig + se_total * 1.96)

print("(interruptibility x mental load):\n")
print("Lab:\n")
print("Original Effect Size (Lab): " + str(r_original_l))
print("PI Low: "+ str(low))
print("PI High: " + str(high))
print("Replicated Effect Size: " + str(r_rep.statistic))

r_original_f = 0.702 # field

fish_orig = np.arctanh(r_original_f)
n_orig_f = 139 # field
n_rep = 82

se_total = np.sqrt(1/(n_orig_f-3) + 1/(n_rep-3))
low = np.tanh(fish_orig - se_total * 1.96)
high = np.tanh(fish_orig + se_total * 1.96)

print("\n")


print("Field:\n")
print("Original Effect Size: (Field)" + str(r_original_f))
print("PI Low: "+ str(low))
print("PI High: " + str(high))
print("Replicated Effect Size: " + str(r_rep.statistic))

#%% prediction interval (interruptibility x disturbance)

# Code based on:
# https://github.com/jtleek/replication_paper/blob/gh-pages/code/replication_analysis.Rmd
# Patil, P., Peng, R. D., & Leek, J. T. (2016).
# What Should Researchers Expect When They Replicate Studies?
# A Statistical View of Replicability in Psychological Science.
# Perspectives on Psychological Science, 11(4), 539–544.
# https://doi.org/10.1177/1745691616646366


x = data[' disturbance']
y = data[' interruptibility']
r_rep = pearsonr(x, y)

r_original_l = 0.807 # lab
fish_orig = np.arctanh(r_original_l)
n_orig_l = 72 # lab
n_rep = 82

se_total = np.sqrt(1/(n_orig_l-3) + 1/(n_rep-3))
low = np.tanh(fish_orig - se_total * 1.96)
high = np.tanh(fish_orig + se_total * 1.96)

print("(interruptibility x disturbance):\n")

print("Lab:\n")
print("Original Effect Size (Lab): " + str(r_original_l))
print("PI Low: "+ str(low))
print("PI High: " + str(high))
print("Replicated Effect Size: " + str(r_rep.statistic))

r_original_f = 0.741 # field

fish_orig = np.arctanh(r_original_f)
n_orig_f = 139 # field
n_rep = 82

se_total = np.sqrt(1/(n_orig_f-3) + 1/(n_rep-3))
low = np.tanh(fish_orig - se_total * 1.96)
high = np.tanh(fish_orig + se_total * 1.96)

print("\n")


print("Field:\n")
print("Original Effect Size: (Field)" + str(r_original_f))
print("PI Low: "+ str(low))
print("PI High: " + str(high))
print("Replicated Effect Size: " + str(r_rep.statistic))


