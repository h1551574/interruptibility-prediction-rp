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

path = "Normalized Data/10/with_pID/normalized_interruption_data_ALL_tw_10s_pID_3hz.csv"
data = pd.read_csv(path)

#%%
print(len(data.columns))

#%%
interruptible_data = data['interruptible'].value_counts()

labels_studies = ['Original Paper', 'Reproduction']
labels = ["interruptible","not interruptible"]
size_reproduction = interruptible_data
size_original_paper_lab = [60,12]
size_original_paper_field = [99,40]

fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5))
ax1.pie(size_original_paper_lab, labels=labels, autopct='%1.2f%%',
       colors=['#ff6666', '#7777ff'])
ax1.set_title("Original Paper [Lab]")

ax2.pie(size_original_paper_field, labels=labels, autopct='%1.2f%%',
       colors=['#ff6666', '#7777ff'])
ax2.set_title("Original Paper [Field]")

ax3.pie(size_reproduction, labels=labels, autopct='%1.2f%%',
       colors=['#ff6666', '#7777ff'])
ax3.set_title("Reproduction [Lab]")

plt.savefig("two_state_piechart_MA.pdf")



#%%

interruptibility_data = data[' interruptibility'].value_counts().sort_index()
size_original_paper_lab = [7,25,28,7,5]
size_original_paper_field = [19,45,35,25,15]

fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5))
ax1.bar(interruptibility_data.keys(), size_original_paper_lab,
       color=['#ff4444', '#ff6666','#ff9999','#7777ff','#4444ff'])
ax1.set_title("Original Paper [Lab]")


ax2.bar(interruptibility_data.keys(), size_original_paper_field,
       color=['#ff4444', '#ff6666','#ff9999','#7777ff','#4444ff'])
ax2.set_title("Original Paper [Field]")


ax3.bar(interruptibility_data.keys(), interruptibility_data,
       color=['#ff4444', '#ff6666','#ff9999','#7777ff','#4444ff'])
ax3.set_title("Reproduction [Lab]")

plt.savefig("five_state_piechart_MA.pdf")




#%%
grouped_data = data.groupby('pID')[' interruptibility'].value_counts()

fig, axs = plt.subplots(1,10,figsize=(15,10), sharex=True, sharey=True)
for ax, p in zip(axs, grouped_data):
    #ax.bar(interruptibility_data.keys(), interruptibility_data,
     #      color=['#ff4444', '#ff6666','#ff9999','#7777ff','#4444ff'])
    #ax.set_title("Reproduction [Lab]")
    ax.bar(p[' interruptibility'], p['count'],
           color=['#ff4444', '#ff6666','#ff9999','#7777ff','#4444ff'])
    ax.set_title(p['pID'])


#%%

#data['rel_ts'] = data['tsStart_millis']

start_ts = data.groupby('pID')['tsStart_millis'].min()
rel_data = data.join(start_ts, on='pID', rsuffix='_min')

rel_data['rel_ts'] = (rel_data['tsStart_millis'] - rel_data['tsStart_millis_min'])/(1000*60)

rel_data.set_index('rel_ts', inplace=True)
rel_data.groupby('pID')[' interruptibility'].plot(subplots=False)



#%%

data[' interruptibility'].plot.scatter(by=data['pID'],sharex=True,sharey=True)

#%%
df = rel_data
df["c_pID"] = df["pID"].astype("category")

fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(x='rel_ts', y=' interruptibility', hue='c_pID', data=df,legend=False) 
plt.show()


fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,6))
ax1 = sns.histplot(x='rel_ts', y='interruptible', data=df,legend=True) 
#sns.scatterplot(x='rel_ts', y='interruptible', hue='c_pID', data=df,legend=False) 
plt.show()

#%%
data['interruption_lag_in_seconds'] = data[' interruption_lag'] / 1000

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,6))
ax1 = sns.histplot(ax = ax1, x="interruption_lag_in_seconds",data=data)
ax2 = sns.histplot(ax = ax2, x="interruption_lag_in_seconds",data=data)
plt.savefig("interruption_lag_distribution_MA.pdf")



#%%
data['interruption_lag_in_seconds'] = data[' interruption_lag'] / 1000

fig, ax = plt.subplots(figsize=(8,6))
sns.histplot(x="interruption_lag_in_seconds",data=data)
plt.savefig("interruption_lag_distribution_hist_MA.pdf")

#%%
data['interruption_lag_in_seconds'] = data[' interruption_lag'] / 1000

fig, ax = plt.subplots(figsize=(8,6))
sns.boxplot(x="interruption_lag_in_seconds",data=data)
plt.savefig("interruption_lag_distribution_box_MA.pdf")


#%%

data['interruption_lag_in_seconds'].median()
data['interruption_lag_in_seconds'].max()


#%%

fig, ax = plt.subplots(figsize=(8,6))
sns.boxplot(y="interruption_lag_in_seconds",x=" interruptibility",data=data)

#%%

fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(y="interruption_lag_in_seconds",x=" interruptibility",data=data)

x = data['interruption_lag_in_seconds']
y = data[' interruptibility']

pearsonr(x, y)
spearmanr(x, y)

#%%

sns.stripplot(y="interruption_lag_in_seconds",x=" interruptibility",data=data)

#%%

sns.stripplot(y=" interruptibility",x=" disturbance",data=data)


x = data['interruption_lag_in_seconds']
y = data[' interruptibility']

pearsonr(x, y)
#spearmanr(x, y)

#%%
x = data[' disturbance']
y = data[' interruptibility']
corr = pearsonr(x, y)

print(corr)

(
    so.Plot(data,x,y)
    .add(so.Dots(), so.Jitter(x=0.3,y=0.3))
)

#%%
x = data[' mental_workload']
y = data[' interruptibility']
pID = data['pID']
corr = pearsonr(x, y)

print(corr)

(
    so.Plot(data,x,y)
    .add(so.Dots(), so.Jitter(x=0.3,y=0.3),color="pID")
)



#%%

interruptibility_data = data[' interruptibility'].value_counts().sort_index()
mental_workload_data = data[' mental_workload'].value_counts().sort_index()

size_original_paper_lab = [7,25,28,7,5]
size_original_paper_field = [19,45,35,25,15]

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
ax1.bar(interruptibility_data.keys(), interruptibility_data)
ax1.set_title("Replication [Interruptibility]")


ax2.bar(mental_workload_data.keys(), mental_workload_data)
ax2.set_title("Replication [Mental Workload]")



#plt.savefig("five_state_piechart_MA.pdf")


#%%

sns.set(rc={'figure.figsize':(8.0,8.0)})

ax = sns.histplot(binwidth=0.9, x=" interruptibility", hue=" mental_workload", data=data, stat="count", multiple="stack",discrete=True)
sns.despine(ax=ax)
plt.show()


#%%

sns.set(rc={'figure.figsize':(8.0,8.0)})

hist = sns.histplot(binwidth=0.9, x=" mental_workload", hue=" interruptibility", data=data, stat="count", multiple="stack",discrete=True)
sns.despine(ax=ax)
plt.show()

#add overall title to replot
hist.fig.suptitle('Overall Title')

#%%
grouped = data.groupby(['pID'])


for g in grouped:
    p_data = g[1]
    p_reset_data = p_data.reset_index(drop=True)
    title = "pID: " + str(p_reset_data["pID"][0])
    sns.set(rc={'figure.figsize':(8.0,8.0)})
    ax = sns.histplot(binwidth=0.9, x=" interruptibility", hue=" mental_workload", data=p_reset_data, stat="count", multiple="stack",discrete=True).set(title=title)
    plt.show()
    
#%%
for p_data in grouped:
    sns.set(rc={'figure.figsize':(8.0,8.0)})
    ax = sns.histplot(binwidth=0.9, x=" mental_workload", hue=" interruptibility", data=p_data, stat="count", multiple="stack",discrete=True).set(title=str())


#%% 

no_outlier_data = data[data.pID != 4]
no_outlier_data = no_outlier_data[no_outlier_data.pID != 6]
x = no_outlier_data[' mental_workload']
y = no_outlier_data[' interruptibility']
r_rep = pearsonr(x, y)
r_rep


#%%

ils = data['interruption_lag_in_seconds']
mean = np.mean(ils)
std = np.std(ils,ddof=1)

sns.boxplot(ils)

print(mean)
print(std)


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

#r_original = 0.815 # lab
r_original = 0.702 # field

fish_orig = np.arctanh(r_original)
#n_orig = 72 # lab
n_orig = 139 # field
n_rep = 82

se_total = np.sqrt(1/(n_orig-3) + 1/(n_rep-3))
low = np.tanh(fish_orig - se_total * 1.96)
high = np.tanh(fish_orig + se_total * 1.96)

print("Original Effect Size: " + str(r_original))
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

#r_original = 0.807 # lab
r_original = 0.741 # field

fish_orig = np.arctanh(r_original)
#n_orig = 72 # lab
n_orig = 139 # field
n_rep = 82

se_total = np.sqrt(1/(n_orig-3) + 1/(n_rep-3))
low = np.tanh(fish_orig - se_total * 1.96)
high = np.tanh(fish_orig + se_total * 1.96)

print("Original Effect Size: " + str(r_original))
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


x = data[' interruption_lag']
y = data[' interruptibility']
r_rep = pearsonr(x, y)

#r_original = 0.382 # lab
r_original = 0.282 # field

fish_orig = np.arctanh(r_original)
#n_orig = 72 # lab
n_orig = 139 # field
n_rep = 82

se_total = np.sqrt(1/(n_orig-3) + 1/(n_rep-3))
low = np.tanh(fish_orig - se_total * 1.96)
high = np.tanh(fish_orig + se_total * 1.96)

print("Original Effect Size: " + str(r_original))
print("PI Low: "+ str(low))
print("PI High: " + str(high))
print("Replicated Effect Size: " + str(r_rep.statistic))



#%%

interruption_lag_no_outliers = data.loc[data[' interruption_lag'] < (60*1000)]
ils = data['interruption_lag_in_seconds']
ils_n_o = interruption_lag_no_outliers['interruption_lag_in_seconds']


#sns.boxplot(ils)
sns.boxplot(ils_n_o)

mean = np.mean(ils)
std = np.std(ils,ddof=1)

print(mean)
print(std)

corrected_mean = np.mean(ils_n_o)
corrected_std = np.std(ils_n_o,ddof=1)

print(corrected_mean)
print(corrected_std)

print("Number of outliers: " + str(len(ils)-len(ils_n_o)))



#%%

p3_data = data.loc[data['pID']==3].reset_index()

first_interruption = p3_data['tsStart'][0]
first_interruption_ts = datetime.fromisoformat(first_interruption).timestamp()


def relativeTimeStamp(dt):
    ts = datetime.fromisoformat(dt).timestamp()
    rel_ts = ((ts - first_interruption_ts)/(60.0))
    return rel_ts
    
    
#date_format = '%Y-%m-%d %I:%M %p'
p3_data['time_in_minutes'] = list(map(relativeTimeStamp, p3_data['tsStart']))
#p3_data['rel_tsStart'] = map(relativeTimeStamp, p3_data['tsStart'])


#first_interruption_ts = np.min(datetime.fromisoformat(p3_data['tsStart']))

#p3_data['rel_tsStart'] = datetime.fromisoformat(p3_data['tsStart']) - first_interruption_ts

p = sns.scatterplot(p3_data,y=" interruptibility",x="time_in_minutes")


#%%

x_label = 'beta'
y_label = 'theta/beta'


x = data[x_label]
y = data[y_label]
r_rep = pearsonr(x, y)
r_rep

print(r_rep)
sns.scatterplot(p3_data,y=y_label,x=x_label)



