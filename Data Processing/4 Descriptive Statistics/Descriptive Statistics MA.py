# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 11:40:41 2023

@author: Florian Poreba
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.lines as mlines


plt.style.use('default')
font = {'family' : 'serif',
        'weight' : 'regular',
        'size'   : 35}

plt.rc('font', **font)

#%%

path = "Data/Interruption Data (ANONYMIZED)/Aggregated Interruption Data.csv"
data = pd.read_csv(path)


#%%
interruptible_data = data['interruptible'].value_counts()

labels_studies = ['Original Paper', 'Reproduction']
labels = ["interruptible","not interruptible"]
size_reproduction = interruptible_data
size_original_paper_lab = [60,12]
size_original_paper_field = [99,40]

fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(30,20))
ax1.pie(size_original_paper_lab, autopct='%1.2f%%',
       colors=['#ff8888', '#6666ff'])
ax1.set_title("Original Paper [Lab]")

ax2.pie(size_original_paper_field, autopct='%1.2f%%',
       colors=['#ff8888', '#6666ff'])
ax2.set_title("Original Paper [Field]")


ax3.pie(size_reproduction, autopct='%1.2f%%',
       colors=['#ff8888', '#6666ff'])
ax3.set_title("Reproduction")


fig.tight_layout()
fig.legend(labels,loc="lower right")
plt.subplots_adjust(bottom=-0.5)

#%%
interruptibility_data = data[' interruptibility'].value_counts().sort_index()
size_original_paper_lab = [7,25,28,7,5]
size_original_paper_field = [19,45,35,25,15]
labels = ["1","2","3","4","5"]
x = [1,2,3,4,5]

fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(30,10),layout='constrained')
ax1.bar(interruptibility_data.keys(), size_original_paper_lab,
       color=['#ff4444', '#ff6666','#ff9999','#7777ff','#4444ff'])
ax1.set_title("Original Paper [Lab]")
ax1.grid(False)
start, end = ax1.get_xlim()
ax1.xaxis.set_ticks(np.arange(1, 6, 1))

ax2.bar(interruptibility_data.keys(), size_original_paper_field,
       color=['#ff4444', '#ff6666','#ff9999','#7777ff','#4444ff'])
ax2.set_title("Original Paper [Field]")
ax2.grid(False)
start, end = ax2.get_xlim()
ax2.xaxis.set_ticks(np.arange(1, 6, 1))


ax3.bar(interruptibility_data.keys(), interruptibility_data,
       color=['#ff4444', '#ff6666','#ff9999','#7777ff','#4444ff'])
ax3.set_title("Reproduction")
ax3.grid(False)
start, end = ax3.get_xlim()
ax3.xaxis.set_ticks(np.arange(1, 6, 1))


one = mlines.Line2D([], [], color='red', lw=8, label='1 = Very Interruptible')
five = mlines.Line2D([], [], color='blue', lw=8, label='5 = Very Uninterruptible')

#fig.legend(['1 = Very Interruptible',"5 = Very Uninterruptible"],loc="outside upper center", 
#           title='"How do you rate your interruptibility at the time of the notification?"',
#)
fig.legend(handles=[one,five],
           loc="outside upper center",
           title='"How do you rate your interruptibility at the time of the notification?"',
)

plt.show()


#%%
data['interruption_lag_in_seconds'] = data[' interruption_lag'] / 1000

fig, ax = plt.subplots(figsize=(15,10))
sns.histplot(x="interruption_lag_in_seconds",data=data,ax=ax)
plt.xlabel("Interruption Lag in Seconds")


plt.savefig("interruption_lag_distribution_hist_MA.pdf")
