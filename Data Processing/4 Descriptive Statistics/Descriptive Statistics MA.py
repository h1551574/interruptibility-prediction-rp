# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 11:40:41 2023

@author: Florian Poreba
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%

path = "Data/Interruption Data.csv"
data = pd.read_csv(path)


#%%
font = {'family' : 'serif',
        'weight' : 'regular',
        'size'   : 28}

plt.rc('font', **font)

interruptible_data = data['interruptible'].value_counts()

labels_studies = ['Original Paper', 'Reproduction']
labels = ["not interruptible","interruptible"]
size_reproduction = interruptible_data
size_original_paper_lab = [60,12]
size_original_paper_field = [99,40]

fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(10,20))
ax1.pie(size_original_paper_lab, labels=labels, autopct='%1.2f%%',
       colors=['#ff8888', '#6666ff'])
ax1.set_title("Original Paper [Lab]")

ax2.pie(size_original_paper_field, labels=labels, autopct='%1.2f%%',
       colors=['#ff8888', '#6666ff'])
ax2.set_title("Original Paper [Field]")

ax3.pie(size_reproduction, labels=labels, autopct='%1.2f%%',
       colors=['#ff8888', '#6666ff'])
ax3.set_title("Reproduction")


#%%

interruptibility_data = data[' interruptibility'].value_counts().sort_index()
size_original_paper_lab = [7,25,28,7,5]
size_original_paper_field = [19,45,35,25,15]

fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(10,20))
ax1.bar(interruptibility_data.keys(), size_original_paper_lab,
       color=['#ff4444', '#ff6666','#ff9999','#7777ff','#4444ff'])
ax1.set_title("Original Paper [Lab]")


ax2.bar(interruptibility_data.keys(), size_original_paper_field,
       color=['#ff4444', '#ff6666','#ff9999','#7777ff','#4444ff'])
ax2.set_title("Original Paper [Field]")


ax3.bar(interruptibility_data.keys(), interruptibility_data,
       color=['#ff4444', '#ff6666','#ff9999','#7777ff','#4444ff'])
ax3.set_title("Reproduction")


