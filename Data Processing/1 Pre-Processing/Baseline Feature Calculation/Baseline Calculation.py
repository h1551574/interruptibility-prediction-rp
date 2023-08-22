# -*- coding: utf-8 -*-
"""
Created on Sat May 27 00:43:46 2023

@author: alerr
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from scipy import signal
import seaborn as sns
sns.set(font_scale=1.2)
import matplotlib.pyplot as plt
import datetime
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import neurokit2 as nk
import pickle


#%%
baselines = []
hrs = {}
#%%
pIDs = range(1,10)
times = []
for id in pIDs:
        pID= str(id)
        nexus_data_p1_file_path = "Experiment Data/Participant "+pID+"/Baseline/Participant "+pID+" Baseline (Fishtank) Data Nexus-10.csv"
        nexus_data_p1 = pd.read_csv(nexus_data_p1_file_path, skiprows=11)
        times.append(len(nexus_data_p1)/256)
min(times) 
# The shortest baseline sample is 137s long
# To avoid artifacts from starting/stopping the recording 
# I can only really use 100s of Data starting 20s after the recording start
# This sample window will be used for all baseline calculations
# %%  
pID= "1"   
baseline_data = {}

nexus_data_p1_file_path = "Experiment Data/Participant "+pID+"/Baseline/Participant "+pID+" Baseline (Fishtank) Data Nexus-10.csv"

nexus_data_p1 = pd.read_csv(nexus_data_p1_file_path, skiprows=11)

# Drop Last Row because it is empty
nexus_data_p1.drop(nexus_data_p1.tail(1).index,inplace=True)


# Read File to extract start date and time
with open(nexus_data_p1_file_path, 'r') as filedata:
    linesList = filedata.readlines(256)
    nexusStartDateString = linesList[5]
    nexusStartTimeString = linesList[6]
    filedata.close()
    
# Filter and concatenate the datetime string
nexusStartDateString = nexusStartDateString[6:len(nexusStartDateString)-1]
nexusStartTimeString = nexusStartTimeString[6:len(nexusStartTimeString)-1]
nexusStartDateTimeString = nexusStartDateString + " " + nexusStartTimeString

# Format into datetime
format = '%Y-%m-%d %H:%M:%S'
dt = datetime.datetime.strptime(nexusStartDateTimeString, format)
# Convert from Datetime to Millis
millisec = int(dt.timestamp() * 1000)

# Calculate timestamps based on start datetime
timestamps = []
# Second to Millis = 1/1000
# Samples 256hz 1/256
fs_eeg_ecg = 256
for n in range(0,len(nexus_data_p1)):
    timestamps.append(n*(1/256*1000)+millisec)

# Store timestamps in original data set
nexus_data_p1['tsMillis'] = timestamps

# Removing Parts containing Artifiacts from Starting/Stopping the Recording
sf = 256
b_sample_start = sf * 20
b_sample_duration_in_seconds = sf*100

b_sample_end = b_sample_start + b_sample_duration_in_seconds
nexus_data_p1 = nexus_data_p1[b_sample_start:b_sample_end]
#nexus_data_p1.reset_index(drop=True)


nexus_data_p1_32 = nexus_data_p1.iloc[0::8]
fs_temp_eda = 32
# Optional: Select relevant measures
nexus_data_p1_32 = nexus_data_p1_32[['Sensor-E:SC/GSR','Sensor-F:Temp.','tsMillis']]

# Filtering EEG Data
nexus_data_p1_eeg = nexus_data_p1['Sensor-A:EEG']

data = nexus_data_p1_eeg
N = np.floor(len(data)/8)
print("N:")
print(N/256)
print("compare: 4s")

# Optional: plot the signal
data = nexus_data_p1_eeg
sf = 256.
time = np.arange(data.size) / sf
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plt.plot(time, data, lw=1.5, color='k')
plt.xlabel('Time (seconds)')
plt.ylabel('uV')
plt.xlim([time.min(), time.max()])
plt.title('EEG Data Full RAW')
sns.despine()

# Create Filter: EEG (3 - 100 Hz) 5th order butterworth bandpass filter
fs = 256
fpass = [3, 100]
#fpass = [0.1, 100]

wpass = [fpass[0] / (fs / 2),fpass[1] / (fs / 2)]
b,a = signal.butter(5, wpass, "pass")
w, h = signal.freqz(b,a, fs = fs)
# Optional: Plot filter frequency response
# plt.plot(w, abs(h), label="order = %d" % 5)


# Filter the whole EEG Data and store it in the original data set
nexus_data_p1_filtered_step_1 = signal.filtfilt(b, a, nexus_data_p1_eeg)

# 50hz Notch Filter to Remove Mains Hum 
# 2nd Order Butterworth Filter Bandstop:49-51hz
fs = 256
# Filtering 50hz mains hum
fstop = [49, 51]
# Uncomment for 60hz mains hum
# fstop = [59, 61]
wstop = [fstop[0] / (fs / 2),fstop[1] / (fs / 2)]
b,a = signal.butter(2, wstop, "stop")
w, h = signal.freqz(b,a, fs = fs)
# Optional: Plot filter frequency response
# plt.plot(w, abs(h), label="order = %d" % 2)

# Apply the 50hz notch filter to the filtered data set
nexus_data_p1['EEG-Filtered(0.1-100hz)'] = signal.filtfilt(b, a, nexus_data_p1_filtered_step_1)

# Optional: plot the filtered signal
data = nexus_data_p1['EEG-Filtered(0.1-100hz)']
sf = 256.
time = np.arange(data.size) / sf
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plt.plot(time, data, lw=1.5, color='k')
plt.xlabel('Time (seconds)')
plt.ylabel('uV')
plt.xlim([time.min(), time.max()])
plt.title('EEG Data Full P'+pID+' Filtered(0.1-100hz) + 50hz Notch Filter')
sns.despine()

#---------------------------------------------------------------------
# 0.1-3hz Bandpass Filter for Blink Rate Extraction
# Unklar ob das wirklich gut das Augenblinken erkennt
# Die Beschreibung in den Papers ist auch nicht ganz klar wegen der Umsetzung
# Ich glaube jetzt wird die Eye Blink Rate eher unterschätzt? (Avg.: 5.45/m)
nexus_data_p1_eeg = nexus_data_p1['Sensor-A:EEG']

fs = 256
fpass = [0.1, 8]
wpass = [fpass[0] / (fs / 2),fpass[1] / (fs / 2)]
b,a = signal.butter(2, wpass, "pass")
w, h = signal.freqz(b,a, fs = fs)

#Optional: Plot Frequency Response 
# plt.plot(w, abs(h), label="order = %d" % 2)
# plt.xlim([0, 10])

#filter the full raw eeg data
nexus_data_p1['EEG-Filtered(0.1-8hz)'] = signal.filtfilt(b, a, nexus_data_p1_eeg)

# Plot the Filtered Data
data = nexus_data_p1['EEG-Filtered(0.1-8hz)']
sf = 256.
time = np.arange(data.size) / sf
# Plot the signal
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plt.plot(time, data, lw=1.5, color='k')
plt.xlabel('Time (seconds)')
plt.ylabel('uV')
plt.xlim([time.min(), time.max()])
plt.title('EEG Data P'+pID+' Full 0.1-8hz')
sns.despine()

nexus_data_p1_ecg = nexus_data_p1['Sensor-B:ECG']

## ECG (0.1 - 40 Hz) butterworth bandpass filter
# High-Pass to avoid low frequency drift
# Low-Pass to filter out noise incl. 50hz Mains Hum
fs = 256
fpass = [0.1, 40]
wpass = [fpass[0] / (fs / 2),fpass[1] / (fs / 2)]
b,a = signal.butter(5, wpass, "pass")
w, h = signal.freqz(b,a, fs = fs)
plt.plot(w, abs(h), label="order = %d" % 5)


#filter the full raw eeg data
nexus_data_p1['ECG-Filtered(0.1-40hz)'] = signal.filtfilt(b, a, nexus_data_p1_ecg)

# Plot the Filtered Data
data = nexus_data_p1['ECG-Filtered(0.1-40hz)']
sf = 256.
time = np.arange(data.size) / sf
# Plot the signal
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plt.plot(time, data, lw=1.5, color='k')
plt.xlabel('Time (seconds)')
plt.ylabel('uV')
# Optionally inspect smaller subset
# plt.xlim([sample_start_in_seconds, sample_end_in_seconds])
plt.xlim([time.min(), time.max()])
plt.title('ECG Data P1 Full 0.1-40hz')
sns.despine()

# Remove Duplicate Rows for lower Sampling Rate Data
nexus_data_p1_32 = nexus_data_p1.iloc[0::8]
fs_temp_eda = 32
# Optional: Select relevant measures
nexus_data_p1_32 = nexus_data_p1_32[['Sensor-E:SC/GSR','Sensor-F:Temp.','tsMillis']]


# FILTERING EEG DATA
nexus_data_p1_eeg = nexus_data_p1['Sensor-A:EEG']

# Optional: plot the signal
data = nexus_data_p1_eeg
sf = 256.
time = np.arange(data.size) / sf
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plt.plot(time, data, lw=1.5, color='k')
plt.xlabel('Time (seconds)')
plt.ylabel('uV')
plt.xlim([time.min(), time.max()])
plt.title('EEG Data Full RAW')
sns.despine()

# Create Filter: EEG (0.1 - 100 Hz) 5th order butterworth bandpass filter
fs = 256
fpass = [3, 100]
#fpass = [0.1, 100]
wpass = [fpass[0] / (fs / 2),fpass[1] / (fs / 2)]
b,a = signal.butter(5, wpass, "pass")
w, h = signal.freqz(b,a, fs = fs)
# Optional: Plot filter frequency response
# plt.plot(w, abs(h), label="order = %d" % 5)


# Filter the whole EEG Data and store it in the original data set
nexus_data_p1_filtered_step_1 = signal.filtfilt(b, a, nexus_data_p1_eeg)

# 50hz Notch Filter to Remove Mains Hum 
# 2nd Order Butterworth Filter Bandstop:49-51hz
fs = 256
# Filtering 50hz mains hum
fstop = [49, 51]
# Uncomment for 60hz mains hum
# fstop = [59, 61]
wstop = [fstop[0] / (fs / 2),fstop[1] / (fs / 2)]
b,a = signal.butter(2, wstop, "stop")
w, h = signal.freqz(b,a, fs = fs)
# Optional: Plot filter frequency response
# plt.plot(w, abs(h), label="order = %d" % 2)

# Apply the 50hz notch filter to the filtered data set
nexus_data_p1['EEG-Filtered(0.1-100hz)'] = signal.filtfilt(b, a, nexus_data_p1_filtered_step_1)

# Optional: plot the filtered signal
data = nexus_data_p1['EEG-Filtered(0.1-100hz)']
sf = 256.
time = np.arange(data.size) / sf
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plt.plot(time, data, lw=1.5, color='k')
plt.xlabel('Time (seconds)')
plt.ylabel('uV')
plt.xlim([time.min(), time.max()])
plt.title('EEG Data Full P1 Filtered(0.1-100hz) + 50hz Notch Filter')
sns.despine()


#EEG FILTERING FOR EYE BLINK---------------------------------------------------------------------
# 0.1-3hz Bandpass Filter for Blink Rate Extraction
# Unklar ob das wirklich gut das Augenblinken erkennt
# Die Beschreibung in den Papers ist auch nicht ganz klar wegen der Umsetzung
# Ich glaube jetzt wird die Eye Blink Rate eher unterschätzt? (Avg.: 5.45/m)
nexus_data_p1_eeg = nexus_data_p1['Sensor-A:EEG']

fs = 256
fpass = [0.1, 8]
wpass = [fpass[0] / (fs / 2),fpass[1] / (fs / 2)]
b,a = signal.butter(2, wpass, "pass")
w, h = signal.freqz(b,a, fs = fs)

#Optional: Plot Frequency Response 
# plt.plot(w, abs(h), label="order = %d" % 2)
# plt.xlim([0, 10])

#filter the full raw eeg data
nexus_data_p1['EEG-Filtered(0.1-8hz)'] = signal.filtfilt(b, a, nexus_data_p1_eeg)

# Plot the Filtered Data
data = nexus_data_p1['EEG-Filtered(0.1-8hz)']
sf = 256.
time = np.arange(data.size) / sf
# Plot the signal
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plt.plot(time, data, lw=1.5, color='k')
plt.xlabel('Time (seconds)')
plt.ylabel('uV')
plt.xlim([time.min(), time.max()])
plt.title('EEG Data P1 Full 0.1-8hz')
sns.despine()


nexus_data_p1_ecg = nexus_data_p1['Sensor-B:ECG']

## ECG (0.1 - 40 Hz) butterworth bandpass filter
# High-Pass to avoid low frequency drift
# Low-Pass to filter out noise incl. 50hz Mains Hum
fs = 256
fpass = [0.1, 40]
wpass = [fpass[0] / (fs / 2),fpass[1] / (fs / 2)]
b,a = signal.butter(5, wpass, "pass")
w, h = signal.freqz(b,a, fs = fs)
plt.plot(w, abs(h), label="order = %d" % 5)


#filter the full raw eeg data
nexus_data_p1['ECG-Filtered(0.1-40hz)'] = signal.filtfilt(b, a, nexus_data_p1_ecg)

# Plot the Filtered Data
#data = nexus_data_p1['ECG-Filtered(0.1-40hz)']
data = nexus_data_p1_ecg
sf = 256.
time = np.arange(data.size) / sf
# Plot the signal
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plt.plot(time, data, lw=1.5, color='k')
plt.xlabel('Time (seconds)')
plt.ylabel('uV')
# Optionally inspect smaller subset
# plt.xlim([sample_start_in_seconds, sample_end_in_seconds])
plt.xlim([time.min(), time.max()])
plt.title('ECG Data P1 Full 0.1-40hz')
sns.despine()




# Method from: https://raphaelvallat.com/bandpower.html
def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.
    
    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.
        
    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band
    
    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf
        
    # Compute the modified periodogram (Welch) (ADAPTED FOR THE REPLICATION)
    # Emulating Matlab pwelch() default behavior of splitting into 8 segments
    # See: https://dsp.stackexchange.com/questions/84193/why-does-python-welch-give-me-a-different-answer-from-matlabs-pwelch
    N = np.floor(len(data)/8)
    # Setting detrend=False
    # Explanation https://github.com/scipy/scipy/issues/8045#issuecomment-337319294
    freqs, psd = welch(data, sf, nperseg=N,detrend=False)
    #freqs, psd = welch(data, sf, nperseg=nperseg,detrend=False)
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]
    
    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    
    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)
    
    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

# data = data[time_window]

bandpower_data = []
data = nexus_data_p1['EEG-Filtered(0.1-100hz)']

win_sec = 4
bandpowers = {}
# Delta
bandpowers["delta"] = bandpower(data,fs,[0.5,4],win_sec,relative=False)
# Theta
bandpowers["theta"] = bandpower(data,fs,[4,8],win_sec,relative=False)
# Alpha
bandpowers["alpha"] = bandpower(data,fs,[8,12],win_sec,relative=False)
# Beta
bandpowers["beta"] = bandpower(data,fs,[12,30],win_sec,relative=False)
# Gamma
bandpowers["gamma"] = bandpower(data,fs,[30,100],win_sec,relative=False)
 # Calculate fractions of all combinations of frequency bands
bands = list(bandpowers.keys())
for band in bands:
    for other_band in bands:
        if band!=other_band:
            bandpowers[band+"/"+other_band] = bandpowers[band]/bandpowers[other_band]
 # Calculate θ/(α + β) and β/(α + θ)
bandpowers["theta/(alpha+beta)"] = bandpowers["theta"] / (bandpowers["alpha"]+bandpowers["beta"])
bandpowers["beta/(alpha+theta)"] = bandpowers["beta"] / (bandpowers["alpha"]+bandpowers["theta"])

#bandpower_data.append(bandpowers)
#for bandname, bandpower in bandpowers.items():
#     measurement_string = bandname+"_tw_"+time_window
  #   tw_subsample_measurement_data[measurement_string] = bandpower

# Add Bandpower data to interruption data set
#bandpower_data_df = pd.DataFrame.from_dict(bandpower_data)
# print(df)

baseline_data = bandpowers
#pd.concat([interruption_data_tw[time_window].reset_index(drop=True),bandpower_data_df.reset_index(drop=True)], axis=1)
# print(interruption_data_tw[time_window])




data = nexus_data_p1['EEG-Filtered(0.1-8hz)']
# PEAK FINDING Algorithm (tweaking parameters (e.g. prominence, height) unknown)
peaks = np.array(signal.find_peaks(data,prominence=100,height=50)[0])
# print(peaks)
num_of_blinks = len(peaks)

#        # Optional: Plot the Peaks
#        sf = 256.
#        time = np.arange(data.size) / sf
#        # Plot the signal
#        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
#        plt.plot(time, data, lw=1.5, color='k')
#        plt.axhline(y=0, color='r', linestyle='-')
#        plt.xlabel('Time (seconds)')
#        plt.ylabel('uV')
#        #plt.xlim([time.min(), time.max()])
#        plt.xlim([time.min(), time.max()])
#        plt.plot(peaks/fs, np.array(data)[peaks], "xr")
#        plt.legend(['peaks'])
#        plt.title('EEG Data P1 Full 2-3hz')
#        sns.despine()
baseline_data['num_of_blinks'] = num_of_blinks



ecg_signal = nexus_data_p1['Sensor-B:ECG']
#peaks = signal.find_peaks(ecg_signal,prominence=1000)[0]
ecg_signals, ecg_info = nk.ecg_process(ecg_signal, sampling_rate=256)
#print(ecg_info)
peaks = ecg_info['ECG_R_Peaks']
peak_onsets = np.where(ecg_signals['ECG_R_Onsets'] == 1)[0]

rri = np.diff(peaks) / sf * 1000
clean_ecg_signal = ecg_signals['ECG_Clean']
peak_vals = clean_ecg_signal.iloc[peaks].reset_index(drop=True)
onset_vals = clean_ecg_signal.iloc[peak_onsets].reset_index(drop=True)
amplitudes = np.subtract(peak_vals,onset_vals)

mean_hr_in_bpm = 60000/(np.mean(rri))
       # print("TW: " + time_window)
        #print(mean_hr_in_bpm)
        #print(rri)

max_peak_amplitude = max(peak_vals)
mean_peak_aplitude = np.mean(peak_vals)
sum_peak_amplitude = sum(peak_vals)
num_of_peaks = len(peaks)

hr_variance_in_bpm = 60000/(np.var(rri,ddof=1))
sdnn = np.std(rri,ddof=1)

diff_rri = abs(np.diff(rri))

nn50 = np.sum(np.abs(diff_rri) > 50)
nn20 = np.sum(np.abs(diff_rri) > 20)
pnn50 = nn50 / (len(diff_rri) + 1) * 100
pnn20 = nn20 / (len(diff_rri) + 1) * 100


baseline_data['mean_hr_in_bpm'] = mean_hr_in_bpm
baseline_data['hr_variance_in_bpm'] = hr_variance_in_bpm
baseline_data['max_peak_amplitude'] = max_peak_amplitude
baseline_data['num_of_peaks'] = num_of_peaks
baseline_data['sum_peak_amplitude'] = sum_peak_amplitude
baseline_data['sdnn'] = sdnn
baseline_data['pnn20'] = pnn20
baseline_data['pnn50'] = pnn50


  #DEBUG------------------------------------------------
data = ecg_signals['ECG_Clean']
peaks = ecg_info['ECG_R_Peaks']

peak_onsets = np.where(ecg_signals['ECG_R_Onsets'] == 1)[0]
sf = 256.
time = np.arange(data.size) / sf
# Plot the signal
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plt.plot(time, data, lw=1.5, color='k')
plt.xlabel('Time (seconds)')
plt.ylabel('uV')
# Optionally inspect smaller subset
#plt.xlim([sample_start_in_seconds, sample_end_in_seconds])
plt.plot(peaks/sf, data[peaks], "xr")
plt.plot(peak_onsets/sf, data[peak_onsets], "xg")

#plt.vlines(x = peaks/sf, ymin = data[peak_onsets], ymax = data[peaks],
#            colors = 'purple',
#           label = 'peak-peak_onset ~ r peak amplitude')

plt.xlim([time.min(), time.max()/10])
plt.title('ECG Data Clean')
sns.despine()

# TEMP DATA-----------------
f=32
# NOTE: Using the 32hz Data Set for SCR
temp_signal = nexus_data_p1_32['Sensor-F:Temp.']
        
mean_temp = np.mean(temp_signal)
max_temp = np.max(temp_signal)


#print(time_window)
baseline_data['mean_temp'] = mean_temp
baseline_data['max_temp'] = max_temp


# Save Baseline Data as Pickle Dump
file_name = "baseline_p" + pID
with open(file_name+'.pkl', 'wb') as fp:
    pickle.dump(baseline_data, fp)

baselines.append(baseline_data)

#DEBUG
print(b_sample_duration_in_seconds/sf)

print(bandpowers)


#%%

df = pd.DataFrame(baselines)
print(df)

df.to_csv("baselines.csv")

#%% Skip EDA
# EDA DATA ______________________________
#import warnings
sf=32

eda_signal = nexus_data_p1['Sensor-E:SC/GSR']

eda_signal = eda_signal.reset_index(drop=True)

# Smoothing Step from Original Study
#        with warnings.catch_warnings():
#            warnings.simplefilter("ignore")
#            eda_signal = SimpleExpSmoothing(eda_signal).fit(
#                smoothing_level=0.15).fittedvalues

signals, info = nk.eda_process(eda_signal, sampling_rate=32)

# Optional: Plot Processed EDA Signal
# This produced errors in my testing
# ValueError: Length of values (2) does not match length of index (3)
#        try:
#            eda_plot = nk.eda_plot(signals)
#            eda_plot.set_size_inches(18.5, 10.5)
#            eda_plot
#        except ValueError:
#            print("EDA Plot Value Error")

peaks_per_time_unit = len(info['SCR_Peaks'])
mean_peak_amplitudes = np.mean(info['SCR_Amplitude'])
sum_peak_amplitudes = np.sum(info['SCR_Amplitude'])
mean_scl = np.mean(signals['EDA_Tonic'])
auc = np.sum(signals['EDA_Phasic'])*sf
    


#print(time_window)
baseline_data['scr_peaks_per_time_unit'] = peaks_per_time_unit
baseline_data['mean_scr_peak_amplitudes'] = mean_peak_amplitudes
baseline_data['sum_scr_peak_amplitudes'] = sum_peak_amplitudes
baseline_data['mean_scl'] = mean_scl
baseline_data['scr_auc'] = auc



#%%
file_name = "baseline_p" + pID
with open(file_name+'.pkl', 'rb') as fp:
    baseline_data2 = pickle.load(fp)


