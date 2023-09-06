# -*- coding: utf-8 -*-
"""
@author: Florian Poreba
"""

# -*- coding: utf-8 -*-
"""
Baseline Feature Calculation
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
from pathlib import Path

#%%
pIDs = range(1,11)
times = []
for id in pIDs:
        pID= str(id)
        nexus_data_file_path = "replication-package/Data"+\
            "/SENSITIVE - Biometric Data"+ \
            "/Participant " + pID + \
            "/Baseline/Participant "+pID+" Baseline (Fishtank) Data Nexus-10.csv"

        nexus_data = pd.read_csv(nexus_data_file_path, skiprows=11)
        times.append(len(nexus_data)/256)
min(times) 
# The shortest baseline sample is 137s long
# To avoid artifacts from starting/stopping the recording 
# I can only really use 100s of Data starting 20s after the recording start
# This sample window will be used for all baseline calculations
# %%      
    
def import_nexus_data(filepath):
    
    nexus_data = pd.read_csv(nexus_data_file_path, skiprows=11)

    # Drop Last Row because it is empty
    nexus_data.drop(nexus_data.tail(1).index,inplace=True)


    # Read File to extract start date and time
    with open(nexus_data_file_path, 'r') as filedata:
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
    for n in range(0,len(nexus_data)):
        timestamps.append(n*(1/256*1000)+millisec)

    # Store timestamps in original data set
    nexus_data['tsMillis'] = timestamps
    
    output_df = pd.DataFrame()
    
    # Removing Parts containing Artifiacts from Starting/Stopping the Recording
    sf = 256
    b_sample_start = sf * 20
    b_sample_duration_in_seconds = sf*100

    b_sample_end = b_sample_start + b_sample_duration_in_seconds
    nexus_data = nexus_data[b_sample_start:b_sample_end]
    #nexus_data.reset_index(drop=True)
    return nexus_data
    
    
def get_eeg_from_nexus_data(nexus_data):    
    return nexus_data['Sensor-A:EEG']
    
def get_ecg_from_nexus_data(nexus_data):    
    return nexus_data['Sensor-B:ECG/EKG']

def get_eda_from_nexus_data(nexus_data):
    # Lower Sample Rate: Remove Repeated Data Points 
    nexus_data_32 = nexus_data.iloc[0::8]      
    return nexus_data_32['Sensor-E:SC/GSR']

def get_temperature_from_nexus_data(nexus_data):
    # Lower Sample Rate: Remove Repeated Data Points 
    nexus_data_32 = nexus_data.iloc[0::8]    
    return nexus_data_32['Sensor-F:Temp.']

#%%

DATA_SF = 256.
EEG_SF  = 256.
ECG_SF  = 256.
EDA_SF  = 32.
TEMP_SF = 32.

baselines = []
pIDs = range(1,11)

for id in pIDs:
    pID= str(id)  
    baseline_data = {}
    
    nexus_data_file_path = "replication-package/Data"+\
        "/SENSITIVE - Biometric Data"+ \
        "/Participant " + pID + \
        "/Baseline/Participant "+pID+" Baseline (Fishtank) Data Nexus-10.csv"
    
    sensor_data = import_nexus_data(nexus_data_file_path)
    eeg_data = get_eeg_from_nexus_data(sensor_data)
    ecg_data = get_ecg_from_nexus_data(sensor_data)
    eda_data = get_eda_from_nexus_data(sensor_data)
    temperature_data = get_temperature_from_nexus_data(sensor_data)

    #---------------------------------------------------------------------
    # Filter EEG Data for Brain-Related Feature Calculation
    # Create Filter: EEG (3 - 100 Hz) 5th order butterworth bandpass filter
    fpass = [3, 100]
    
    wpass = [fpass[0] / (EEG_SF / 2),fpass[1] / (EEG_SF / 2)]
    b,a = signal.butter(5, wpass, "pass")
    w, h = signal.freqz(b,a, fs = EEG_SF)
    
    filtered_eeg_data = signal.filtfilt(b, a, eeg_data)
    
    # 50hz Notch Filter to Remove Mains Hum 
    # 2nd Order Butterworth Filter Bandstop:49-51hz
    # Filtering 50hz mains hum
    fstop = [49, 51]
    # Uncomment for 60hz mains hum
    # fstop = [59, 61]
    wstop = [fstop[0] / (EEG_SF / 2),fstop[1] / (EEG_SF / 2)]
    b,a = signal.butter(2, wstop, "stop")
    w, h = signal.freqz(b,a, fs = EEG_SF)
    
    
    # Apply the 50hz notch filter to the filtered data set
    filtered_eeg_data = signal.filtfilt(b, a, filtered_eeg_data)
    
    #---------------------------------------------------------------------
    # Filter Data for Eye Blink Extraction
    fpass = [0.1, 8]
    wpass = [fpass[0] / (EEG_SF / 2),fpass[1] / (EEG_SF / 2)]
    b,a = signal.butter(2, wpass, "pass")
    w, h = signal.freqz(b,a, fs = EEG_SF)
    
    #filter the full raw eeg data
    filtered_eye_blink_eeg_data = signal.filtfilt(b, a, eeg_data)
    
    
    #---------------------------------------------------------------------
    # Calculate EEG Features
    
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
    win_sec = 4
    bandpowers = {}
    # Delta
    bandpowers["delta"] = bandpower(eeg_data,EEG_SF,[0.5,4],win_sec,relative=False)
    # Theta
    bandpowers["theta"] = bandpower(eeg_data,EEG_SF,[4,8],win_sec,relative=False)
    # Alpha
    bandpowers["alpha"] = bandpower(eeg_data,EEG_SF,[8,12],win_sec,relative=False)
    # Beta
    bandpowers["beta"] = bandpower(eeg_data,EEG_SF,[12,30],win_sec,relative=False)
    # Gamma
    bandpowers["gamma"] = bandpower(eeg_data,EEG_SF,[30,100],win_sec,relative=False)
     # Calculate fractions of all combinations of frequency bands
    bands = list(bandpowers.keys())
    for band in bands:
        for other_band in bands:
            if band!=other_band:
                bandpowers[band+"/"+other_band] = bandpowers[band]/bandpowers[other_band]
     # Calculate θ/(α + β) and β/(α + θ)
    bandpowers["theta/(alpha+beta)"] = bandpowers["theta"] / (bandpowers["alpha"]+bandpowers["beta"])
    bandpowers["beta/(alpha+theta)"] = bandpowers["beta"] / (bandpowers["alpha"]+bandpowers["theta"])
    
    baseline_data = bandpowers
    
    
    
    #---------------------------------------------------------------------
    # Calculate Eye Blink (EEG) Features
    data = filtered_eye_blink_eeg_data
    # PEAK FINDING Algorithm (tweaking parameters (e.g. prominence, height) unknown)
    peaks = np.array(signal.find_peaks(data,prominence=100,height=50)[0])
    # print(peaks)
    num_of_blinks = len(peaks)
    baseline_data['num_of_blinks'] = num_of_blinks
    
    
    
    #---------------------------------------------------------------------
    # Calculate Heart-Related (ECG) Features
    ecg_signals, ecg_info = nk.ecg_process(ecg_data, sampling_rate=256)
    peaks = ecg_info['ECG_R_Peaks']
    peak_onsets = np.where(ecg_signals['ECG_R_Onsets'] == 1)[0]
    
    rri = np.diff(peaks) / ECG_SF * 1000
    clean_ecg_signal = ecg_signals['ECG_Clean']
    peak_vals = clean_ecg_signal.iloc[peaks].reset_index(drop=True)
    onset_vals = clean_ecg_signal.iloc[peak_onsets].reset_index(drop=True)
    amplitudes = np.subtract(peak_vals,onset_vals)
    
    mean_hr_in_bpm = 60000/(np.mean(rri))
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
    
    #---------------------------------------------------------------------
    # Calculate Temperature Features
    mean_temp = np.mean(temperature_data)
    max_temp = np.max(temperature_data)
    
    
    #print(time_window)
    baseline_data['mean_temp'] = mean_temp
    baseline_data['max_temp'] = max_temp
    
    dir_name = "replication-package/Data/SENSITIVE - Baselines/"
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    # Save Baseline Data as Pickle Dump
    file_name = "replication-package/Data/SENSITIVE - Baselines/baseline_p" + pID
    with open(file_name+'.pkl', 'wb') as fp:
        pickle.dump(baseline_data, fp)
    
    baselines.append(baseline_data)
    
    
    
#%%
"""
#---------------------------------------------------------------------
# (DISCARDED) Calculate EDA Features



# Smoothing Step from Original Study
# filtered_eda_data = eda_data.reset_index(drop=True)
#
#        with warnings.catch_warnings():
#            warnings.simplefilter("ignore")
#            eda_signal = SimpleExpSmoothing(eda_signal).fit(
#                smoothing_level=0.15).fittedvalues

signals, info = nk.eda_process(eda_data, sampling_rate=EDA_SF)

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
auc = np.sum(signals['EDA_Phasic'])*EDA_SF
    


baseline_data['scr_peaks_per_time_unit'] = peaks_per_time_unit
baseline_data['mean_scr_peak_amplitudes'] = mean_peak_amplitudes
baseline_data['sum_scr_peak_amplitudes'] = sum_peak_amplitudes
baseline_data['mean_scl'] = mean_scl
baseline_data['scr_auc'] = auc
"""
