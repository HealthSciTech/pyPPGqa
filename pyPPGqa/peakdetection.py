"""
Functions to detect peaks of an ECG data and extract time-frequency features.

This module provides functions that process ECG data and extract R-peak locations
and extract HRV features.

Copyright 2020, Emad Kasaeyan Naeini
Licence: MIT, see LICENCE for more details.

"""
import numpy as np
import pandas as pd
import csv
from itertools import islice
import os, time, glob
import heartpy as hp

def peakDetection(signal, sr=512, wd={}):
    # Get peak location indices and their probabilities
    peaks, probs = dt.find_peaks(signal)

    # Filter out R-peaks that occur unnaturally close
    filtered_peaks = dt.remove_close(peaks=peaks, peak_probs=probs, threshold_ms=300)
    wd['peaklist'] = filtered_peaks
    wd['ybeat'] = [signal[x] for x in filtered_peaks]
    return wd

def featureExtraction(ecg, sr=512, freq_method='welch', clean_rr_method='quotient-filter', m={}, wd={}):
    t_start = time.perf_counter()
    peakDetection(ecg, sr=sr, wd=wd)
    wd = hp.analysis.calc_rr(wd['peaklist'], sample_rate=sr, working_data=wd)
    wd = hp.peakdetection.check_peaks(wd['RR_list'], wd['peaklist'], wd['ybeat'], reject_segmentwise=True, working_data=wd)
    wd = hp.analysis.clean_rr_intervals(wd, method=clean_rr_method)
    wd, m = hp.analysis.calc_ts_measures(wd['RR_list_cor'], wd['RR_diff'],
                                              wd['RR_sqdiff'], measures=m, 
                                              working_data=wd)
    wd, m = hp.analysis.calc_fd_measures(method=freq_method, measures=m, working_data=wd)
    print('\nFinished in {:.8s} sec'.format(time.perf_counter()-t_start))
    return wd, m

def ecg2peak5min(dt, sr=512):
    t_start = time.perf_counter()
    ecg_dir = 'data/all5min/ecg5min/'
    ecg_files = sorted(glob.glob(ecg_dir+'*'))
    peak_dir = 'data/all5min/peak5min/'
    ecg_hrv_dir =  'data/all5min/ecg_hrv/'
    if not os.path.isdir(peak_dir):
        os.makedirs(peak_dir)
        print( 'Directory ' + peak_dir + ' is Created!')
    if not os.path.isdir(ecg_hrv_dir):
        os.makedirs(ecg_hrv_dir)
        print( 'Directory ' + ecg_hrv_dir + ' is Created!')

    for i in range(len(ecg_files)):
        sub = os.path.basename(ecg_files[i]).split('_')[0]
        ind = os.path.basename(ecg_files[i]).split('_')[1][:2]
        print(sub, ind)
        ecg_peak = peak_dir +  '{}_{}_peak.csv'.format(sub,ind)
        ecg_hrv = ecg_hrv_dir + '{}_{}_hrv.csv'.format(sub,ind)
        print()
        print('ECG source  --> \t', ecg_files[i])
        print('Destination --> \t', ecg_hrv)
#         df_prv = pd.read_csv(prv_files[ind], skipinitialspace=True)
        
        if os.path.exists(ecg_hrv):
            print( ecg_hrv + ' is Created!')
            continue
        
        feature_cols = ['timestamp', 'ecg_hr_mean', 'ecg_nni_mean', 'ecg_rmssd', 'ecg_sdnn',
                       'ecg_lf_fft_abs', 'ecg_hf_fft_abs', 'ecg_fft_ratio']
        df_ecgHRV = pd.DataFrame(columns=feature_cols)
        # df_peaks = pd.DataFrame()
        src = open(ecg_files[i])
        be = 0
        fi = None
        with open(ecg_peak,'a') as dst:
            newFileWriter = csv.writer(dst)
            for line in islice(src, be, fi):
                record = np.array([line.split(",")])
                if len(record[0])<2:
                    ts_ecg = record[0][0].astype(np.float)
                    peaks_list = [ts_ecg]+[np.nan]
                    hrvParams = [ts_ecg]+[np.nan]*7
                else:
                    record = np.delete(record,-1).astype(np.float)
                    ts_ecg = record[0]
                    ecg = record[1:]
    #                 ecg = hp.remove_baseline_wander(ecg, sample_rate=512)
                    # ecg = hp.scale_data(ecg)
                    wd = {}
                    m = {}
                    wd, m = emad_main(ecg, sr, freq_method='welch', clean_rr_method='quotient-filter', m=m, wd=wd)
                    peaks_list = [ts_ecg]+wd['peaklist']
                    hrvParams = [m['bpm'], m['ibi'], m['rmssd'], m['sdnn'], m['lf'], m['hf'], m['lf/hf']]
                    hrvParams = [ts_ecg]+hrvParams
                newFileWriter.writerow(peaks_list)
                df_ecgHRV = df_ecgHRV.append(pd.Series(hrvParams, index=df_ecgHRV.columns), ignore_index=True)

            # df_peaks.to_csv(ecg_peak, index=False)
            df_ecgHRV.to_csv(ecg_hrv, index=False)
        
    print('\nFinished in %.8s sec' %(time.perf_counter()-start))
    