"""
Functions to prepare 5 minute training data.

This module provides functions that read data from Shimmer ECG sesnor,
extract data and segmentize data into 5 minutes.

Copyright 2020, Emad Kasaeyan Naeini
Licence: MIT, see LICENCE for more details.

"""
def sh2ecg(sub, be=3, fi=None):
    start = timeit.default_timer()
    sh3 = 'data/{}/ecg/'.format(sub)
    src = open(sh3+'{}_Shimmer3.csv'.format(sub))
    with open(sh3+'{}_ecg.csv'.format(sub),'a') as dst:
        newFileWriter = csv.writer(dst)
        newFileWriter.writerow(['timestamp', 'ecg'])
        for line in islice(src,be,fi):
            record = np.array([line.split("\t")])
            record = np.delete(record,-1).astype(np.float)
            newFileWriter.writerow([record[0],record[9]])

    gc.collect()
    src.close()

    stop = timeit.default_timer()
    print('Time: ', stop-start)

def ecgSegmentize(subs, sr=512):
    t_start = time.perf_counter()
    hrv_files = sorted(glob.glob('data/all5min/hrv5min/*'))
    ecg_files = dict.fromkeys(subs)
    for sub in subs:
        ecg_files[sub] = 'data/{}/ecg/{}_ecg.csv'.format(sub, sub)
    ecg_dir = 'data/all5min/ecg5min/'
    if not os.path.isdir(ecg_dir):
        os.makedirs(ecg_dir)
        
    for i in range(len(hrv_files)):
        sub = hrv_files[i][-13:-10]
        ind = hrv_files[i][-9:-7]
        print('User:\t' + sub)
        print('Index:\t' + ind)
        ecg_sync = ecg_dir+'{}_{:02d}_ecg.csv'.format(sub,ind)
        print()
        print(ecg_sync)
        if os.path.exists(ecg_sync):
            print( ecg_sync + ' is Created!')
            continue
        df_hrv = pd.read_csv(hrv_files[i], skipinitialspace=True, usecols=['shiftedTimestamp'])
        df_ecg = pd.read_csv(ecg_files[sub])

        ws = 5*60 * sr # 60 sec
        with open(ecg_sync,'a') as dst:
            newFileWriter = csv.writer(dst)
            for i in range(len(df_hrv)):
                ts_hrv = df_hrv['shiftedTimestamp'].iloc[i]
                be_idx = (df_ecg['timestamp']-ts_hrv).abs().idxmin()
                fi_idx = be_idx+ws
                signal = df_ecg['ecg'].values[be_idx :fi_idx]
                signal = signal.tolist()
                newFileWriter.writerow([ts_hrv]+signal)
        
    print('\nFinished Segmenting in 5 minutes for user {} in {:.8s} sec'.format(sub, time.perf_counter()-t1))