import numpy as np
import pandas as pd
import csv
from itertools import islice
import os, time, glob
import heartpy as hp
from ecg2rr import detector
from matplotlib import pyplot as plt

from preprocessing import sh2ecg, ecgSegmentize
from peakdetection import peakDetection, featureExtraction
###############################
###  General Initialization ###
###############################
os.chdir('/Users/emad/Desktop/uci/papers/chase2020')
print('Directory changed to ' + os.path.basename(os.getcwd()))

# list of user ids included in analysis 
#IDs 6,8 are not included due to lack of enough info
subs = ['ID'+str("{0:0=1d}".format(i)) for i in range(1,11)]
subs.remove('ID6')
subs.remove('ID8')

sensor_names = ["Shimmer3"]
samplerate_sh = 512
window_size = 60
ws_filter = 1


#Test sh2ecg()
#sh2ecg(sub='ID39')

# os.chdir('./data/all5min/5min')
# print('Directory changed to ' + os.path.basename(os.getcwd()))
# hrv_files = sorted(glob.glob(os.getcwd()+'/*'))
# for i in range(len(hrv_files)):
#     # tmp = os.path.basename(hrv_files[i])
#     splt = hrv_files[i].split('_')

#     if len(splt[1])==8:
#         os.rename(hrv_files[i],splt[0]+'_0'+splt[1][:-7]+'p'+splt[1][-6:])
#         # print(splt[0]+'_0'+splt[1][:-7]+'p'+splt[1][-6:])
#     else:
#         os.rename(hrv_files[i],splt[0]+'_'+splt[1][:-7]+'p'+splt[1][-6:])
#         # print(splt[0]+'_'+splt[1][:-7]+'p'+splt[1][-6:])