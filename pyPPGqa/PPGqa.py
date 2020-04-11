import numpy as np
import pandas as pd
import csv
from itertools import islice
import os, time, glob
import heartpy as hp
from ecg2rr import detector
from matplotlib import pyplot as plt

from .preprocessing import sh2ecg, ecgSegmentize
from .peakdetection import peakDetection, featureExtraction
###############################
###  General Initialization ###
###############################

# list of user ids included in analysis 
#IDs 6,8 are not included due to lack of enough info
subs = ['ID'+str("{0:0=1d}".format(i)) for i in range(1,11)]
subs.remove('ID6')
subs.remove('ID8')

sensor_names = ["Shimmer3"]
samplerate_sh = 512
window_size = 60
ws_filter = 1


ecgSegmentize(subs, samplerate_sh)
