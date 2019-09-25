#!/usr/bin/env python3

# HMM Settings
from settings import *
from ISC_settings import *

nTR = 750
TR1 = 12 #12 sec
TR2 = 300 #300 sec (5 min)
k_list = np.unique(np.round((10*60)/np.arange(TR1,TR2,TR1))).astype(int)
nsplit = 5
bins = [0,nbinseq-1]
win_range = np.arange(7,12) #range of windows tested for within - across corr
nshuff = 100