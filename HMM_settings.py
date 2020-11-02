#!/usr/bin/env python3

# HMM Settings
from settings import *
from ISC_settings import *
from event_ratings import event_list,ev_conv,xcorr

tasks = ['DM','TP']
TR=0.8
nTR=[750,250]

TR1 = 12 #12 sec
TR2 = 300 #300 sec (5 min)
k_list = np.unique(np.round((10*60)/np.arange(TR1,TR2,TR1))).astype(int)
nsplit = 5
bins = [0,nbinseq-1]
nshuff = 100

