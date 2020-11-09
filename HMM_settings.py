#!/usr/bin/env python3

# HMM Settings
from settings import *
from ISC_settings import *
from event_ratings import ev_conv
from sklearn.model_selection import KFold

roidir = ISCpath+'Yeo_parcellation/'
nkdir = HMMpath+'nk_moreshuff_paper/'#'nk/'
nkh5 = HMMpath+'nk_paper.h5' #formerly nk.h5
llh5 = HMMpath+'ll_diff_paper.h5' # formerly ll_diff

tasks = ['DM','TP']
TR=0.8
nTR=[750,250]

TR1 = 12 #12 sec
TR2 = 300 #300 sec (5 min)
k_list = np.unique(np.round((10*60)/np.arange(TR1,TR2,TR1))).astype(int)
nsplit = 5
bins = [0,nbinseq-1]
nshuff = 100
ll_thresh = 0.002

nsub= 40
y = [0]*int(np.floor(nsub/nsplit))*4+[1]*(int(np.floor(nsub/nsplit)))
kf = KFold(n_splits=nsplit, shuffle=True, random_state=2)

