#!/usr/bin/env python3

# For each ROI:
# Subtract: max-ll - min-ll
# Look for where max-ll - min-ll > 0 or some threshold
# Also include: min(tune_ll) at optimal number of events

import glob
import tqdm
import numpy as np
import deepdish as dd
from HMM_settings import *

HMMdir = HMMpath+'shuff_5bins_train04/'
nkdir = HMMpath+'nk_moreshuff/'
task= 'DM'
nTR = nTR[0]
trainl = ['together','seperate']

lldict = {k:{} for k in trainl}
for roi in glob.glob(HMMdir+'*.h5'):
	roi_short = roi.split('/')[-1][:-3]
	for train in trainl: lldict[train][roi_short] = {}
	ll = dd.io.load(roi,'/'+task+'/tune_ll')
	nbins,nsplit,nk = ll.shape
	for b in [0,4]:
		ll_sep = dd.io.load(nkdir+roi_short+'.h5','/'+str(b)+'/tune_ll')
		lldict['together'][roi_short][str(b)+'_ll_max'] = np.max(np.mean(ll[b],axis=0))/nTR
		lldict['seperate'][roi_short][str(b)+'_ll_max'] = np.max(np.mean(ll_sep[0],axis=0))/nTR
		lldict['together'][roi_short][str(b)+'_ll_min'] = np.min(np.mean(ll[b],axis=0))/nTR
		lldict['seperate'][roi_short][str(b)+'_ll_min'] = np.min(np.mean(ll_sep[0],axis=0))/nTR
		lldict['together'][roi_short][str(b)+'_ll_diff'] = \
		lldict['together'][roi_short][str(b)+'_ll_max'] - \
		lldict['together'][roi_short][str(b)+'_ll_min']
		lldict['seperate'][roi_short][str(b)+'_ll_diff'] = \
		lldict['seperate'][roi_short][str(b)+'_ll_max'] - \
		lldict['seperate'][roi_short][str(b)+'_ll_min']
		lldict['together'][roi_short][str(b)+'_2k'] = np.mean(ll[b,:,0],axis=0)/nTR
		lldict['seperate'][roi_short][str(b)+'_2k'] = np.mean(ll_sep[0,:,0],axis=0)/nTR
		lldict['together'][roi_short][str(b)+'_2k_diff'] = \
		lldict['together'][roi_short][str(b)+'_ll_max'] - \
		lldict['together'][roi_short][str(b)+'_2k']
		lldict['seperate'][roi_short][str(b)+'_2k_diff'] = \
		lldict['seperate'][roi_short][str(b)+'_ll_max'] - \
		lldict['seperate'][roi_short][str(b)+'_2k']
		
dd.io.save(HMMpath+'ll_diff.h5',lldict)
lldict = dd.io.load(HMMpath+'ll_diff.h5')

comp = '_2k_diff'
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 40})
for train in trainl:
	df=pd.DataFrame(lldict[train]).T
	df=df.drop(columns=['0_ll_diff','4_ll_diff','0_ll_max','4_ll_max','0_2k','4_2k'])
	thresh = 0.005 if train=='together' else 0.002
	x = df['0'+comp]; y = df['4'+comp]
	idx1 = np.intersect1d(np.where(y<thresh)[0],np.where(x<thresh)[0])
	idx2 = np.unique(np.concatenate((np.where(y>thresh)[0],np.where(x>thresh)[0])))
	x1 = x.iloc[idx1]; y1 = y.iloc[idx1]
	x2 = x.iloc[idx2]; y2 = y.iloc[idx2]
	fig, ax = plt.subplots(figsize=(15,15))
	ax.scatter(x1,y1,s=100,alpha=0.25)
	ax.scatter(x2,y2,s=100,alpha=0.25)
	ax.set_xlim([np.min(x)-0.005,np.max(x)+0.005])
	ax.set_ylim([np.min(y)-0.005,np.max(y)+0.005])
	ax.set_xlabel('Youngest '+comp[1:])
	ax.set_ylabel('Oldest '+comp[1:])
	ax.set_title('Outlier: '+df.loc[df['4'+comp]==df['4'+comp].max()].index[0])
	fig.savefig(figurepath+'HMM/ll/'+train+comp++'_'+str(thresh)+'.png', bbox_inches='tight')


