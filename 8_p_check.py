#!/usr/bin/env python3

# Make file with p-values and ISC/HMM-values, and voxels associated with those values
#

import glob
import tqdm
import numpy as np
import deepdish as dd
from settings import *

HMMdir = HMMpath+'shuff_5bins_train04/'
ISCdir = ISCpath+'shuff_Yeo/'
vals = ['ISC_e','ISC_g','ll_diff','auc_diff']
task='DM'

savedict = {}
for roi in tqdm.tqdm(glob.glob(ISCdir+'*h5')):
	roi_short = roi.split('/')[-1][:-3]
	savedict[roi_short] = {}
	ISCvals = dd.io.load(roi,'/'+task)
	#savedict[roi_short]['vall'] = ISCvals['vall']
	for isc in ['ISC_e','ISC_g']:
		savedict[roi_short][isc] = {}
		savedict[roi_short][isc]['val'] = np.nanmean(ISCvals[isc][0])
		savedict[roi_short][isc]['shuff'] = np.nanmean(ISCvals[isc][1:],axis=1)
		if 'g' in isc:
			savedict[roi_short][isc]['p'] = np.sum(abs(savedict[roi_short][isc]['val']) > savedict[roi_short][isc]['shuff'])/len(savedict[roi_short][isc]['shuff'])
		else:
			savedict[roi_short][isc]['p'] = np.sum(abs(savedict[roi_short][isc]['val']) < abs(savedict[roi_short][isc]['shuff']))/len(savedict[roi_short][isc]['shuff'])
	
	HMMvals = dd.io.load(HMMdir+roi_short+'.h5','/'+task)
	savedict[roi_short]['vall'] = HMMvals['vall']
	for HMMd in ['ll_diff','auc_diff']:
		savedict[roi_short][HMMd] = {}
		if HMMd not in HMMvals.keys():
			savedict[roi_short][HMMd]['val'] = HMMvals['shuff_0'][HMMd]
			if HMMd == 'auc_diff': savedict[roi_short][HMMd]['val']=savedict[roi_short][HMMd]['val']/4
			savedict[roi_short][HMMd]['shuff'] = np.array([abs(HMMvals[shuff][HMMd])\
						 	for shuff in HMMvals.keys() \
						 	if 'shuff' in shuff and 'shuff_0' not in shuff])
			if HMMd == 'auc_diff': savedict[roi_short][HMMd]['shuff']=np.array([a/4 for a in savedict[roi_short][HMMd]['shuff']])
		else:
			savedict[roi_short][HMMd]['val'] = HMMvals[HMMd][0]
			savedict[roi_short][HMMd]['shuff'] = HMMvals[HMMd][1:]
		savedict[roi_short][HMMd]['p'] = np.sum(abs(savedict[roi_short][HMMd]['val'])<\
			abs(savedict[roi_short][HMMd]['shuff']))/len(savedict[roi_short][HMMd]['shuff'])
		
		
dd.io.save(ISCpath+'p_vals.h5',savedict)
			
	



