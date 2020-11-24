#!/usr/bin/env python3

# Make file with p-values and ISC/HMM-values, and voxels associated with those values
#

import glob
import tqdm
import numpy as np
import deepdish as dd
from HMM_settings import *

HMMdir = HMMpath+'shuff_5bins_train04_paper/'
ISCdir = ISCpath+'shuff_Yeo_paper/'
task='DM'

savedict = {}
for roi in tqdm.tqdm(glob.glob(roidir+'*h5')):
	roi_short = roi.split('/')[-1][:-3]
	savedict[roi_short] = {}
	savedict[roi_short]['vall'] = dd.io.load(roi,'/vall')
	ISCvals = dd.io.load(ISCdir+roi_short+'.h5','/'+task)
	for isc in ['ISC_e','ISC_g']:
		savedict[roi_short][isc] = {}
		savedict[roi_short][isc]['val'] = np.nanmean(ISCvals[isc][0])
		savedict[roi_short][isc]['shuff'] = np.nanmean(ISCvals[isc][1:],axis=1)
		if 'g' in isc:
			savedict[roi_short][isc]['p'] = np.sum(abs(savedict[roi_short][isc]['val']) > savedict[roi_short][isc]['shuff'])/len(savedict[roi_short][isc]['shuff'])
		else:
			savedict[roi_short][isc]['p'] = np.sum(abs(savedict[roi_short][isc]['val']) < abs(savedict[roi_short][isc]['shuff']))/len(savedict[roi_short][isc]['shuff'])
			
	if (df.index == roi_short).any():
		savedict[roi_short]['k_diff'] = {}
		savedict[roi_short]['k_diff']['val'] = df.loc[roi_short]['4'] - df.loc[roi_short]['0']
		savedict[roi_short]['k_diff']['shuff'] = df.loc[roi_short]['shuff']
		savedict[roi_short]['k_diff']['p'] = df.loc[roi_short]['k_diff_p']
		savedict[roi_short]['k_diff']['q'] = df.loc[roi_short]['k_diff_q']
		
		HMMvals = dd.io.load(HMMdir+roi_short+'.h5','/'+task)
		for HMMd in ['ll_diff','auc_diff']:
			savedict[roi_short][HMMd] = {}
			savedict[roi_short][HMMd]['val'] = HMMvals[HMMd][0]
			savedict[roi_short][HMMd]['shuff'] = HMMvals[HMMd][1:]
			savedict[roi_short][HMMd]['p'] = np.sum(abs(savedict[roi_short][HMMd]['val'])<\
				abs(savedict[roi_short][HMMd]['shuff']))/len(savedict[roi_short][HMMd]['shuff'])
			
for comp in ['ISC_e','ISC_g','ll_diff','auc_diff']:
	ROIl = []
	ps = []
	qs = []
	for roi in savedict.keys():
		if comp in savedict[roi].keys():
			ROIl.append(roi)
			ps.append(savedict[roi][comp]['p'])
	qs = FDR_p(np.array(ps))
	for i,roi in enumerate(ROIl):
		savedict[roi][comp]['q'] = qs[i]
		
dd.io.save(ISCpath+'p_vals_paper.h5',savedict)
			
	



