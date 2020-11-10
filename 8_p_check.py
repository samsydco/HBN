#!/usr/bin/env python3

# Make file with p-values and ISC/HMM-values, and voxels associated with those values
#

import glob
import tqdm
import numpy as np
import deepdish as dd
from settings import *

HMMdir = HMMpath+'shuff_5bins_train04_paper/'
ISCdir = ISCpath+'shuff_Yeo_paper/'
vals = ['ISC_e','ISC_g','ll_diff','auc_diff']
task='DM'

def FDR_p(pvals):
    # Port of AFNI mri_fdrize.c

    # Ensure p values are valid, and not exactly equal to 0 or 1
    assert np.all(pvals >= 0) and np.all(pvals <= 1)
    pvals[pvals < np.finfo(np.float_).eps] = np.finfo(np.float_).eps
    pvals[pvals == 1] = 1-np.finfo(np.float_).eps
    n = pvals.shape[0]

    # Compute q using step-down procedure
    qvals = np.zeros((n))
    sorted_ind = np.argsort(pvals)
    sorted_pvals = pvals[sorted_ind]
    qmin = 1.0
    for i in range(n-1, -1, -1):
        qval = (n * sorted_pvals[i])/(i+1)
        if qval > qmin:
            qval = qmin
        else:
            qmin = qval
        qvals[sorted_ind[i]] = qval

    # Estimate number of true positives m1 and adjust q
    if n >= 233:
        phist = np.histogram(pvals, bins=20, range=(0, 1))[0]
        sorted_phist = np.sort(phist[3:19])
        if np.sum(sorted_phist) >= 160:
            median4 = n - 20*np.dot(np.array([1, 2, 2, 1]),
                                    sorted_phist[6:10])/6
            median6 = n - 20*np.dot(np.array([1, 2, 2, 2, 2, 1]),
                                    sorted_phist[5:11])/10
            m1 = min(median4, median6)

            qfac = (n - m1)/n
            if qfac < 0.5:
                qfac = 0.25 + qfac**2
            qvals *= qfac

    return qvals

savedict = {}
for roi in tqdm.tqdm(glob.glob(ISCdir+'*h5')):
	roi_short = roi.split('/')[-1][:-3]
	savedict[roi_short] = {}
	ISCvals = dd.io.load(roi,'/'+task)
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
			
	



