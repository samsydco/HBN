#!/usr/bin/env python3

# Make file with p-values and ISC/HMM-values, and voxels associated with those values
#

import glob
import tqdm
import numpy as np
import deepdish as dd
from HMM_settings import *

HMMdir = HMMpath+'shuff_5bins_train04_outlier_'
ISCdir = ISCpath+'shuff_Yeo_outlier'
task='DM'
nTR_ = 750

savedict = {}
for seed in tqdm.tqdm(seeds):
	savedict[seed] = {}
	df=pd.DataFrame(dd.io.load(nkh5+seed+'.h5')).T
	df.loc[:,'k_diff_q'] = FDR_p(df['k_diff_p'])
	for roi in glob.glob(roidir+seed+'/'+'*h5'):
		roi_short = roi.split('/')[-1][:-3]
		savedict[seed][roi_short] = {}
		savedict[seed][roi_short]['vall'] = dd.io.load(roi,'/vall')
		for isc in ['ISC_w','ISC_e','ISC_g','ISC_yy-yo']:
			if 'yy' not in isc:
				ISCvals = dd.io.load(ISCdir+seed+'/'+roi_short+'.h5','/'+task+'/'+isc)
			else: # Young-Young ISC minus Young-Old ISC:
				ISCvals = dd.io.load(ISCdir+seed+'/'+roi_short+'.h5','/'+task+'/ISC_w')[:,0] - np.nanmean(dd.io.load(ISCdir+seed+'/'+roi_short+'.h5','/'+task+'/ISC_b'),1)
			savedict[seed][roi_short][isc] = {}
			if 'w' in isc:
				savedict[seed][roi_short][isc]['val'] = np.nanmean(ISCvals[1:])
				savedict[seed][roi_short][isc]['val_'] = np.nanmean(ISCvals[0],-1)
			else:
				savedict[seed][roi_short][isc]['val'] = np.nanmean(ISCvals[0])
				savedict[seed][roi_short][isc]['shuff'] = np.nanmean(ISCvals[1:],axis=1)
				if 'g' in isc:
					savedict[seed][roi_short][isc]['p'] = np.sum(abs(savedict[seed][roi_short][isc]['val']) > savedict[seed][roi_short][isc]['shuff'])/len(savedict[seed][roi_short][isc]['shuff'])
				else:
					savedict[seed][roi_short][isc]['p'] = np.sum(abs(savedict[seed][roi_short][isc]['val']) < abs(savedict[seed][roi_short][isc]['shuff']))/len(savedict[seed][roi_short][isc]['shuff'])
		
		if roi_short in ROIl:
			for b in ['0','4']:
				savedict[seed][roi_short]['k'+b] = {}
				savedict[seed][roi_short]['k'+b]['val'] = np.round(TR*(nTR_/df.loc[roi_short][b]),2)
			savedict[seed][roi_short]['k_diff'] = {}
			savedict[seed][roi_short]['k_diff']['val'] = df.loc[roi_short]['4'] - df.loc[roi_short]['0']
			savedict[seed][roi_short]['k_diff']['shuff'] = df.loc[roi_short]['shuff']
			savedict[seed][roi_short]['k_diff']['p'] = df.loc[roi_short]['k_diff_p']
			savedict[seed][roi_short]['k_diff']['q'] = df.loc[roi_short]['k_diff_q']
	
			for HMMd in ['ll_diff','auc_diff','tune_ll_perm']:
				HMMvals = dd.io.load(HMMdir+seed+'/'+roi_short+'.h5','/'+HMMd)
				savedict[seed][roi_short][HMMd] = {}
				# dumb coding error
				if HMMd == 'auc_diff': 
					k = dd.io.load(HMMdir+seed+'/'+roi_short+'.h5','/best_k')
					HMMvals = (HMMvals*k)/(k-1)
				if 'tune_ll' not in HMMd:
					savedict[seed][roi_short][HMMd]['val'] = HMMvals[0]
					savedict[seed][roi_short][HMMd]['shuff'] = HMMvals[1:]
					savedict[seed][roi_short][HMMd]['p'] = np.sum(abs(savedict[seed][roi_short][HMMd]['val']) < abs(savedict[seed][roi_short][HMMd]['shuff']))/len(savedict[seed][roi_short][HMMd]['shuff'])
				else:
					savedict[seed][roi_short][HMMd]['val_'] = np.take(np.mean(HMMvals[0],axis=1),[0,-1])/nTR_
				
roidict = {}
for roi in glob.glob(roidir+seed+'/'+'*h5'):
	roi_short = roi.split('/')[-1][:-3]
	roidict[roi_short] = {}
	for comp in ['ISC_w','ISC_e','ISC_g','ISC_yy-yo', 'k0','k4','k_diff','ll_diff','auc_diff','tune_ll_perm']:
		if comp in savedict[seed][roi_short].keys():
			roidict[roi_short][comp] = {}
			if comp != 'tune_ll_perm':
				roidict[roi_short][comp]['val'] = np.mean([savedict[seed][roi_short][comp]['val'] for seed in seeds])
			if any(a==comp for a in ['ISC_w','tune_ll_perm']):
				for bi,b in enumerate(bins):
					roidict[roi_short][comp][str(b)] = np.mean([savedict[seed][roi_short][comp]['val_'][bi] for seed in seeds])
			if not any(n in comp for n in ['w','0','4','perm']):
				arrs = [np.array(savedict[seed][roi_short][comp]['shuff']) for seed in seeds]
				arr = np.ma.empty((np.max([len(i) for i in arrs]),len(arrs)))
				arr.mask = True
				for idx, l in enumerate(arrs):
					arr[:len(l),idx] = l
				roidict[roi_short][comp]['shuff'] = arr.mean(axis = -1)
				if 'g' in comp:
					roidict[roi_short][comp]['p'] = np.sum(abs(roidict[roi_short][comp]['val']) > roidict[roi_short][comp]['shuff'])/len(roidict[roi_short][comp]['shuff'])
				else:
					roidict[roi_short][comp]['p'] = np.sum(abs(roidict[roi_short][comp]['val']) < abs(roidict[roi_short][comp]['shuff']))/len(roidict[roi_short][comp]['shuff'])
				
for seed in seeds+['-']:
	d = roidict if seed == '-' else savedict[seed]
	for comp in ['ISC_e','ISC_g','ISC_yy-yo','ll_diff','auc_diff']:
		ROIl = []
		ps = []
		qs = []
		for roi in d.keys():
			if comp in d[roi].keys():
				ROIl.append(roi)
				if d[roi][comp]['p'] == 0:
					ps.append(1/(len(d[roi][comp]['shuff'])+1))
				else:
					ps.append(d[roi][comp]['p'])
		qs = FDR_p(np.array(ps))
		for i,roi in enumerate(ROIl):
			if seed == '-':
				roidict[roi][comp]['q'] = qs[i]
			else:
				savedict[seed][roi][comp]['q'] = qs[i]
		print(seed,comp,np.sum(qs<0.05))
		
dd.io.save(pvals_file,{'seeddict':savedict,'roidict':roidict})
			
	



