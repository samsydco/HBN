#!/usr/bin/env python3

# Input: Age bin file from 6_ISC_agebin
# Output: different per-age analyses

import os
import glob
import h5py
import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,spearmanr
from datetime import date
import numpy as np
import deepdish as dd
from settings import *
from ISC_settings import *
agediffroidir = path+'ROIs/agediff/'
gdiffroidir = path+'ROIs/g_diff/'
smooth = True
smoothtimes = 6

# what voxels change in ISC with age, and in which direction?
iscf = ISCpath + 'ISC_2019-09-06_age_2.h5'#'ISC_2019-08-13_age_2.h5'
agediff_f = ISCpath + 'ISC_' + str(date.today())+'_agediff'
agediff_f = agediff_f+'_2' if 'age_2' in iscf and not smooth else agediff_f+'_smooth' if smooth else agediff_f
agediff_f = agediff_f+'.h5'
if os.path.exists(agediff_f):
    os.remove(agediff_f)
	
if smooth:
	global cols
	cols = {}
	for hem in ['left','right']:
		hemi = 'lh' if hem == 'left' else 'rh'
		X = (dd.io.load('/data/Schema/intact/fsaverage6_adj.h5','/'+hem))
		cols[hemi] = [None] * (len(X['jc'])-1)
		for i in range(len(cols[hemi])):
			cols[hemi][i] = X['ir'][X['jc'][i]:X['jc'][i+1]]
	def smooth_fun(ISC):
		hl = len(ISC)//2
		ISC2 = np.zeros(ISC.shape)
		for hemi in ['lh','rh']:
			st = 0 if hemi == 'lh' else hl
			colh = cols[hemi]
			for idx in range(hl):
				if ~np.isnan(ISC[idx+st]):
					ISC2[idx+st] = np.nanmean([ISC[idx+st],np.nanmean(ISC[colh[idx]+st])])
				else:
					ISC2[idx+st] = np.nan
		return ISC2

anall = ['corr','spearman','stddivmean','diffidx','diff','yodiff']
with h5py.File(agediff_f,'a') as hf:
	for task in ['DM','TP']:
		grp = hf.create_group(task)
		for comp in ['all']:#['0','1','all','err_diff','g_diff']:
			print(task,comp)
			#verts = {key: np.zeros(81924) for key in anall}
			for shuff in tqdm.tqdm(list(dd.io.load(iscf).keys())):
				ISCs = {key: np.zeros((81924,smoothtimes)) for key in anall}
				ISC = np.zeros((nbinseq,81924))
				ISCsm = np.zeros((nbinseq,81924,smoothtimes))
				for b in range(nbinseq):
					if comp != 'all' and 'age_2' in iscf:
						print('Must do computation over "all" with this iscf')
						break
					elif comp == 'all' and 'age_2' in iscf:
						ISC[b] = dd.io.load(iscf,
								 '/'+shuff+'/bin_'+str(b)+'/'+task+'/ISC_SH')
					elif comp in ['0','1'] and 'age_2' not in iscf:
						ISC[b] = dd.io.load(iscf,
							'/'+shuff+'/bin_'+str(b)+'/'+task+'/ISC_SH_w_'+s)
					elif comp == 'err_diff' and 'age_2' not in iscf:
						ISC[b] = dd.io.load(iscf,
						'/'+shuff+'/bin_'+str(b)+'/'+task+'/ISC_SH_w_1') - \
						dd.io.load(iscf,
						'/'+shuff+'/bin_'+str(b)+'/'+task+'/ISC_SH_w_0')
					elif comp == 'g_diff' and 'age_2' not in iscf:
						for i in ['ISC_SH_b_0_0', 'ISC_SH_b_0_1', 'ISC_SH_b_1_0', 'ISC_SH_b_1_1']:
							ISC[b] += dd.io.load(iscf,
							'/'+shuff+'/bin_'+str(b)+'/'+task+'/'+i)
						ISC[b] = ISC[b]/4/ \
						(np.sqrt(dd.io.load(iscf,
						'/'+shuff+'/bin_'+str(b)+'/'+task+'/ISC_SH_w_1'))
						*np.sqrt(dd.io.load(iscf,
						'/'+shuff+'/bin_'+str(b)+'/'+task+'/ISC_SH_w_0')))
					elif comp == 'all' and 'age_2' not in iscf:
						for i in list(dd.io.load(iscf)['shuff_0']['bin_0'][task].keys()):
							ISC[b] += dd.io.load(iscf,
							'/'+shuff+'/bin_'+str(b)+'/'+task+'/'+i)
						ISC[b] = ISC[b]/6
					for s in range(smoothtimes):
						ISC[b] = smooth_fun(ISC[b])
						ISCsm[b,:,s] = ISC[b]
				for v in range(81924):
					for s in range(smoothtimes):
						ISCs['yodiff'][v,s] = ISCsm[0,v,s] - ISCsm[-1,v,s]
						ISCs['corr'][v,s] = np.corrcoef([e+agespan/2 for e in eqbins[:-1]],ISCsm[:,v,s])[0,1]
						diffidx = np.argmax(abs(np.diff(ISCsm[:,v,s])))
						if np.sum(np.isnan(ISCsm[:,v,s]))<4:
							ISCs['spearman'][v,s] = spearmanr([e+agespan/2 for e in eqbins[:-1]],ISCsm[:,v,s])[0]
							ISCs['diffidx'][v,s] = (diffidx+1)*np.sign(np.diff(ISCsm[:,v,s])[diffidx])
							ISCs['diff'][v,s] = np.max(abs(np.diff(ISCsm[:,v,s])))*np.sign(np.diff(ISCsm[:,v,s])[diffidx])
						else:
							ISCs['spearman'][v,s] = np.nan
							ISCs['diffidx'][v,s] = np.nan
							ISCs['diff'][v,s] = np.nan
						ISCs['stddivmean'][v,s] = np.std(ISCsm[:,v,s])/np.mean(ISCsm[:,v,s])
				for k in ISCs.keys():
					for s in range(smoothtimes):
						grp.create_dataset('ISC_'+k+'_'+comp+'_sm'+str(s)+'_'+shuff,data=ISCs[k][:,s])
						
			

		
	
