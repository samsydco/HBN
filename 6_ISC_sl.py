#!/usr/bin/env python3

import os
import h5py
import tqdm
import numpy as np
import deepdish as dd
from datetime import date
from scipy.stats import pearsonr
from settings import *
from ISC_settings import *
SLlist = dd.io.load(ISCpath+'SLlist.h5')

def corr_col(X, Y):
    assert(X.shape[0] == Y.shape[0])
    cc = np.zeros(X.shape[0])
    for c in range(X.shape[0]):
        cc[c] = pearsonr(X[c], Y[c])[0]
    return cc

ISCf = 'ISC_'+str(date.today())+'_sl.h5'
if os.path.exists(ISCpath+ISCf):
    os.remove(ISCpath+ISCf)
dd.io.save(ISCpath+ISCf,{'subs':subord,'ages':agel,'phenodict':phenol,'pcs':pcl})
nsh = 1 #5 split-half iterations
for s in range(nsh):
	for task in ['DM','TP']:
		sh = dd.io.load(subord[0],['/'+task+'/L'])[0].shape
		for hem in ['L','R']:
			print('sh = ',s,task,hem)
			D = np.empty((len(subord),sh[0],sh[1]),dtype='float16')
			for sidx, sub in tqdm.tqdm(enumerate(subord)):
				D[sidx,:,:] = dd.io.load(sub,['/'+task+'/'+hem])[0]
			D = np.transpose(D,(1,2,0))
			n_vox,n_time,n_subj=D.shape
			nSL = len(SLlist[hem])
			with h5py.File(ISCpath+ISCf) as hf:
				grp = hf.create_group(task+str(s)+hem)
				for k,v in phenol.items():
					print(k)
					v2 = phenol['sex'] if k!='sex' else phenol['age']
					subh = even_out(v,v2)
					grp.create_dataset('subs_'+k,data=subh)
					ISCvox = np.zeros((n_vox,3))
					ISCcount = np.zeros((n_vox,3))
					for sl in range(nSL):
						Dsl = D[SLlist[hem][sl]]
						ISCvox[SLlist[hem][sl]] += \
						[corr_col(\
						np.nanmean(Dsl[:,:,subh[0][0]],axis=2),\
						np.nanmean(Dsl[:,:,subh[0][1]],axis=2)).mean(),
						corr_col(\
						np.nanmean(Dsl[:,:,subh[1][0]],axis=2),\
						np.nanmean(Dsl[:,:,subh[1][1]],axis=2)).mean(),
						(corr_col(\
						np.nanmean(Dsl[:,:,subh[0][0]],axis=2),\
						np.nanmean(Dsl[:,:,subh[1][0]],axis=2)).mean() +
               corr_col(np.nanmean(Dsl[:,:,subh[0][1]],axis=2),\
						np.nanmean(Dsl[:,:,subh[1][1]],axis=2)).mean() +
               corr_col(np.nanmean(Dsl[:,:,subh[0][0]],axis=2),\
						np.nanmean(Dsl[:,:,subh[1][1]],axis=2)).mean() +
               corr_col(np.nanmean(Dsl[:,:,subh[0][1]],axis=2),\
						np.nanmean(Dsl[:,:,subh[1][0]],axis=2)).mean())/4]
						ISCvox[np.isnan(ISCvox)] = 0
						ISCcount[SLlist[hem][sl]] += 1
					ISCcount[ISCcount == 0] = np.nan
					ISCvox = ISCvox / ISCcount
					ISCvox = np.column_stack((ISCvox, ISCvox[:,0]-ISCvox[:,1], ISCvox[:,2]/np.sqrt(ISCvox[:,0]*ISCvox[:,1])))
					ISCvox[np.isnan(ISCvox)] = 0
					grp.create_dataset('ISC_SL_'+k,data=ISCvox)




