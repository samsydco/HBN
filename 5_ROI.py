#!/usr/bin/env python3

# Grab RSC ROI and save in h5 file
import time
import h5py
import glob
import numpy as np
import deepdish as dd
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from settings import *

subs = glob.glob(prepath+'sub*.h5')
#subs=['/data/HBN/test2/fmriprep_output/fmriprep/PythonData/sub-NDARAW179AYF_copy.h5']

hemis = ['lh','rh']
froi = h5py.File('/data/Schema/intact/Yeo17net.h5', 'r')
f6 = h5py.File('/data/Schema/intact/fsaverage6_adj.h5', 'r')
roi = 16
RSCroi ={}
for hemi in hemis:
	cnum = 0 if hemi == 'lh' else 1 # cluster number associated with RSC for 5 cluster parcel
	rois = np.nan_to_num(froi[hemi][:][0])
	dispcoords = f6[hemi+'inflatedcoords'][:]
	tt = [np.zeros(coord.shape) if rois[idx]!=roi else coord for idx,coord in enumerate(np.transpose(dispcoords))]
	clustering = AgglomerativeClustering(n_clusters=5).fit(tt)
	RSCroi[hemi] = clustering.labels_==cnum
dd.io.save(ISCpath+'RSC.h5',RSCroi)
RSCroi = dd.io.load(ISCpath+'RSC.h5')

ISCclust = dd.io.load(ISCpath+'ISCclusters_5.h5')

for sub in subs:
    with h5py.File(sub) as f:
		for task in ['DM','TP']:
			if 'RSC' in list(f[task].keys()): del f[task]['RSC']
			rscdata = []
			for hemi in hemis:
				hem = 'L' if hemi == 'lh' else 'R'
				temp = f[task][hem][RSCroi[hemi],:]
				rscdata.append(np.delete(temp,np.unique([i[0] for i in np.argwhere(np.isnan(temp))]),axis=0))
				if 'ISC_'+hem in list(f[task].keys()): del f[task]['ISC_'+hem]
				grp = f[task].create_group('ISC_'+hem)
				isctask = 'TP' if task == 'DM' else 'DM'
				for i,r in enumerate(ISCclust['clusters'][isctask][hemi]):
					grp.create_dataset(str(i), data=f[task][hem][[i for i,x in enumerate(r) if x==1],:])
			f[task].create_dataset('RSC', data=np.concatenate(rscdata))
			
			
				





''' 
sub_rsc = [] 
sub_a1 = []
for sub in subs:
    with h5py.File(sub) as f:
        for task in ['DM','TP']:
            if not any("RSC" == s for s in list(f[task].keys())):
                sub_rsc.append(sub)
            if not any("A1" == s for s in list(f[task].keys())):
                sub_a1.append(sub)  
            
'''