#!/usr/bin/env python3

# Grab RSC ROI and save in h5 file
import time
import h5py
import glob
import numpy as np
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from settings import *

subs = glob.glob(h5path+'sub*.h5')
#subs=['/data/HBN/test2/fmriprep_output/fmriprep/PythonData/sub-NDARAW179AYF_copy.h5']

hemis = ['lh','rh']
HEMIS = ['L','R']
cnum = [0,1] # cluster number associated with RSC for 5 cluster parcel
froi = h5py.File('/data/Schema/intact/Yeo17net.h5', 'r')
f6 = h5py.File('/data/Schema/intact/fsaverage6_adj.h5', 'r')
roi = 16
RSCroi ={}
for hidx,hemi in enumerate(hemis):
    rois = np.nan_to_num(froi[hemi][:][0])
    dispcoords = f6[hemi+'inflatedcoords'][:]
    tt = [np.zeros(coord.shape) if rois[idx]!=roi else coord for idx,coord in enumerate(np.transpose(dispcoords))]
    clustering = AgglomerativeClustering(n_clusters=5).fit(tt)
    RSCroi[hemi] = clustering.labels_==cnum[hidx]
for sub in subs:
    with h5py.File(sub) as f:
        for task in ['DM','TP']:
            del f[task]['RSC']
            data = []
            for hidx,hemi in enumerate(hemis):
                temp = f[task][HEMIS[hidx]][RSCroi[hemi],:]
                data.append(np.delete(temp,list(set([i[0] for i in np.argwhere(np.isnan(temp))])),axis=0))
                #data.append(f[task][HEMIS[hidx]][~np.isnan(f[task][HEMIS[hidx]][RSCroi[hemi],0]),:])
            f[task].create_dataset('RSC', data=np.concatenate(data))


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