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
            data = []
            for hidx,hemi in enumerate(hemis):
                if 'RSC_'+HEMIS[hidx] in list(f[task].keys()):
                    del f[task]['RSC_'+HEMIS[hidx]]
                # RERUN with ddof = 1 !
                data.append(f[task][HEMIS[hidx]][RSCroi[hemi],:])
            f[task].create_dataset('RSC', data=stats.zscore(np.mean(np.concatenate(data),axis=0),ddof=1))
                    