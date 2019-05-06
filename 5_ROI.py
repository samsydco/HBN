#!/usr/bin/env python3

# Grab RSC ROI and save in h5 file
import os
import h5py
import glob
import numpy as np
import deepdish as dd
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from settings import *

subs = glob.glob(prepath+'sub*.h5')
#subs=['/data/HBN/test2/fmriprep_output/fmriprep/PythonData/sub-NDARAW179AYF_copy.h5']

hemis = ['lh','rh']
if not os.path.isfile(ISCpath+'RSC.h5'):
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
else:
	RSCroi = dd.io.load(ISCpath+'RSC.h5')

ISCclust = dd.io.load(ISCpath+'ISCclusters_5.h5')

ISC_SUMA_clust = {}
for hemi in ['left','right']:
	clusts = glob.glob(path+'ROIs/'+hemi+'*.roi')
	hem = 'lh' if hemi == 'left' else 'rh'
	ISC_SUMA_clust[hem] = {}
	for clust in clusts:
		part = clust.split('_')[1]
		ISC_SUMA_clust[hem][part] = []
		with open(clust, 'r') as inputfile:
			for line in inputfile:
				if len(line.split(' ')) == 3:
					ISC_SUMA_clust[hem][part].append(int(line.split(' ')[1]))

f = h5py.File(ISCpath+'A1.h5','r')
A1roi={}
ROI = np.nan_to_num(f['A1corr'][:])>0.26
A1roi['L'] = ROI[:len(ROI)//2]
A1roi['R'] = ROI[len(ROI)//2:]

for sub in tqdm(subs):
	print(sub)
	with h5py.File(sub) as f:
		if set(['DM','TP']).issubset(f.keys()):
			for task in ['DM','TP']:
				if 'A1' in list(f[task].keys()): del f[task]['A1']
				if 'RSC' in list(f[task].keys()): del f[task]['RSC']
				a1data = []
				rscdata = []
				for hemi in hemis:
					hem = 'L' if hemi == 'lh' else 'R'
					temp = f[task][hem][RSCroi[hemi],:]
					rscdata.append(np.delete(temp,np.unique([i[0] for i in np.argwhere(np.isnan(temp))]),axis=0))
					a1data.append(f[task][hem][A1roi[hem],:])
					if 'ISC_'+hem in list(f[task].keys()): del f[task]['ISC_'+hem]
					grp = f[task].create_group('ISC_'+hem)
					isctask = 'TP' if task == 'DM' else 'DM'
					for i,r in enumerate(ISCclust['clusters'][isctask][hemi]):
						grp.create_dataset(str(i), data=f[task][hem][[i for i,x in enumerate(r) if x==1],:])
					if 'ISC_SUMA_'+hem in list(f[task].keys()): del f[task]['ISC_SUMA_'+hem]
					grp = f[task].create_group('ISC_SUMA_'+hem)
					for key, values in ISC_SUMA_clust[hemi].items():
						grp.create_dataset(key, data=f[task][hem][values,:])
				f[task].create_dataset('RSC', data=np.concatenate(rscdata))
				f[task].create_dataset('A1', data=np.concatenate(a1data))
		else:
			os.remove(sub)

# save subject x node x time array for each ROI in h5 file
ROIf = 'ROIstack_'
for i in glob.glob(ISCpath+ROIf+'*'):
	if os.path.exists(i):
		os.remove(i)

for task in tqdm(['DM','TP']):
	ff = dd.io.load(subs[0], ['/'+task])[0]
	# create list of ROIs
	ROIs = []
	for k in list(ff.keys()):
		if type(ff[k]) == np.ndarray:
			if ff[k].shape[0] < 20000:
				ROIs.append(k)
		if type(ff[k]) == dict:
			for kk in list(ff[k].keys()):
				if ff[k][kk].shape[0] < 20000:
					ROIs.append(k+'.'+kk)
	del ff
	ROIs = [r for r in ROIs if not any(i == r for i in ['reg'])]
	for roi in ROIs:
		print(roi)
		if '.' in roi:
			roi_ = roi.split('.')
			sh = dd.io.load(subs[0],['/'+task+'/'+roi_[0]+'/'+roi_[1]])[0].shape
			ROIdata = np.empty((len(subs),sh[0],sh[1]))
			for s, sub in enumerate(subs):
				ROIdata[s,:,:] = dd.io.load(sub,['/'+task+'/'+roi_[0]+'/'+roi_[1]])[0]
		else:
			sh = dd.io.load(subs[0],['/'+task+'/'+roi_[0]+'/'+roi_[1]])[0].shape
			ROIdata = np.empty((len(subs),sh[0],sh[1]))
			for s, sub in enumerate(subs):
				ROIdata[s,:,:] = dd.io.load(sub,['/'+task+'/'+roi])[0]
		if np.count_nonzero(np.isnan(ROIdata)) == 0:
			with h5py.File(ISCpath+ROIf+task+'_'+roi+'.h5') as hf:
				hf.create_dataset(roi,data=ROIdata)
				string_dt = h5py.special_dtype(vlen=str)
			#hf.create_dataset("subs",data=np.array(subs).astype('|S71'),dtype=string_dt)