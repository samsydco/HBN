#!/usr/bin/env python3

import h5py
import deepdish as dd
import os
import glob
import numpy as np
from scipy.spatial.distance import squareform
from settings import *

from brainiak import isfc
EventSegf = 'EventSeg.h5'
if os.path.exists(ISCpath+EventSegf):
	os.remove(ISCpath+EventSegf)

subord = glob.glob(prepath+'sub*.h5')
subord = subord[0:5] # for testing!
ff = dd.io.load(subord[0])
for task in ['DM','TP']:
	ROIs = []
	for k in list(ff[task].keys()):
		if type(ff[task][k]) == np.ndarray:
			ROIs.append(k)
		if type(ff[task][k]) == dict:
			for kk in list(ff[task][k].keys()):
				ROIs.append(k+'_'+kk)
	ROIdata = {}
	for roi in ROIs:
		ROIdata[roi] = []
	for subidx, sub in enumerate(subord):
		ff = dd.io.load(sub)
		for k in list(ff[task].keys()):
			if type(ff[task][k]) == np.ndarray:
				ROIdata[k].append(ff[task][k][:])
			if type(ff[task][k]) == dict:
				for kk in list(ff[task][k].keys()):
					ROIdata[k+'_'+kk].append(ff[task][k][kk][:])
	for roi in ROIs:
		ROIdata[roi] = np.stack(ROIdata[roi])
	with h5py.File(ISCpath+EventSegf) as hf:
		grp = hf.create_group(task)
		for roi in ROIs:
			grp.create_dataset(roi,data=ROIdata[roi])

