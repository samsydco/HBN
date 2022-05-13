#!/usr/bin/env python3

# check for TPJ / parcel overlap
# must run in conda env: new_nilearn
# needs nilearn version: 0.8.1

import glob
import nilearn
import numpy as np
import deepdish as dd
from nilearn import datasets
from nilearn import surface
from settings import *

fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage6')

TPJs = glob.glob('*TPJ_thresholded/*img')

TPJdict = {k:{} for k in ['L','R']}
for tpj in TPJs:
	hemi = tpj[0]
	surf = 'pial_left' if hemi=='L' else 'pial_right'
	img = nilearn.image.load_img(tpj)
	texture = np.round(surface.vol_to_surf(img, fsaverage[surf]))
	TPJdict[hemi]['TPJ'] = np.where(texture==1)[0]
	TPJdict[hemi]['rois'] = []
	for roi in glob.glob(ISCpath+'Yeo_parcellation_0/'+hemi+'*h5'):
		vall = dd.io.load(roi,'/vall')
		overlap = np.intersect1d(TPJdict[hemi]['TPJ'],vall)
		if len(overlap)>0:
			TPJdict[hemi]['rois'].append(roi.split('/')[-1][:-3])

dd.io.save(ISCpath+'TPJ.h5',TPJdict)
	
			
