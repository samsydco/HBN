#!/usr/bin/env python3

# make h5 files for each parcel
# data = subjects x verticies x TRs

import os
import tqdm
import nibabel.freesurfer.io as free
from HMM_settings import *

tasks=['DM']
bins = np.arange(nbinseq)

for seed in np.arange(5):
	savepath = roidir+str(seed)+'/'
	if not os.path.exists(savepath):
		os.makedirs(savepath)
		for hemi in glob.glob(path+'ROIs/annot/*'):
			print(seed,hemi)
			lab = free.read_annot(hemi)
			for ri,roi_tmp in tqdm.tqdm(enumerate(lab[2])):
				roi=roi_tmp.decode("utf-8")
				roi_short=roi_tmp.decode("utf-8")[11:]
				roidict = {}
				vall = np.where(lab[0]==ri)[0]
				roidict['hemi'] = (hemi.split('/')[-1][0]).upper()
				for ti,task in enumerate(tasks):
					roidict[task] = {}
					nTR_ = nTR[ti]
					for b in bins:
						if len(vall) > 0:
							roidict[task]['bin_'+str(b)] = {}
							np.random.seed(seed)
							subl = [ageeq[i][1][b][idx] for i in [0,1] for idx in np.random.choice(lenageeq[i][b],minageeq[i],replace=False)]
							roidict[task]['bin_'+str(b)]['subl'] = subl
							nsub = len(subl)
							# Load data
							D = np.empty((nsub,len(vall),nTR_),dtype='float16')
							badvox = []
							for sidx, sub in enumerate(subl):
								D[sidx,:,:] = dd.io.load(sub,['/'+task+'/'+roidict['hemi']],sel=dd.aslice[vall,:])[0]
								badvox.extend(np.where(np.isnan(D[sidx,:,0]))[0]) # Some subjects missing some voxels
							D = np.delete(D,badvox,1)
							vall = np.delete(vall,badvox)
							roidict['vall'] = vall
							roidict[task]['bin_'+str(b)]['D'] = D
				if len(vall) > 0:
					dd.io.save(savepath+roi_short+'.h5',roidict)
			
