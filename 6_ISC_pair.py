#!/usr/bin/env python3

# Calculate pairwise ISC
# See if older subjects cluster more than younger subjects

import glob
import tqdm
import scipy.cluster.hierarchy as spc
from HMM_settings import *
agediffroidir = path+'ROIs/SfN_2019/Fig2_'
ISCdir = ISCpath + 'Pairwise_ISC/'

for f in glob.glob(agediffroidir+'*roi'):
	print(f)
	fbits = f.split(agediffroidir)[1].split('_')
	hemi = fbits[1][0]
	task = fbits[0]
	nTR_ = nTR[tasks.index(task)]
	vs = []
	with open(f, 'r') as inputfile:
		for line in inputfile:
			if len(line.split(' ')) == 3:
				vs.append(int(line.split(' ')[1]))
	print(len(vs))
	for b in bins:
		subl = [ageeq[i][1][b][idx] for i in [0,1] for idx in np.random.choice(lenageeq[i][b],minageeq[i],replace=False)]
		nsub = len(subl)
		D = np.empty((nsub,nTR_),dtype='float16')
		for sidx, sub in enumerate(subl):
			D[sidx] = np.mean(dd.io.load(sub,['/'+task+'/'+hemi], sel=dd.aslice[vs,:])[0],0)
		ISC = np.corrcoef(D)
		clust = spc.linkage(ISC, method='complete')
		
