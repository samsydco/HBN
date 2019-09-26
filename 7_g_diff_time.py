#!/usr/bin/env python3

# Look at g_difference over time in ROIs with big g_difference
# avg temporal ISC in ROI
# avg spatial ISC in ROI

import glob
import numpy as np

import deepdish as dd
from datetime import date,datetime
from settings import *
from ISC_settings import *

ROIfold = path+'ROIs/g_diff/'
g_diff_f = ISCpath+'g_diff_time_'+str(date.today())+'.h5'
HMMfdate = str(min([datetime.strptime(i.split('/')[-1].split('.h5')[0].split('HMM_')[1],'%Y-%m-%d') for i in glob.glob(HMMpath+'*') if '2019' in i], \
	key=lambda x: abs(x - datetime.today()))).split(' ')[0]
HMMf = HMMpath+'HMM_'+HMMfdate+'.h5'
rs = dd.io.load(HMMf)


for f in glob.glob(ROIfold+'*roi'):
	fn = f.split(ROIfold)[1]
	roin = fn[:-7]
	vall = dd.io.load(HMMf,'/'+roin+'/vall')
	n_vox = len(vall)
	task = 'DM' if fn[:2] == 'TP' else 'TP'
	hemi = fn[3]
	for b in [0,nbinseq-1]:
		subl = [ageeq[i][1][b][idx] for i in [0,1] for idx in np.random.choice(lenageeq[i][b],minageeq[i],replace=False)]
		nsub = len(subl)
		# Load data
		_,n_time = dd.io.load(subl[0],['/'+task+'/'+hemi])[0].shape
		D = np.empty((nsub,n_vox,n_time),dtype='float16')
		for sidx, sub in enumerate(subl):
			D[sidx,:,:] = dd.io.load(sub,['/'+task+'/'+hemi],sel=dd.aslice[vall,:])[0]
			
		avg_temp_isc = 
		avg_spac_isc
		
	

