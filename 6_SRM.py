#!/usr/bin/env python3

import os
import tqdm
import glob
import numpy as np
import deepdish as dd
from datetime import date
import brainiak.funcalign.srm
from scipy.stats import zscore
import matplotlib.pyplot as plt
from settings import *
from ISC_settings import *

agediffroidir = path+'ROIs/'
ROIopts = ['agediff/*v2','SfN_2019/Fig2_']
agedifffs = agediffroidir + ROIopts[0]
SRMf = ISCpath+'SRM_'+str(date.today())
SRMf = SRMf+'_v2.h5' if ROIopts[0] in agedifffs else SRMf+'.h5'
if os.path.exists(SRMf):
	os.remove(SRMf)

agedifffs = agediffroidir+'agediff/*v2'#'SfN_2019/Fig2_'#'ROIs/agediff/'
agedifffs2 = agediffroidir+'SfN_2019/Fig2_'
klist = np.arange(2,100)
n_iter = 20
train_task = 'DM'
test_task = 'TP'
nTRtest = 250


def load_data(subl,task,hemi,vall):
	data = []
	for i,sub in enumerate(subl):
		data.append(dd.io.load(sub,['/'+task+'/'+hemi],sel=dd.aslice[vall,:])[0])
		data[i] = zscore(data[i], axis=1, ddof=1)
		data[i] = np.nan_to_num(data[i])
	return data
def loo(data,losub):
	data = data[:losub] + data[losub + 1:]
	return data

fits = {}
for f in tqdm.tqdm(glob.glob(agedifffs+'*roi')):#glob.glob(agediffroidir+'*v2*'):
	print(f)
	fbits = f.split(agediffroidir)[1].split('/')[1].split('_')
	vall = []
	with open(f, 'r') as inputfile:
		for line in inputfile:
			if len(line.split(' ')) == 3:
				vall.append(int(line.split(' ')[1]))
	nvox = len(vall)
	hemi = 'L' if 'SfN' in agedifffs else fbits[3][0].capitalize()
	fits['_'.join(fbits)] = {}
	fits['_'.join(fbits)]['nvox'] = nvox
	for b in [0,nbinseq-1]:
		print('b =',b)
		fits['_'.join(fbits)][str(b)] = {}
		subl = []
		for i in [0,1]:
			subg = [ageeq[i][1][b][idx] for idx in np.random.choice(lenageeq[i][b],divmod(minageeq[i],2)[0]*2,replace=False)]
			subl.extend(subg)
		nsub = len(subl)
		train_data = load_data(subl,train_task,hemi,vall)
		test_data  = load_data(subl,test_task, hemi,vall)
		for k in [k for k in klist if k<nvox]:
			print('k =',k)
			fits['_'.join(fbits)][str(b)][str(k)] = {}
			for losub in range(nsub):
				fits['_'.join(fbits)][str(b)][str(k)][str(losub)] = {}
				srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=k)
				srm.fit(loo(train_data,losub))
				subw = srm.transform_subject(train_data[losub]) # Need latest brainiak version for this
				# Now test on TP
				sub_pred = np.dot(subw,sum(srm.transform(loo(test_data,losub)))/nsub)
				fits['_'.join(fbits)][str(b)][str(k)][str(losub)]['frobnorm'] = np.sqrt(np.sum((test_data[losub] - sub_pred)**2))
				fits['_'.join(fbits)][str(b)][str(k)][str(losub)]['tcorr'] = np.mean(np.sum(np.multiply(test_data[losub],sub_pred),axis=1)/(nTRtest-1))
				fits['_'.join(fbits)][str(b)][str(k)][str(losub)]['scorr'] = np.mean(np.sum(np.multiply(test_data[losub],sub_pred),axis=0)/(nvox-1))
				
dd.io.save(SRMf,fits)
fits = dd.io.load(SRMf)

labels = ['youngest','oldest']
yaxis = ['dist','r','r']
for roi,r in fits.items():
	fig, ax = plt.subplots(nrows=3, sharex=True)
	fig.suptitle(roi.replace('_',' ')+' '+str(r['nvox'])+' vox', fontsize="x-large")
	plt.style.use('seaborn-muted')
	for i,fit in enumerate(['frobnorm','tcorr','scorr']):
		for bi,b in enumerate(list(r.keys())[:-1]):
			x = []
			y = []
			error = []
			for k,subs in r[b].items():
				x.append(int(k))
				y.append(np.mean([subs[sub][fit] for sub in subs.keys()]))
				error.append(np.std([subs[sub][fit] for sub in subs.keys()]))
			ax[i].errorbar(x, y, yerr=error, fmt='-o',label=labels[bi])
		ax[i].set_title(fit)
		ax[i].set_ylabel(yaxis[i])
		ax[i].set_xlim(np.min(x),np.max(x))
	plt.legend()
	plt.tight_layout()
	plt.xlabel("k dimensions")
	#fig1 = plt.gcf()
	#plt.show()
	#plt.draw()
	plt.savefig(figurepath+'SRM/'+roi+'.png')
		


				
				
				
                
				
		
	
	
	