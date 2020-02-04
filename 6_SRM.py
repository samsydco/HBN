#!/usr/bin/env python3

import os
import tqdm
import glob
import numpy as np
import deepdish as dd
import brainiak.funcalign.srm
from scipy.stats import zscore
import matplotlib.pyplot as plt
from HMM_settings import *

agediffroidir = path+'ROIs/'
ROIopts = ['agediff/*v2','SfN_2019/Fig2_','YeoROIsforSRM_2020-01-03.h5','YeoROIsforSRM_sel_2020-01-14.h5']
agedifffs = agediffroidir + ROIopts[-1]
SRMf = ISCpath+'SRM/'
SRMf = SRMf+'v2/' if ROIopts[0] in agedifffs else SRMf+'SfN_Fig2/' if ROIopts[1] in agedifffs else SRMf+'Yeo/'
if not os.path.exists(SRMf):
    os.makedirs(SRMf)
ROIl = agedifffs if any(r in agedifffs for r in ROIopts[-2:]) else agedifffs

n_iter = 20
train_task = 'DM'
test_task = 'TP'
nTRtrain = 750
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

fits = makeROIdict(ROIl)

for f in tqdm.tqdm(fits):
	SRMff = SRMf+f+'.h5'
	if not os.path.exists(SRMff):
		fitsSRM = {}
		vall = fits[f]['vall']
		nvox = fits[f]['nvox']
		hemi = fits[f]['hemi']
		fitsSRM['vall'] = vall
		fitsSRM['nvox'] = nvox
		fitsSRM['hemi'] = hemi
		fitsSRM['klist'] = np.unique(np.round(np.logspace(np.log(2),											   np.log(np.min([nTRtrain,nvox])),base=np.e))).astype('int')
		for b in [0,nbinseq-1]:
			fitsSRM[str(b)] = {}
			subl = []
			for i in [0,1]:
				subg = [ageeq[i][1][b][idx] for idx in np.random.choice(lenageeq[i][b],divmod(minageeq[i],2)[0]*2,replace=False)]
				subl.extend(subg)
			nsub = len(subl)
			train_data = load_data(subl,train_task,hemi,vall)
			test_data  = load_data(subl,test_task, hemi,vall)
			for k in fitsSRM['klist']:
				fitsSRM[str(b)][str(k)] = {}
				for losub in range(nsub):
					fitsSRM[str(b)][str(k)][str(losub)] = {}
					srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=k)
					srm.fit(loo(train_data,losub))
					subw = srm.transform_subject(train_data[losub]) # Need latest brainiak version for this
					# Now test on TP
					sub_pred = np.dot(subw,sum(srm.transform(loo(test_data,losub)))/nsub)
					fitsSRM[str(b)][str(k)][str(losub)]['frobnorm'] = np.sqrt(np.sum((test_data[losub] - sub_pred)**2))
					fitsSRM[str(b)][str(k)][str(losub)]['tcorr'] = np.mean(np.sum(np.multiply(test_data[losub],sub_pred),axis=1)/(nTRtest-1))
					fitsSRM[str(b)][str(k)][str(losub)]['scorr'] = np.mean(np.sum(np.multiply(test_data[losub],sub_pred),axis=0)/(nvox-1))
		dd.io.save(SRMff,fitsSRM)

fits = {}
for f in glob.glob(SRMf+'*.h5'):
	fits[f.split(SRMf)[-1][:-3]] = dd.io.load(f)

def sortlist(l,order):
    newlist = [l[i] for i in order]
    return newlist
    
labels = ['youngest','oldest']
yaxis = ['dist','r','r']
for roi,r in fits.items():
	fig, ax = plt.subplots(nrows=3, sharex=True)
	fig.suptitle(roi.replace('_',' ')+' '+str(r['nvox'])+' vox', fontsize="x-large")
	plt.style.use('seaborn-muted')
	for i,fit in enumerate(['frobnorm','tcorr','scorr']):
		for bi,b in enumerate(bins):
			x = []
			y = []
			error = []
			for k,subs in r[str(b)].items():
				x.append(int(k))
				y.append(np.mean([subs[sub][fit] for sub in subs.keys()]))
				error.append(np.std([subs[sub][fit] for sub in subs.keys()]))
				csort=np.argsort(x) # sort x,y,and error to eliminate funny lines in plots!
			ax[i].errorbar(sortlist(x,csort), sortlist(y,csort), yerr=sortlist(error,csort), fmt='-o',label=labels[bi])
		ax[i].set_title(fit)
		ax[i].set_ylabel(yaxis[i])
		ax[i].set_xlim(np.min(x),np.max(x))
	plt.legend()
	plt.tight_layout()
	plt.xlabel("k dimensions")
	#fig1 = plt.gcf()
	#plt.show()
	#plt.draw()
	plt.savefig(figurepath+'SRM/Yeo/'+roi+'.png')
		


	
	