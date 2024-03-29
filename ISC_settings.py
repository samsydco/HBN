#!/usr/bin/env python3

from settings import *
from scipy.stats import binned_statistic
from scipy.stats import zscore
import glob
import pandas as pd
import numpy as np
import deepdish as dd

Phenodf = pd.concat((pd.read_csv(f) for f in glob.glob(phenopath+'HBN_R*Pheno.csv')),ignore_index=True)
datadf, pc = dd.io.load(metaphenopath+'data.h5',['/data','/pc'])

def shortsub(sub):
	return sub.split('sub-')[1].split('.h5')[0]

# make a list of all subjects age/sex, do median split, put in dict
def make_phenol(subl):
	agel = []
	sexidx = []
	for sub in subl:
		subbool = Phenodf['EID'] == shortsub(sub)
		agel.append(Phenodf['Age'][subbool].iloc[0])
		sexidx.append(Phenodf['Sex'][subbool].iloc[0])
	phenol = {'all':[True]*len(subl),
		  'age':agel > np.median(agel),
		  'sex':[s == 1 for s in sexidx]} #True is Female
	# true if sub in subl, False otherwise
	pcidx = datadf['sub'].isin(['sub-'+shortsub(s) for s in subl]).tolist()
	pcl = [None]*4
	for i in range(4):
		pcl[i] = [pc.factors[datadf['sub'].isin(['sub-'+shortsub(sub)])[datadf['sub'].isin(['sub-'+shortsub(sub)])==True].index.tolist()[0],i] if datadf['sub'].str.contains('sub-'+shortsub(sub)).any() else np.nan for sub in subl]
		phenol['pc'+str(i)] =  [p>np.median(pc.factors[pcidx,i]) if not np.isnan(p) else p for p in pcl[i]]
	return agel, pcl, phenol

agel,pcl,phenol = make_phenol(subord)

def even_out(demo1,demo2):
	# demo1 T/F status of group to be subdivided into 2 T and 2 F groups
	# demo2 T/F status of group to be equally divided into 4 groups
	demo1idx = []
	demo2idx = [[[], []] for i in range(2)]
	d2sum = np.zeros((2,2))
	for tf1 in [0,1]:
		demo1idx.append([idx for idx,p in enumerate(demo1) if p==tf1])
		for tf2 in [0,1]:
			demo2idx[tf1][tf2].extend([idx for idx,p in enumerate(demo2) if idx in demo1idx[tf1] and p==tf2])
			d2sum[tf2,tf1] = len(demo2idx[tf1][tf2])
	subh = [[[], []] for i in range(2)]
	for tf2 in [0,1]:
		if len(demo1idx[0])!=0:
			mins = (min(d2sum[tf2])).astype(int)
			for tf1 in [0,1]:
				subidx = np.random.choice(demo2idx[tf1][tf2],mins,replace=False)
				for h in [0,1]:
					# subject indices for demo2=tf2, demo1=tf1,split=s
					subh[tf1][h].extend(subidx[0+mins//2*h:mins//2+mins//2*h])
		else:
			mins = (min(d2sum[:,1])).astype(int)
			subidx = np.random.choice(demo2idx[1][tf2],mins,replace=False)
			for tf1 in [0,1]:
				for h in [0,1]:
					subh[tf1][h].extend(subidx[0+mins//4*h+(mins//2)*tf1:mins//4+mins//4*h+(mins//2)*tf1])
	return subh
# Check for equivelent demo2 distribution in all 4 subh groups:
'''
d2sum = np.zeros((2,2,2))
for tf1 in [0,1]:
	for h in [0,1]:
		for tf2 in [0,1]:
			d2sum[tf2,tf1,h] = len([idx for idx,p in enumerate(demo2) if idx in subh[tf1][h] and p==tf2])
'''


def bin_split(subord):
	agel,pcl,phenol = make_phenol(subord)
	nsub = 15 # supposed number of subjects in each bin
	agespan = np.max(np.diff(np.interp(np.linspace(0, len(agel), len(agel)//nsub + 1),np.arange(len(agel)),np.sort(agel))))
	nbinseq = ((max(agel)-min(agel))//agespan).astype('int')
	eqbins = []
	for b in range(nbinseq+1):
		eqbins.append(min(agel)+agespan*b)

	# make equal width bins, w same # of ages, and consistent sex dist
	ageeq = [[[[] for _ in range(nbinseq)] for _ in range(2)] for _ in range(2)]
	lenageeq = [[] for _ in range(2)]
	minageeq = []
	# Are M and F evenly dist in age
	for i in np.unique(phenol['sex']):
		ages = [a for idx,a in enumerate(agel) if phenol['sex'][idx]==i]
		for b in range(nbinseq):
			ageeq[i][0][b] = [idx for idx,a in enumerate(ages) 
						   if a>=eqbins[b] and a<eqbins[b+1]]
			lenageeq[i].append(len(ageeq[i][0][b]))
		minageeq.append(min(lenageeq[i]))
		for idx,sub in enumerate([s for idx,s in enumerate(subord) if phenol['sex'][idx]==i]):
			if ages[idx] < eqbins[nbinseq]:
				ageeq[i][1][[b for b in range(nbinseq) if idx in ageeq[i][0][b]][0]].append(sub)
	return agespan,nbinseq,eqbins,ageeq,lenageeq,minageeq

agespan,nbinseq,eqbins,ageeq,lenageeq,minageeq = bin_split(subord)

nshuff = 100
	
def p_calc(ISC,ISCtype='e'):
	nshuff = ISC.shape[0]-1
	if ISCtype == 'e':
		p = np.sum(abs(np.nanmean(ISC[0]))<abs(np.nanmean(ISC[1:],axis=1)))/nshuff
	else:
		p = np.sum(np.nanmean(ISC[0])>np.nanmean(ISC[1:],axis=1))/nshuff
	return p,nshuff

def load_D(roi,task,bins):
	D = []
	Age = []
	Sex = []
	for bi,b in enumerate(bins):
		bstr = 'bin_'+str(b)
		subl = dd.io.load(roi,'/'+'/'.join([task,bstr,'subl']))
		Sex.extend([Phenodf['Sex'][Phenodf['EID'] == shortsub(sub)].iloc[0] for sub in subl])
		Age.extend([bi]*len(subl))
		D.append(dd.io.load(roi,'/'+'/'.join([task,bstr,'D'])))
	D = np.concatenate(D)
	return D,Age,Sex

def shuff_demo(shuff,Age,Sex):
	np.random.seed(shuff) # same random order on same shuffs
	# Now shuffle Age, and Sex in same order:
	neword = np.random.permutation(len(Age))
	Age = [Age[neword[ai]] for ai,a in enumerate(Age)]
	Sex = [Sex[neword[ai]] for ai,a in enumerate(Sex)]
	return Age,Sex
	
def ISC_w_calc(D,n_vox,n_time,nsub,subh):
	nbins = len(subh)
	ISC_w = np.zeros((nbins,n_vox))
	groups = np.zeros((nbins,2,n_vox,n_time),dtype='float16')
	for h in range(nbins):
		for htmp in [0,1]:
			group = np.zeros((n_vox,n_time),dtype='float16')
			groupn = np.ones((n_vox,n_time),dtype='int')*nsub//2
			for i in subh[h][htmp]:
				group = np.nansum(np.stack((group,D[i])),axis=0)
				nanverts = np.argwhere(np.isnan(D[i,:]))
				groupn[nanverts[:, 0],nanverts[:,1]] = groupn[nanverts[:,0],nanverts[:,1]]-1
			groups[h,htmp] = zscore(group/groupn,axis=1)
		ISC_w[h] = np.sum(np.multiply(groups[h,0],groups[h,1]), axis=1)/(n_time-1)
	return ISC_w,groups