#!/usr/bin/env python3

from settings import *
#subord = subord[0:5] # for testing!
from datetime import date,datetime
from scipy.stats import binned_statistic
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
# Only deploy if pc's seem interesting - trying to control for more than one pheno:
'''
demo2idx = [[[[], []] for i in range(2)] for i in range(2)]
for tf1 in [0,1]:
	for h in [0,1]:
		idx = subh[tf1][h]
		for tf2 in [0,1]:
			demo2idx[tf1][tf2][h].extend([idx for idx,p in enumerate(demo2) if idx in subh[tf1][h] and p==tf2])
			print(tf1,tf2,h,len(demo2idx[tf1][tf2][h]))
'''

def split_subj(phenol):
	splitidx = {}
	subdist = {}
	subsum = {}
	for k,v in phenol.items():
		splitidx[k] = [[None]*2 for i in range(2)]
		subdist[k] = dict(zip(list(phenol.keys()), [[[None]*2 for i in range(2)] for ii in range(len(phenol))]))
		subsum[k] = dict(zip(list(phenol.keys()), [np.zeros((2,2))]*len(phenol)))
		for h in [0,1]: # split all or between T / F
			if k == 'all':
				subh = np.arange(0+len(v)//2*h,len(v)//2+len(v)//2*h)
			else:
				subh = [idx for idx,i in enumerate(v) if i == h]
			for htmp in [0,1]:
				splitidx[k][h][htmp] = subh[0+len(subh)//2*htmp:len(subh)//2+len(subh)//2*htmp]
				for kt,vt in phenol.items():
					subdist[k][kt][h][htmp] = [p for idx,p in enumerate(vt) if idx in splitidx[k][h][htmp]]
					subsum[k][kt][h,htmp] = np.nansum(subdist[k][kt][h][htmp])
	return splitidx,subdist,subsum

# max diff between old/young in hist bins of equal height:
nsub = 15 # supposed number of subjects in each bin
agespan = np.max(np.diff(np.interp(np.linspace(0, len(agel), len(agel)//nsub + 1),np.arange(len(agel)),np.sort(agel))))
nbinseq = ((max(agel)-min(agel))//agespan).astype('int')
eqbins = []
for b in range(nbinseq+1):
	eqbins.append(min(agel)+agespan*b)

plot = 'off'
nbins = 19
bins = np.linspace(min(agel), max(agel), nbins+1)
agedist = [None]*2
# make equal width bins, w same # of ages, and consistent sex dist
ageeq = [[[[] for _ in range(nbinseq)] for _ in range(2)] for _ in range(2)]
lenageeq = [[] for _ in range(2)]
minageeq = []
# Are M and F evenly dist in age
for i in np.unique(phenol['sex']):
	ages = [a for idx,a in enumerate(agel) if phenol['sex'][idx]==i]
	agedist[i] = [binned_statistic(ages,ages,statistic='count',bins=bins),[ [] for i in range(nbins) ]]
	for b in range(nbinseq):
		ageeq[i][0][b] = [idx for idx,a in enumerate(ages) 
					   if a>=eqbins[b] and a<eqbins[b+1]]
		lenageeq[i].append(len(ageeq[i][0][b]))
	minageeq.append(min(lenageeq[i]))
	for idx,sub in enumerate([s for idx,s in enumerate(subord) if phenol['sex'][idx]==i]):
		agedist[i][1][agedist[i][0][2][idx]-1].append(sub)
		if ages[idx] < eqbins[nbinseq]:
			ageeq[i][1][[b for b in range(nbinseq) if idx in ageeq[i][0][b]][0]].append(sub)
		
nsubbin = [min([agedist[1][0][0][b],agedist[0][0][0][b]]).astype(int) for b in range(nbins)]

def binagesubs(agel,phenol,eqbins,subord):
	nbinseq = len(eqbins) - 1
	ageeq = [[[[] for _ in range(nbinseq)] for _ in range(2)] for _ in range(2)]
	lenageeq = [[] for _ in range(2)]
	minageeq = []
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
	return ageeq,lenageeq,minageeq
	

if plot != 'off':
	import matplotlib.pyplot as plt
	#%matplotlib inline
	# Are M and F evenly dist in age
	for i in np.unique(phenol['sex']):
		ages = [a for idx,a in enumerate(agel) if phenol['sex'][idx]==i]
		plt.hist(ages, bins, alpha=0.5, label=str(i))
	plt.axvline(np.median(agel), color='k', linestyle='dashed', linewidth=1)
	plt.legend(loc='upper right')
	plt.title('Sex Dist by Age')
	plt.xlabel('Ages')
	plt.ylabel('Frequency')
	plt.gcf().savefig(figurepath+'AgeSexDist.png')
	plt.show()

#dd.io.save(metaphenopath+'pheno_'+str(date.today())+'.h5',{'subs':subord,'ages':agel,'phenodict':phenol,'pcs':pcl})