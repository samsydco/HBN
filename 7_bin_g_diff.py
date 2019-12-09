#!/usr/bin/env python3

# binned_g_diff figures for ROIs

import os
import glob
import h5py
import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,spearmanr
from sklearn.manifold import MDS
from datetime import date
import numpy as np
import deepdish as dd
from settings import *
from ISC_settings import *
gdiffroidir = path+'ROIs/SfN_2019/ROIs_Fig3/Fig3_'#'ROIs/g_diff/'

iscf = ISCpath + 'ISC_2019-09-06_age_2.h5'#'ISC_2019-08-13_age_2.h5'
ROIl = glob.glob(gdiffroidir+'*roi')

plt.rcParams.update({'font.size': 15})
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import squareform
from itertools import combinations
dfs = {}
X_transformed = {}
mask = np.zeros((nbinseq,nbinseq))
mask[np.triu_indices_from(mask)] = True
for f in ROIl:
	fn = f.split(gdiffroidir)[1]
	roin = fn[:-7]
	vall = []
	X_transformed[roin] = {}
	dfs[roin] = {}
	with open(f, 'r') as inputfile:
		for line in inputfile:
			if len(line.split(' ')) == 3:
				vall.append(int(line.split(' ')[1]))
	n_vox = len(vall)
	for task in ['DM','TP']:
		gdict = {'Age1':[],'Age2':[],'g_diff':[]}
		for p in combinations(range(nbinseq),2):
			ISCg = np.zeros(n_vox) 
			for htmp1 in [0,1]:
				for htmp2 in [0,1]:
					ISCg += dd.io.load(iscf,'/shuff_'+str(0)+'/'+task+'/'+
								   'bin_'+str(p[0])+'_'+str(p[1])+'/'+str(htmp1)+'_'+str(htmp2))[vall]
			ISCg = ISCg/4/(np.sqrt(dd.io.load(iscf,'/shuff_'+str(0)+'/'+task+'/'+
								   'bin_'+str(p[0])+'/'+'ISC_w')[vall])\
				  *np.sqrt(dd.io.load(iscf,'/shuff_'+str(0)+'/'+task+'/'+
								   'bin_'+str(p[0])+'/'+'ISC_w')[vall]))
			ISCg = np.nanmean([i for i in ISCg if i<1])
			for k in gdict.keys():
				ir = [0,1] if '1' in k else [1,0]
				if 'Age' in k:
					for i in ir:
						gdict[k].append(str(int(round(eqbins[p[i]])))+\
								  ' - '+str(int(round(eqbins[p[i]+1])))+' y.o.')
			gdict['g_diff'].extend([ISCg,ISCg])
		df = pd.DataFrame(data=gdict).pivot("Age1", "Age2", "g_diff")
		cols = df.columns.tolist()
		df = df[cols[-2:]+cols[:-2]]
		df = df.reindex(cols[-2:]+cols[:-2])
		dfs[roin][task] = df
		X_transformed[roin][task] = {}
		for ndims in [1,2]:
			embedding = MDS(n_components=ndims, dissimilarity='precomputed')
			X_transformed[roin][task][str(ndims)] = embedding.fit_transform((1-np.nan_to_num(df.values)))
for task in ['DM']:#dfs[roin].keys():
	allvals = [dfs[r][task].values for r in dfs.keys()]
	for roin in dfs.keys():
		print(task,roin)
		df = dfs[roin][task]
		with sns.axes_style("white"):
			ax = sns.heatmap(df, mask=mask, square=True,cbar_kws={'label': 'g diff ISC'},cmap='viridis')#,vmin=0.7,vmax=0.9)
		ax.set_xlabel(None)
		ax.set_ylabel(None)
		plt.xticks(rotation=30,ha="right")
		plt.yticks(rotation=30)
		plt.tight_layout()
		plt.show()
		ax.figure.savefig(figurepath+'SfN_2019/Figure_3_thresh/'+task+'_'+roin+'.png')

# display mds
xlab = df.columns.tolist()
plt.rcParams.update({'font.size': 8})
for task in ['DM','TP']:
	fig, ax = plt.subplots(nrows=5, ncols=2, figsize = (4,10))
	for i,roiv in enumerate(X_transformed.items()):
		for ndims,data in roiv[1][task].items():
			axid = int(ndims)-1
			xvals = data if ndims == '1' else data[:,0]
			yvals = np.zeros(len(xlab)) if ndims == '1' else data[:,1]
			ax[i,axid].scatter(xvals,yvals)
			ax[i,axid].set_xlim(-.2,.2)
			ax[i,axid].set_ylim(-.2,.2)
			#if ndims == 1: ax[i,axid].set_ylim(-.5,.5)
			ax[i,axid].set_aspect(1.0)
			for ii, txt in enumerate(xlab):
				sub = 0.05 if ndims =='2' else 0
				rot = 0 if ndims =='2' else 30
				ax[i,axid].annotate(txt, xy=(xvals[ii],yvals[ii]+sub),ha='center',rotation=rot)
			ax[i,axid].set_title('ROI:'+roiv[0])
	plt.tight_layout()
	plt.show()
	fig.savefig(figurepath+'agediff_g_diff/'+task+'_MDS.png')
		
	
