#!/usr/bin/env python3

# binned_g_diff figures for ROIs

import os
import glob
import h5py
import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,spearmanr
from datetime import date
import numpy as np
import deepdish as dd
from settings import *
from ISC_settings import *
agediffroidir = path+'ROIs/agediff/'
gdiffroidir = path+'ROIs/g_diff/'

# what voxels change in ISC with age, and in which direction?
iscf = ISCpath + 'ISC_2019-09-06_age_2.h5'#'ISC_2019-08-13_age_2.h5'
agediff_f = ISCpath + 'ISC_' + str(date.today())+'_agediff'
agediff_f = agediff_f+'_2' if 'age_2' in iscf and not smooth else agediff_f+'_smooth' if smooth else agediff_f
agediff_f = agediff_f+'.h5'
if os.path.exists(agediff_f):
    os.remove(agediff_f)
	
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import squareform
from itertools import combinations
ISCmat = {}
mask = np.zeros((nbinseq,nbinseq))
mask[np.triu_indices_from(mask)] = True
for f in glob.glob(gdiffroidir+'*roi'):
	fn = f.split(gdiffroidir)[1]
	roin = fn[:-7]
	ISCmat[roin] = {}
	task = 'DM' if fn[:2] == 'TP' else 'TP'
	hemi = fn[3]
	vall = []
	with open(f, 'r') as inputfile:
		for line in inputfile:
			if len(line.split(' ')) == 3:
				vall.append(int(line.split(' ')[1]))
	n_vox = len(vall)
	for task in ['DM','TP']:
		gdict = {'Age1':[],'Age2':[],'g_diff':[]}
		ISCmat[roin][task] = []
		for p in combinations(range(nbinseq),2):
			if p[0]==p[1]:
				ISCg = 0
			else:
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
			ISCmat[roin][task].append(ISCg)
		df = pd.DataFrame(data=gdict).pivot("Age1", "Age2", "g_diff")
		cols = dfp.columns.tolist()
		df = df[cols[-2:]+cols[:-2]]
		df = df.reindex(cols[-2:]+cols[:-2])
		with sns.axes_style("white"):
			ax = sns.heatmap(df, mask=mask, square=True,cbar_kws={'label': 'g diff ISC'})
		ax.set_xlabel(None)
		ax.set_ylabel(None)
		plt.xticks(rotation=30,ha="right")
		plt.yticks(rotation=30)
		plt.tight_layout()
		plt.show()
		ax.figure.savefig(figurepath+'agediff_g_diff/'+roin+'_'+task+'.png')
		

		
	
