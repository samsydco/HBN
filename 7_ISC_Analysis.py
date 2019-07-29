#!/usr/bin/env python3

# Analyze ISC findings:

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

# what voxels change in ISC with age, and in which direction?
iscf = ISCpath + 'ISC_2019-07-01_age.h5'
agediff_f = ISCpath + 'ISC_' + str(date.today())+'_agediff.h5'
if os.path.exists(agediff_f):
    os.remove(agediff_f)

import seaborn as sns
import pandas as pd
xticks = [round(e,2) for e in eqbins[:-1]+agespan/2]
for f in tqdm.tqdm(glob.glob(agediffroidir+'*v2*')):
	print(f)
	fbits = f.split(agediffroidir)[1].split('_')
	task = fbits[0]
	if 'txt' in f:
		vs = np.loadtxt(glob.glob(agediffroidir+'*txt')[0]).astype(int)
	else:
		vs = []
		with open(f, 'r') as inputfile:
			for line in inputfile:
				if len(line.split(' ')) == 3:
					vs.append(int(line.split(' ')[1]))
	
	fdict = {'Age':[],'ISC':[],'vox':[],'corr':[]}
	ISCs = np.zeros((nbinseq,len(vs)))
	vadd = 0 if fbits[3] == 'lh' else 40962
	for idx,v in enumerate(vs):
		for b in range(nbinseq):
			if fbits[2] == 'all':
				for i in list(dd.io.load(iscf)['bin_0'][task].keys()):
					ISCs[b,idx] += dd.io.load(iscf,
					'/bin_'+str(b)+'/'+task+'/'+i)[v+vadd]
				ISCs[b,idx] = ISCs[b,idx]/6
				fdict['Age'].append(xticks[b])
				fdict['ISC'].append(ISCs[b,idx])
				fdict['vox'].append(v)
		fdict['corr'] = fdict['corr'] + [np.corrcoef([e+agespan/2 for e in eqbins[:-1]],ISCs[:,idx])[0,1] for b in range(nbinseq)]
	
	#ax = plt.errorbar(xticks, list(df.groupby('Age')['ISC'].mean()), yerr=list(df.groupby('Age')['ISC'].std()),ls='none', fmt='k+', markersize=25, capsize=5)
	df = pd.DataFrame(data=fdict)
	fig = sns.swarmplot(x="Age", y="ISC", data=df,zorder=1)
	fig = sns.pointplot(x="Age", y="ISC", data=df,markers='+',join=False,color='k',ci='sd',capsize=.1, zorder=100)
	plt.show()
	fig.figure.savefig(figurepath+'agediff/'+'_'.join(fbits)+'.png')
	
	'''
	df = pd.DataFrame(data=fdict)
	g = sns.FacetGrid(df, col="corr", col_wrap=len(vs)//20)
	g = g.map(plt.plot, "Age", "ISC", marker=".")
	g.savefig(figurepath+'agediff_'+'_'.join(fbits)+'.png')
	'''
	'''
	ax = sns.swarmplot(x='Age',y='ISC',hue='vox',data=df,palette="Set1")
	l = ax.legend()
	l.remove()
	'''
	
	#ax = plt.plot(np.matlib.repmat(xticks,len(vs),1).T,ISCs)
	#ax = sns.swarmplot(data=ISCs)

#with h5py.File(agediff_f) as hf:
with h5py.File(agediff_f,'a') as hf:
	for task in ['DM','TP']:
		grp = hf.create_group(task)
		for s in ['all']:#['0','1','all','err_diff','g_diff']:
			print(task,s)
			ISCr = np.zeros(81924)
			ISCs = np.zeros(81924)
			ISCstddivmean = np.zeros(81924)
			ISCdiff = np.zeros(81924) # The interval between which two bins is greatest?
			for v in tqdm.tqdm(range(81924)):
				ISC = np.zeros(nbinseq)
				for b in range(nbinseq):
					if s in ['0','1']:
						ISC[b] = dd.io.load(iscf,
						'/bin_'+str(b)+'/'+task+'/ISC_SH_w_'+s)[v]
					if s == 'err_diff':
						ISC[b] = dd.io.load(iscf,
						'/bin_'+str(b)+'/'+task+'/ISC_SH_w_1')[v] - \
						dd.io.load(iscf,
						'/bin_'+str(b)+'/'+task+'/ISC_SH_w_0')[v]
					if s == 'g_diff':
						for i in ['ISC_SH_b_0_0', 'ISC_SH_b_0_1', 'ISC_SH_b_1_0', 'ISC_SH_b_1_1']:
							ISC[b] += dd.io.load(iscf,
							'/bin_'+str(b)+'/'+task+'/'+i)[v]
						ISC[b] = ISC[b]/4/ \
						(np.sqrt(dd.io.load(iscf,
						'/bin_'+str(b)+'/'+task+'/ISC_SH_w_1')[v])
						*np.sqrt(dd.io.load(iscf,
						'/bin_'+str(b)+'/'+task+'/ISC_SH_w_0')[v]))
					if s == 'all':
						for i in list(dd.io.load(iscf)['bin_0'][task].keys()):
							ISC[b] += dd.io.load(iscf,
							'/bin_'+str(b)+'/'+task+'/'+i)[v]
						ISC[b] = ISC[b]/6
				ISCr[v] = np.corrcoef([e+agespan/2 for e in eqbins[:-1]],ISC)[0,1]
				diffidx = np.argmax(abs(np.diff(ISC)))
				if np.sum(np.isnan(ISC))<4:
					ISCs[v] = spearmanr([e+agespan/2 for e in eqbins[:-1]],ISC)[0]
					ISCdiff[v] = (diffidx+1)*np.sign(np.diff(ISC)[diffidx])
				else:
					ISCs[v] = np.nan
					ISCdiff[v] = np.nan
				ISCstddivmean[v] = np.std(ISC)/np.mean(ISC)
			grp.create_dataset('diff_'+s,data=ISCdiff)
			grp.create_dataset('spearman_'+s,data=ISCs)
			grp.create_dataset('corr_'+s,data=ISCr)
			grp.create_dataset('stddivmean_'+s,data=ISCstddivmean)
			'''
			hf.create_dataset(task+'/spearman_'+s,data=ISCs)
			hf.create_dataset(task+'/corr_'+s,data=ISCr)
			hf.create_dataset(task+'/stddivmean_'+s,data=ISCstddivmean)
			'''
					
			
					




# 1) Are vertices with high ISC in both groups (either age or sex) ones with biggest difference?
# 2) Are vertices with high ISC in age also high in sex?
# How to do this?? - Sex is not a continuous variable!!
'''
iscf = ISCpath+'ISC_2019-05-28.h5'
subord,phenol = dd.io.load(metaphenopath+'pheno_'+iscf.split('ISC_')[1],['/subs','/phenodict'])
phenol = { your_key: phenol[your_key] for your_key in ['age','sex'] }

for task in ['DM','TP']:
	print(task)
	for k,v in phenol.items():
		ISC = dd.io.load(iscf,['/'+task+'/ISC_persubj_'+k])[0]
		ISCdiff = np.nan_to_num(np.nanmean(ISC[:,[i == True for i in v]],axis=1) - \
				np.nanmean(ISC[:,[i == False for i in v]],axis=1))
		ISC = np.nan_to_num(np.nanmean(ISC,axis=1))
		r,p = pearsonr(ISC,ISCdiff)
		plt.scatter(ISC,ISCdiff,alpha=0.01)
		plt.title(k+' diff vs ISC r = '+str(r)+', p = '+str(p))
		plt.xlabel('ISC')
		plt.ylabel('ISC diff')
		#plt.gcf().savefig(figurepath+k+' diff vs ISC for '+task+'.png')
		plt.show()
'''
		
	
