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
smooth = True
smoothtimes = 6

# what voxels change in ISC with age, and in which direction?
iscf = ISCpath + 'ISC_2019-08-13_age_2.h5'
agediff_f = ISCpath + 'ISC_' + str(date.today())+'_agediff'
agediff_f = agediff_f+'_2' if 'age_2' in iscf and not smooth else agediff_f+'_smooth_' if smooth else agediff_f
agediff_f = agediff_f+'.h5'
if os.path.exists(agediff_f):
    os.remove(agediff_f)
	
if smooth:
	global cols
	cols = {}
	for hem in ['left','right']:
		hemi = 'lh' if hem == 'left' else 'rh'
		X = (dd.io.load('/data/Schema/intact/fsaverage6_adj.h5','/'+hem))
		cols[hemi] = [None] * (len(X['jc'])-1)
		for i in range(len(cols[hemi])):
			cols[hemi][i] = X['ir'][X['jc'][i]:X['jc'][i+1]]
	def smooth_fun(ISC):
		ISC2 = np.zeros(ISC.shape)
		for hemi in ['lh','rh']:
			colh = cols[hemi]
			idxs = 	np.arange(0,len(ISC)//2) if hemi == 'lh' else np.arange(len(ISC)//2,len(ISC))
			for idx1,idx in enumerate(idxs):
				ISC2[idx] = np.mean([ISC[idx],np.mean(ISC[colh[idx1]])])
		return ISC2

'''
# which age bins are most different?
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
	vadd = vs if fbits[3] == 'lh' else [v+40962 for v in vs]
	for b in range(nbinseq):
		if fbits[2] == 'all':
			shl = list(dd.io.load(iscf)['shuff_0']['bin_0'][task].keys())
			for i in shl:
				ISCs[b] += dd.io.load(iscf,
				'/shuff_0/bin_'+str(b)+'/'+task+'/'+i)[vadd]
			ISCs[b] = ISCs[b]/len(shl)
		fdict['Age'].extend(xticks[b]*np.ones(len(vs)))
		fdict['ISC'].extend(ISCs[b])
		fdict['vox'].extend(vadd)
	for idx in range(len(vadd)):
		fdict['corr'].extend([np.corrcoef([e+agespan/2 for e in eqbins[:-1]],ISCs[:,idx])[0,1] for b in range(nbinseq)])
	
	#ax = plt.errorbar(xticks, list(df.groupby('Age')['ISC'].mean()), yerr=list(df.groupby('Age')['ISC'].std()),ls='none', fmt='k+', markersize=25, capsize=5)
	df = pd.DataFrame(data=fdict)
	fig = sns.swarmplot(x="Age", y="ISC", data=df,zorder=1)
	fig = sns.pointplot(x="Age", y="ISC", data=df,markers='+',join=False,color='k',ci='sd',capsize=.1, zorder=100)
	plt.show()
	fig.figure.savefig(figurepath+'agediff/'+'_'.join(fbits)+'.png')
	
	#df = pd.DataFrame(data=fdict)
	#g = sns.FacetGrid(df, col="corr", col_wrap=len(vs)//20)
	#g = g.map(plt.plot, "Age", "ISC", marker=".")
	#g.savefig(figurepath+'agediff_'+'_'.join(fbits)+'.png')

	#ax = sns.swarmplot(x='Age',y='ISC',hue='vox',data=df,palette="Set1")
	#l = ax.legend()
	#l.remove()
	
	#ax = plt.plot(np.matlib.repmat(xticks,len(vs),1).T,ISCs)
	#ax = sns.swarmplot(data=ISCs)
'''

#with h5py.File(agediff_f) as hf:
anall = ['corr','spearman','stddivmean','diffidx','diff']
with h5py.File(agediff_f,'a') as hf:
	for task in ['DM','TP']:
		grp = hf.create_group(task)
		for comp in ['all']:#['0','1','all','err_diff','g_diff']:
			print(task,comp)
			#verts = {key: np.zeros(81924) for key in anall}
			for shuff in tqdm.tqdm(list(dd.io.load(iscf).keys())):
				ISCs = {key: np.zeros((81924,smoothtimes)) for key in anall}
				ISC = np.zeros((nbinseq,81924))
				ISCsm = np.zeros((nbinseq,81924,smoothtimes))
				for b in range(nbinseq):
					if comp != 'all' and 'age_2' in iscf:
						print('Must do computation over "all" with this iscf')
						break
					elif comp == 'all' and 'age_2' in iscf:
						ISC[b] = dd.io.load(iscf,
								 '/'+shuff+'/bin_'+str(b)+'/'+task+'/ISC_SH')
					elif comp in ['0','1'] and 'age_2' not in iscf:
						ISC[b] = dd.io.load(iscf,
							'/'+shuff+'/bin_'+str(b)+'/'+task+'/ISC_SH_w_'+s)
					elif comp == 'err_diff' and 'age_2' not in iscf:
						ISC[b] = dd.io.load(iscf,
						'/'+shuff+'/bin_'+str(b)+'/'+task+'/ISC_SH_w_1') - \
						dd.io.load(iscf,
						'/'+shuff+'/bin_'+str(b)+'/'+task+'/ISC_SH_w_0')
					elif comp == 'g_diff' and 'age_2' not in iscf:
						for i in ['ISC_SH_b_0_0', 'ISC_SH_b_0_1', 'ISC_SH_b_1_0', 'ISC_SH_b_1_1']:
							ISC[b] += dd.io.load(iscf,
							'/'+shuff+'/bin_'+str(b)+'/'+task+'/'+i)
						ISC[b] = ISC[b]/4/ \
						(np.sqrt(dd.io.load(iscf,
						'/'+shuff+'/bin_'+str(b)+'/'+task+'/ISC_SH_w_1'))
						*np.sqrt(dd.io.load(iscf,
						'/'+shuff+'/bin_'+str(b)+'/'+task+'/ISC_SH_w_0')))
					elif comp == 'all' and 'age_2' not in iscf:
						for i in list(dd.io.load(iscf)['shuff_0']['bin_0'][task].keys()):
							ISC[b] += dd.io.load(iscf,
							'/'+shuff+'/bin_'+str(b)+'/'+task+'/'+i)
						ISC[b] = ISC[b]/6
					for s in range(smoothtimes):
						ISC[b] = smooth_fun(ISC[b])
						ISCsm[b,:,s] = ISC[b]
				for v in range(81924):
					for s in range(smoothtimes):
						ISCs['corr'][v,s] = np.corrcoef([e+agespan/2 for e in eqbins[:-1]],ISCsm[:,v,s])[0,1]
						diffidx = np.argmax(abs(np.diff(ISCsm[:,v,s])))
						if np.sum(np.isnan(ISCsm[:,v,s]))<4:
							ISCs['spearman'][v,s] = spearmanr([e+agespan/2 for e in eqbins[:-1]],ISCsm[:,v,s])[0]
							ISCs['diffidx'][v,s] = (diffidx+1)*np.sign(np.diff(ISCsm[:,v,s])[diffidx])
							ISCs['diff'][v,s] = np.max(abs(np.diff(ISCsm[:,v,s])))*np.sign(np.diff(ISCsm[:,v,s])[diffidx])
						else:
							ISCs['spearman'][v,s] = np.nan
							ISCs['diffidx'][v,s] = np.nan
							ISCs['diff'][v,s] = np.nan
						ISCs['stddivmean'][v,s] = np.std(ISCsm[:,v,s])/np.mean(ISCsm[:,v,s])
				for k in ISCs.keys():
					for s in range(smoothtimes):
						grp.create_dataset('ISC_'+k+'_'+comp+'_sm'+str(s)+'_'+shuff,data=ISCs[k][:,s])
						
			
					




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
		
	
