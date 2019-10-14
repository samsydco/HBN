#!/usr/bin/env python3

# which age bins are most different?

import matplotlib.pyplot as plt
from datetime import date
import numpy as np
import deepdish as dd
import seaborn as sns
import pandas as pd
from settings import *
from ISC_settings import *
agediffroidir = path+'ROIs/SfN_2019/Fig2_'#'ROIs/agediff/'

# what voxels change in ISC with age, and in which direction?
iscf = ISCpath + 'ISC_2019-09-06_age_2.h5'#'ISC_2019-08-13_age_2.h5'

plt.rcParams.update({'font.size': 15})
xticks = [str(int(round(eqbins[i])))+\
		  ' - '+str(int(round(eqbins[i+1])))+' y.o.' for i in range(len(eqbins)-1)]
for f in glob.glob(agediffroidir+'*roi'):#glob.glob(agediffroidir+'*v2*'):
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
	print(len(vs))
	fdict = {'Age':[],'ISC':[],'vox':[],'corr':[],'null':[]}
	null = np.zeros((nbinseq,nshuff,len(vs)))
	ISCs = np.zeros((nbinseq,len(vs)))
	vadd = vs if fbits[1] == 'LH' else [v+40962 for v in vs]
	#vadd = vs if fbits[3] == 'lh' else [v+40962 for v in vs]
	for b in range(nbinseq):
		ISCs[b] = dd.io.load(iscf,'/shuff_0/'+task+'/bin_'+str(b)+'/ISC_w')[vadd]
		for n in range(1,nshuff+1):
			null[b,n-1] = dd.io.load(iscf,'/shuff_'+str(n)+'/'+task+'/bin_'+str(b)+'/ISC_w')[vadd]
		fdict['null'].extend(np.mean(null[b],axis=0))
		fdict['Age'].extend([xticks[b]]*len(vs))
		fdict['ISC'].extend(ISCs[b])
		fdict['vox'].extend(vadd)
	for idx in range(len(vadd)):
		fdict['corr'].extend([np.corrcoef([e+agespan/2 for e in eqbins[:-1]],ISCs[:,idx])[0,1] for b in range(nbinseq)])
	color = 'tab:red' if np.mean(fdict['corr'])>0 else 'royalblue'

	nullmean = null.mean()
	nullstd = np.std(null,axis=1,ddof=1).mean()
	df = pd.DataFrame(data=fdict)
	fig = plt.fill_between(np.arange(nbinseq+1)-0.5,
                    nullmean-nullstd, nullmean+nullstd,
                    alpha=0.2, edgecolor='none', facecolor='grey')
	fig = sns.swarmplot(x="Age", y="ISC", data=df,zorder=1,color=color)
	fig = sns.pointplot(x="Age", y="ISC", data=df,markers='+',join=False,color='k',ci='sd',capsize=.1, zorder=100)
	plt.xticks(rotation=30,ha="right")
	plt.tight_layout()
	plt.show()
	fig.figure.savefig(figurepath+'SfN_2019/Figure_2/'+'_'.join(fbits)[:-7]+'.png')



						
			
					

		
	
