#!/usr/bin/env python3

import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
from HMM_settings import *

# Remove subjects over max(eqbins) age:
incl_idx = [a<eqbins[-1] for a in agel]
subord = [s for si,s in enumerate(subord) if incl_idx[si]]
agel,pcl,phenol = make_phenol(subord)
task = 'DM'
n_time=750
nsub=40

for HPC in ['HPC','aHPC','pHPC']:
	labid = 1 if 'p' in HPC else 2
	D = {key:{} for key in range(nbinseq)}
	print(HPC)
	for b in range(nbinseq):
		subl = [hpcprepath+subord[i].split('/')[-1] for i in [ai for ai,a in enumerate(agel) if a>=eqbins[b] and a<eqbins[b+1]]]
		for sub in subl:
			data = dd.io.load(sub,['/'+task+'/HPC'])[0]
			if HPC!='HPC':
				lab = dd.io.load(sub,['/'+task+'/aplab'])[0]
				data = data[lab==labid]		
			D[b][sub] = np.mean(ss.zscore(data,axis=1),axis=0)		
	dd.io.save(ISCpath+HPC+'.h5',{'D':D})
		
# look at HPC volume vs age:
sizedict = {'Subj':[],'Age':[],'HPC':[],'aHPC':[],'pHPC':[]}
for subi,sub_ in enumerate(subord):
	sizedict['Age'].append(agel[subi])
	sub = hpcprepath+sub_.split('/')[-1]
	sizedict['Subj'].append(sub)
	lab = dd.io.load(sub,['/'+task+'/aplab'])[0]
	sizedict['HPC'].append(len(lab))
	sizedict['aHPC'].append(np.sum(lab==1))
	sizedict['pHPC'].append(np.sum(lab==2))
dfsize = pd.DataFrame(data=sizedict)
dd.io.save(ISCpath+'HPC_vol.h5',sizedict)

grey = 211/256
color = '#8856a7'
for HPC in ['HPC','aHPC','pHPC']:
	r,p = ss.pearsonr(dfsize['Age'],dfsize[HPC])
	sns.set(font_scale = 2,rc={'axes.facecolor':(grey,grey,grey)})
	fig,ax=plt.subplots(figsize=(7,5))
	sns.regplot(x='Age', y=HPC, data=dfsize,color=color).set_title('r = '+str(np.round(r,2))+', p = '+str(np.round(p,2)))
	ax.set_xlabel('Age')
	ax.set_ylabel(HPC+' size')
	plt.rcParams['axes.xmargin'] = 0
	print(HPC,', r = '+str(np.round(r,2)),', p = '+str(np.round(p,2)))
	fig.savefig(figurepath+'HPC/'+HPC+'_size_vs_age.png', bbox_inches='tight', dpi=300)

