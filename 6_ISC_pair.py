#!/usr/bin/env python3

# Calculate pairwise ISC
# See if older subjects cluster more than younger subjects
# Used: https://github.com/TheLoneNut/CorrelationMatrixClustering/blob/master/CorrelationMatrixClustering.ipynb
# As reference for code

import os
import glob
import tqdm
import pandas as pd
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.cluster.hierarchy as sch
from HMM_settings import *
agediffroidir = path+'ROIs/SfN_2019/Fig2_'
ISCdir = ISCpath + 'Pairwise_ISC/'
Figdir = figurepath + 'Pairwise_ISC/'

plt.rcParams.update({'font.size': 15})
	
ROIl = glob.glob(agediffroidir+'*roi')
pairwisedict = {f:{'bin_'+str(b):{} for b in bins} for f in ROIl}
for f in tqdm.tqdm(ROIl):
	fbits = f.split(agediffroidir)[1].split('_')
	fshort = '_'.join(fbits)[:-7]
	hemi = fbits[1][0]
	task = fbits[0]
	nTR_ = nTR[tasks.index(task)]
	# Find the min and max of all colors for use in setting the color scale.
	vmins = []
	vmaxs = []
	if not os.path.exists(ISCdir+fshort+'.h5'):
		vs = []
		with open(f, 'r') as inputfile:
			for line in inputfile:
				if len(line.split(' ')) == 3:
					vs.append(int(line.split(' ')[1]))
		pairwisedict[f]['vs'] = vs
		for b in bins:
			bstr = 'bin_'+str(b)
			subl = [ageeq[i][1][b][idx] for i in [0,1] for idx in np.random.choice(lenageeq[i][b],minageeq[i],replace=False)]
			pairwisedict[f][bstr]['subl'] = subl
			nsub = len(subl)
			D = np.empty((nsub,nTR_),dtype='float16')
			for sidx, sub in enumerate(subl):
				D[sidx] = np.mean(dd.io.load(sub,['/'+task+'/'+hemi], sel=dd.aslice[vs,:])[0],0)
			pairwisedict[f][bstr]['D'] = D
			ISCdf = pd.DataFrame(D.T)
			X = ISCdf.corr().values
			d = pdist(X)   # vector of ('41' choose 2) pairwise distances
			L = sch.linkage(d, method='complete',optimal_ordering=True)
			c, _ = sch.cophenet(L, d)
			pairwisedict[f][bstr]['c'] = c
			ind = sch.fcluster(L, 0.5*d.max(), 'distance')
			columns = [ISCdf.columns.tolist()[i] for i in list((np.argsort(ind)))]
			ISCdf = ISCdf.reindex(columns, axis=1)
				   
			# Compute the correlation matrix for the received dataframe
			corr = ISCdf.corr()
			pairwisedict[f][bstr]['corr'] = corr
			vmins.append(np.min(np.min(corr)))
			vmaxs.append(np.max(np.max(corr.replace(1, 0))))
		dd.io.save(ISCdir+fshort+'.h5',pairwisedict[f])
	else:
		pairwisedict[f] = dd.io.load(ISCdir+fshort+'.h5')
		for b in bins:
			corr = pairwisedict[f]['bin_'+str(b)]['corr']
			vmins.append(np.min(np.min(corr)))
			vmaxs.append(np.max(np.max(corr.replace(1, 0))))
	vmin = min(vmins)
	vmax = max(vmaxs)
	fig, ax = plt.subplots(2,figsize=(10, 15))
	fig.suptitle(' '.join(fbits)+' nvox = '+str(len(pairwisedict[f]['vs'])), y=0.94,x=0.55)
	image = []
	for bi,b in enumerate(bins):
		lab = 'Ages '+str(int(round(eqbins[b])))+' - '+str(int(round(eqbins[b+1])))
		corr = pairwisedict[f]['bin_'+str(b)]['corr']
		# Plot the correlation matrix
		image.append(ax[bi].matshow(corr, vmax=vmax, vmin=vmin, cmap='RdYlGn'))
		plt.sca(ax[bi])
		plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
		plt.yticks(range(len(corr.columns)), corr.columns)
				   
		ax[bi].set_title(lab+',c = '+str(round(pairwisedict[f]['bin_'+str(b)]['c'],2)))
	cbar = fig.colorbar(image[0], ax=ax, orientation='vertical', fraction=.05)
	cbar.set_label('Correlation (r)')
	#fig.tight_layout()
	#plt.show()
	fig.savefig(Figdir+fshort+'.png')
	


		
		
		

		
		
