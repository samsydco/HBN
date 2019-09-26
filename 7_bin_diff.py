#!/usr/bin/env python3

# which age bins are most different?


import matplotlib.pyplot as plt
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



						
			
					

		
	
