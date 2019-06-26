#!/usr/bin/env python3

# Analyze ISC findings:
# 1) Are vertices with high ISC in both groups (either age or sex) ones with biggest difference?
# 2) Are vertices with high ISC in age also high in sex?
# How to do this?? - Sex is not a continuous variable!!
# 

import glob
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from datetime import datetime
import numpy as np
import deepdish as dd
from settings import *

iscf = ISCpath+'ISC_'+str(min([datetime.strptime(i.split('/')[-1].split('.h5')[0].split('ISC_')[1],'%Y-%m-%d') for i in glob.glob(ISCpath+'ISC_*') if '2019' in i], key=lambda x: abs(x - datetime.today()))).split(' ')[0]+'.h5'
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
		plt.gcf().savefig(figurepath+k+' diff vs ISC for '+task+'.png')
		plt.show()
	
