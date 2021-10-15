#!/usr/bin/env python3

import tqdm
import random
import numpy as np
import deepdish as dd
from ISC_settings import *

nsub = 40
bins = [0,4]
task='DM'
n_time=750

D2 = {}
outliers = []
vals2 = {}
vals3 = {}
for b in range(nbinseq):
	subl = np.concatenate([ageeq[i][1][b] for i in [0,1]])
	D2[b] = np.zeros((len(subl),n_time))
	for sidx, sub in enumerate(subl):
		D2[b][sidx] = dd.io.load(sub,['/'+task+'/reg'])[0][:,2]
	vals2[b] = np.median(D2[b],1)
	vals3[b] = vals2[b][vals2[b] < np.std(vals2[0])*3]
	outliers.extend(subl[vals2[b] > np.std(vals2[0])*3])

if __name__ == "__main__":
	import scipy.stats as stats
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(figsize=(5, 5))	
	histbins=np.histogram(np.hstack((vals3[0],vals3[4])), bins=15)[1]
	for b in bins:
		ax.hist(vals3[b], histbins)
	ax.legend(['Young', 'Old'])
	fig.tight_layout()
	stats.ttest_ind(vals3[0],vals3[4])
	df = len(vals3[0])+len(vals3[4])-2

			