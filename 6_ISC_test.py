#!/usr/bin/env python3

# 1) Compute many leave-one-out values(hundreds?), by randomly selecting subjects and correlating against the N-1 other subjects.
# 2) Compute many split-half values, by randomly selecting an N/2 split of subjects and correlating the mean timecourses of the halves.
# 3) Compute many pairwise values, by randomly selecting two subjects and correlating their timecourses.
# In all cases - time how long it takes to collect these values.
# then, compute f using subsets of each of these values

import time
import h5py
import tqdm
import numpy as np
import deepdish as dd
from scipy.stats import zscore
from scipy.spatial.distance import squareform
from random import shuffle
import matplotlib.pyplot as plt
from settings import *
# Only using 233 subj
subord = dd.io.load(metaphenopath+'pheno_2019-05-28.h5',['/subs'])[0]
ISCf = ISCpath+'old_ISC/ISC_2019-05-28.h5'
n_subj = len(subord)
n_vox = 5
ISCversions = ['Loo','SH','Pair']

# Some math to conver between correlation values
def corr_convert(r,N,corrtype='SH'):
	if corrtype == 'SH':
		f = 2*r/(N*(1-r))
	elif corrtype == 'Pair':
		f = r/(1-r)
	else: 
		f = (N*np.square(r)+np.sqrt((N**2)*np.power(r,4,dtype=np.float16)+4*np.square(r)*(N-1)*(1-np.square(r))))/(2*(N-1)*(1-np.square(r)))
	r_pw = f/(f+1)
	r_sh = (N*f)/(N*f+2)
	r_loo = (np.sqrt(N-1)*f)/(np.sqrt(f+1)*np.sqrt((N-1)*f+1))
	return r_pw,r_sh,r_loo

for task in ['DM','TP']:
	print(task)
	non_nan_verts = np.where(~np.isnan(np.concatenate([dd.io.load(subord[0],['/'+task+'/L'])[0], dd.io.load(subord[0],['/'+task+'/R'])[0]], axis=0))[:,0])[0]
	dictall = {k:{'ISC':None,'Time':None} for k in ISCversions}
	_,n_time = dd.io.load(subord[0],['/'+task+'/L'])[0].shape
	dictall['verts'] = non_nan_verts[np.random.choice(len(non_nan_verts),n_vox,replace=False)]
	D = np.empty((n_vox,n_time,n_subj),dtype='float16')
	keys = list(h5py.File(ISCf)[task]['data'].keys())
	for key in keys:
		D[:,0+250*int(key):250+250*int(key),:] = dd.io.load(ISCf,['/'+task+'/data/'+key], sel=dd.aslice[dictall['verts'],:,:])[0]
	# Loo
	print('Leave one out...')
	dictall['Loo']['ISC'] = np.zeros((n_vox,n_subj),dtype='float16')
	dictall['Loo']['Time'] = []
	# Loop across choice of leave-one-out subject
	for loo_subj in tqdm.tqdm(range(n_subj)):
		t = time.process_time()
		group = np.zeros((n_vox,n_time),dtype='float16')
		groupn = np.ones((n_vox,n_time),dtype='int')*n_subj-1
		for i in range(n_subj):
			if i != loo_subj:
				group = np.nansum(np.stack((group,D[:,:,i])),axis=0)
				nanverts = np.argwhere(np.isnan(D[:,:,i]))
				groupn[nanverts[:, 0],nanverts[:,1]] = groupn[nanverts[:,0],nanverts[:,1]]-1
		group = zscore(group/groupn,axis=1)
		subj = zscore(D[:, :, loo_subj],axis=1)
		dictall['Loo']['ISC'][:,loo_subj] = np.sum(np.multiply(group,subj),axis=1)/(n_time-1)
		dictall['Loo']['Time'].append(time.process_time() - t)
	# SH
	print('Split Halves...')
	dictall['SH']['ISC'] = np.zeros((n_vox,n_subj),dtype='float16')
	dictall['SH']['Time'] = []
	subjl = np.arange(n_subj)
	for sh in tqdm.tqdm(range(n_subj)):
		t = time.process_time()
		shuffle(subjl)
		groups = np.zeros((2,n_vox,n_time),dtype='float16')
		for h in [0,1]:
			group = np.zeros((n_vox,n_time),dtype='float16')
			groupn = np.ones((n_vox,n_time),dtype='int')*n_subj
			for i in subjl[0+n_subj//2*h:n_subj//2+n_subj//2*h]:
				group = np.nansum(np.stack((group,D[:,:,i])),axis=0)
				nanverts = np.argwhere(np.isnan(D[:,:,i]))
				groupn[nanverts[:, 0],nanverts[:,1]] = groupn[nanverts[:,0],nanverts[:,1]]-1
			groups[h] = zscore(group/groupn,axis=1)
		dictall['SH']['ISC'][:,sh] = np.sum(np.multiply(groups[0],groups[1]),axis=1)/(n_time-1)
		dictall['SH']['Time'].append(time.process_time() - t)
	# Pairwise
	print('Pairwise...')
	dictall['Pair']['Time'] = []
	for i in range(n_subj):
		voxel_iscs = []
		t = time.process_time()
		for v in np.arange(n_vox):
			voxel_data = D[v, :, :i+1].T
			# Correlation matrix for all pairs of subjects (triangle)
			iscs = squareform(np.corrcoef(voxel_data), checks=False)
			voxel_iscs.append(iscs)
		dictall['Pair']['Time'].append(time.process_time() - t)
	dictall['Pair']['ISC'] = np.column_stack(voxel_iscs)

	n_it = 10
	for g in ISCversions:
		i = dictall[g]
		i['TimeCum'] = []
		i['ISCCum'] = np.zeros((n_subj-1,n_vox,n_it))
		i['f'] = np.zeros((n_subj-1,n_vox,n_it))
		for s in np.arange(2,n_subj+1):
			i['TimeCum'].append(np.sum(i['Time'][:s]))
			for it in range(n_it):
				if g != 'Pair':
					randsubjs = np.random.choice(n_subj,s,replace=False)
					i['ISCCum'][s-2,:,it] = np.nanmean(i['ISC'][:,randsubjs],axis=1)
				else:
					randsubjs = np.random.choice(int((n_subj*n_subj-n_subj)/2),int((s*s-s)/2),replace=False)
					i['ISCCum'][s-2,:,it] = np.nanmean(i['ISC'][randsubjs,:],axis=0)
			N = n_subj
			r = i['ISCCum'][s-2]
			if g == 'Pair':
				i['f'][s-2] = r/(1-r)
			if g == 'SH':
				i['f'][s-2] = 2*r/(N*(1-r))
			if g == 'Loo':
				i['f'][s-2] = (N*np.square(r)+ \
				 np.sqrt((N**2)*np.power(r,4,dtype=np.float16)+4*np.square(r)*(N-1)*(1-np.square(r))))/(2*(N-1)*(1-np.square(r)))
		
	figsubj = 200
	fig = plt.figure()
	for v in range(n_vox):
		ax = fig.add_subplot(n_vox,1,v+1)
		if v == 0:
			ax.set_title(task)
		for g in ISCversions:
			i = dictall[g]
			# plot Time vs Accuracy:
			final_f = np.mean(i['f'][-1,v,:])
			y = np.mean(i['f'][:figsubj,v,:]-final_f,axis=1)
			error = np.max(i['f'][:figsubj,v,:]-final_f,axis=1)
			ax.plot(i['TimeCum'][:figsubj],y,label=g)
			ax.fill_between(i['TimeCum'][:figsubj], y-error, y+error,alpha=0.2)
			ax.set_ylabel('f acc\nvert =\n'+str(dictall['verts'][v]),size=7)
			#if g == 'Pair':
			#	ax.set_ylim(y[0]-error[0],y[0]+error[0])
		ax.set_xlim(min(dictall['Pair']['TimeCum'][:figsubj]),max(dictall['SH']['TimeCum'][:figsubj]))
		if v == n_vox-1:
			ax.legend(bbox_to_anchor=(0.7,1,0.5,3.5))
			ax.set_xlabel('Time [s]')
	fig.savefig(figurepath+'TimeVsFacc_'+task+'_'+str(figsubj)+'.png', bbox_inches = "tight",dpi=300)
		
	for figsubj in [25,50,n_subj-1]:
		fig = plt.figure()
		for v in range(n_vox):
			ax0 = fig.add_subplot(n_vox+1,1,v+1)
			if v == 0:
				ax0.set_title(task)
			for g in ISCversions:
				i = dictall[g]
				# plot subj vs f for vox:
				ax0.plot(np.arange(1,figsubj+1),np.mean(i['f'][:figsubj,v,:],axis=1))
				final_f = np.mean((dictall['SH']['f'][-1,v,:]+dictall['Loo']['f'][-1,v,:]+dictall['Pair']['f'][-1,v,:])/3)
				ax0.plot([figsubj-1,figsubj],[final_f,final_f],'k-')
				ax0.set_ylabel('f\nvert =\n'+str(dictall['verts'][v]),size=7)
				ax0.set_xlim(2,figsubj)
				if v == n_vox-1:
					ax1 = fig.add_subplot(n_vox+1,1,n_vox+1)
					ax1.plot(np.arange(1,figsubj+1),i['TimeCum'][:figsubj],label=g) # plot subj vs compute time for vox
					ax1.set_xlim(2,figsubj)
		ax1.legend(bbox_to_anchor=(0.7,1,0.5,3.5))
		ax1.set_xlabel('Subjects')
		ax1.set_ylabel('Time [s]')
		fig.savefig(figurepath+'FvsCompT_'+task+'_'+str(figsubj)+'.png', bbox_inches = "tight",dpi=300)
		
	with h5py.File(ISCpath+'ISC_test.h5') as hf:
		grp = hf.create_group(task)
		if type(i) == dict:
			for g,i in dictall.items():
				ds = grp.create_group(g)
				for gi,ii in i.items():
					ds.create_dataset(gi,data=ii)
			else:
				grp.create_dataset(g,data=i)
				

	
	
	