#!/usr/bin/env python3

# Make HMM plots (for each ROI):
# 1) within_r - across_r across k's, for both age groups
# 2) tune_ll across k's, for both age groups
# 3) for best fitting k, plot boundary timing for both age groups
import os
import h5py
import tqdm
import glob
import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
import matplotlib as mpl
from HMM_settings import *

ROIopts = ['YeoROIsforSRM_sel_2020-01-14.h5','YeoROIsforSRM_2020-01-03.h5','ROIs/SfN_2019/ROIs_Fig3/Fig3_','ROIs/g_diff/']
ROInow = ROIopts[0]
HMMf = HMMpath+ROInow+'/'

#from datetime import date,datetime
#HMMfdate = str(min([datetime.strptime(i.split('/')[-1].split('.h5')[0].split('HMM_')[1],'%Y-%m-%d') for i in glob.glob(HMMpath+'*') if 'fig' not in i], key=lambda x: abs(x - datetime.today()))).split(' ')[0]
#HMMf = HMMpath+'HMM_'+HMMfdate+'.h5'#'HMM_2019-09-12.h5'
HMMff = HMMf+'fig.h5'
ROIl = [f for f in glob.glob(HMMf+'*h5') if 'fig' not in f]#list(dd.io.load(HMMf).keys())
nROI = len(ROIl)

def plot_tt_similarity_matrix(ax, data_matrix, bounds, start,n_TRs, TR, cmin, cmax, flip=False):
	# upper or lower triangle zeroed-out
	# May need to edit k based on tril vs triu!!!!!
    cc = np.tril(np.corrcoef(data_matrix.T), k=-1) if flip else np.triu(np.corrcoef(data_matrix.T), k=1)
    
    cmap = plt.cm.viridis
    colors = Normalize(cmin, cmax, clip=True)(cc)
    colors = cmap(colors)
    colors[..., -1] = cc != 0

    im = ax.imshow(colors, extent=(start*TR-0.5, n_TRs*TR-0.5, n_TRs*TR-0.5, start*TR-0.5))

    if flip:
        plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=cmin, vmax=cmax), cmap=cmap), ax=ax)
    ax.set_xlabel('Time [min]')#'Time (sec)')
    ax.set_ylabel('Time [min]')#'Time (sec)')
    time = np.arange(start*TR,n_TRs*TR+1,(n_TRs-start)*TR//4)
    ax.set_xticks(time)
    ax.set_yticks(time)
    ax.set_xticklabels([str(int(s//60))+':'+str(int(s%60)) for s in time])
    ax.set_yticklabels([str(int(s//60))+':'+str(int(s%60)) for s in time])
    # plot the boundaries 
    bounds_aug = np.concatenate(([0],bounds,[n_TRs]))*TR
    for i in range(len(bounds_aug)-1):
        if i >= start or i <= n_TRs:
            if flip:
                l = mlines.Line2D([bounds_aug[i],bounds_aug[i]],[bounds_aug[i],bounds_aug[i+1]],
							  linewidth=2,color='w')
                ax.add_line(l)
                l = mlines.Line2D([bounds_aug[i],bounds_aug[i+1]],[bounds_aug[i+1],bounds_aug[i+1]],
							  linewidth=2,color='w')
                ax.add_line(l)
            else:
                l = mlines.Line2D([bounds_aug[i],bounds_aug[i+1]],[bounds_aug[i],bounds_aug[i]],
							  linewidth=2,color='w')
                ax.add_line(l)
                l = mlines.Line2D([bounds_aug[i+1],bounds_aug[i+1]],[bounds_aug[i],bounds_aug[i+1]],
							  linewidth=2,color='w')
                ax.add_line(l)
    l = mlines.Line2D(np.array([start*TR,n_TRs*TR])-0.5,np.array([start*TR,n_TRs*TR])-0.5,
					  linewidth=6,color='w')
    ax.add_line(l)

# pick the best model parameter
best_model_id = {key: {key: {key: [] for key in ['bin_0','bin_4']} for key in ROIl} for key in tasks}
best_k = {key: {key: {key: [] for key in ['bin_0','bin_4']} for key in ROIl} for key in tasks}
linelist = ['-','--',':']
dashList = [(5,2),(2,5),(4,10),(3,3,2,2),(5,2,20,2)]

def extend_for_TP(array,task):
	xnans = np.nan*np.zeros(4)
	if task == 'DM':
		array = array
	else:
		array = np.concatenate((array,xnans), axis=0)
	return array

# make a dict with log likelihoods to save for later: plotting on brain!
lls = {f:{task:{b:{key: np.zeros((len(k_list),nsplit)) for key in ['tune_ll','perm_ll','mismatch_ll']} for b in bins} for task in tasks} for f in ROIl}
for f in ROIl:
	fshort = f.split('/')[-1][:-3]
	lls[fshort] = {task:{b:{key: np.zeros((len(k_list),nsplit)) for key in ['tune_ll','perm_ll','mismatch_ll']} for b in bins} for task in tasks}
	figs = {key: plt.subplots(2,sharex=True) for key in ['fig1','fig2','fig3']}
	ax = {key: [] for key in ['fig1','fig2','fig3']}
	for k,v in figs.items():
		#axa = figs[k][0]  # The big subplot
		axa = figs[k][0].add_subplot(111, frameon=False)
		axa.set_title(fshort)
		axa.set_xlabel('Average Event Duration',labelpad=20)
		if '1' in k:
			axa.set_ylabel('Within- minus across-boundary\npattern correlation (r)',labelpad=40)
		else:
			axa.set_ylabel('Log likelihood\nPer TR',labelpad=40)
		axa.set_yticks([],[])
		axa.set_xticks([],[])
	for ti,task in enumerate(tasks):
		nTR_ = nTR[ti]
		x_list = [np.round(TR*(nTR_/k),2) for k in k_list]
		if task == 'TP': x_list = x_list+[120,150,200,300]
		for k,v in figs.items():
			ax[k].append(figs[k][1][ti])
			ax[k][ti].set_title(task)
			if ti == 1:
				ax[k][ti].set_xticks(x_list,[])
				ax[k][ti].set_xticklabels([])
		for b in bins:
			was = {key: np.zeros((len(k_list),nsplit,len(win_range))) for key in ['wa','perm']}
			try:
				for key in lls[fshort][task][b]:
					lls[fshort][task][b][key] = dd.io.load(f,'/'.join(['/'+task,'bin_'+str(b),key]))
				for key in was:
					was[key] = dd.io.load(f,'/'.join(['/'+task,'bin_'+str(b),key]))
			except ValueError:
				for ki,k in enumerate(k_list):
					for split in range(nsplit):
						bstr = '/'.join(['/'+task,'bin_'+str(b),'split_'+str(split),'k_'+str(k)])
						for key in lls[fshort][task][b]:
							lls[fshort][task][b][key][ki,split] = np.mean(dd.io.load(f,'/'.join([bstr,key])))
						for wi,w in enumerate(win_range):
							bstrw = '/'.join([bstr,'w_'+str(w)])
							was['wa'][ki,split,wi] = np.mean(dd.io.load(f,'/'.join([bstrw,'within_r']))) - \
													 np.mean(dd.io.load(f,'/'.join([bstrw,'across_r'])))
							perms = dd.io.load(f,'/'.join([bstrw,'perms']))
							was['perm'][ki,split,wi] = np.mean([np.mean(perms[p]) for p in perms if 'within' in p]) - np.mean([np.mean(perms[p]) for p in perms if 'across' in p]) 
				hf = h5py.File(f,'a') # open the file
				for key in lls[fshort][task][b]:
					hf.create_dataset('/'.join(['/'+task,'bin_'+str(b),key]), data=lls[fshort][task][b][key])
					#data = hf['/'.join(['/'+task,'bin_'+str(b),key])]       # load the data
					#data[...] = lls[key]
				for key in was:
					hf.create_dataset('/'.join(['/'+task,'bin_'+str(b),key]),data=was[key])
					#data = hf['/'.join(['/'+task,'bin_'+str(b),key])]       # load the data
					#data[...] = was[key]
				hf.close()		
			c = '#1f77b4' if b == 0 else '#ff7f0e'
			lab = 'Ages '+str(int(round(eqbins[b])))+' - '+str(int(round(eqbins[b+1])))
			best_model_id[task][fshort]['bin_'+str(b)] = np.argmax(np.mean(lls[fshort][task][b]['tune_ll'],1))
			best_k[task][fshort]['bin_'+str(b)] = k_list[best_model_id[task][fshort]['bin_'+str(b)]]
			fig1y = extend_for_TP(np.mean(np.nanmean(was['wa'],1),1),task)
			fig1yerr = extend_for_TP(np.mean(np.nanmean(was['perm'],1),1),task)
			ax['fig1'][ti].errorbar(x_list, fig1y, yerr=fig1yerr,color = c,label=lab) #Need nanmean for TP, 50 events (some events are shorter than longest window)			
			for lli,ll in enumerate(lls[fshort][task][b].keys()):
				fig2y = extend_for_TP(np.mean(lls[fshort][task][b][ll],1)/nTR_,task)
				fig2yerr = extend_for_TP(np.std(lls[fshort][task][b][ll],1)/nTR_,task)
				ax['fig2'][ti].errorbar(x_list, fig2y, yerr=fig2yerr, color=c, linestyle=linelist[lli],label=lab+' '+ll)
			fig3y = extend_for_TP(np.mean(lls[fshort][task][b]['tune_ll'],1)-np.mean(lls[fshort][task][b]['perm_ll'],1),task)
			subvar = extend_for_TP(np.sqrt(np.var(lls[fshort][task][b]['tune_ll'],1)+np.var(lls[fshort][task][b]['perm_ll'], 1)-[2*np.mean((lls[fshort][task][b]['tune_ll'][i,:]-np.mean(lls[fshort][task][b]['tune_ll'][i,:]))*(lls[fshort][task][b]['perm_ll'][i,:]-np.mean(lls[fshort][task][b]['perm_ll'][i,:]))) for i in range(len(k_list))]),task)
			ax['fig3'][ti].errorbar(x_list,fig3y, yerr=subvar, color=c,label=lab+' tune - perm ll')
			for key in ['fig2','fig3']:
				ax[key][ti].axvline(x=TR*(nTR_/best_k[task][fshort]['bin_'+str(b)]),color=c,linestyle='--')
	lgd = {}
	for k,v in figs.items():
		ax[k][ti].set_xticklabels(x_list,rotation=45)
		figs[k][0].set_size_inches(6, 6)
		figs[k][0].tight_layout()
		lgd[k] = ax[k][ti].legend(loc='lower right', bbox_to_anchor=(1.55, 0))
	plt.show()
	figs['fig1'][0].savefig(figurepath+'HMM/within_r-across_r/'+fshort+'.png', bbox_extra_artists=(lgd['fig1'],), bbox_inches='tight')
	figs['fig2'][0].savefig(figurepath+'HMM/tune_ll/'+fshort+'.png', bbox_extra_artists=(lgd['fig2'],), bbox_inches='tight')
	figs['fig3'][0].savefig(figurepath+'HMM/tune_sub_ll/'+fshort+'.png', bbox_extra_artists=(lgd['fig2'],), bbox_inches='tight')
	
dd.io.save(HMMpath+'lls_'+ROInow,lls)


plt.rcParams.update({'font.size': 15})
bounds_aug = {key: {key: {key: [] for key in ['bin_0','bin_1']} for key in ROIl} for key in tasks}
for f in ROIl: # ROIs
	fshort = f.split('/')[-1][:-3]
	for task in ['TP']: # still need to do this for DM...
		nTR_ = [nTR[i] for i in range(len(tasks)) if tasks[i]==task][0]
		plt.figure(figsize=(8,8))
		for bi,b in enumerate(bins): # For young/old group
			bstr = '/'+task+'/bin_'+str(b)+'/'
			ftf = True if b == 0 else False
			D = np.mean(dd.io.load(f,bstr+'D'),axis=0)
			kb = best_k[task][f]['bin_'+str(b)]
			segs = dd.io.load(f,bstr+'all_sub_events/k_'+str(kb)+'/seg_og')
			bounds = np.where(np.diff(np.argmax(segs, axis=1)))[0]
			plot_tt_similarity_matrix(plt.gca(), D, bounds, 0, nTR_, TR=TR, cmin=-0.5, cmax=0.7, flip=ftf)
			bounds_aug[task][f]['bin_'+str(b)] = [str(int(s//60))+':'+str(int(s%60)) for s in np.concatenate(([0],bounds,[nTR_]))*TR]
		plt.title(fshort+'\nYoung, k ='+str(best_k[task][f]['bin_0'])+'Old, k ='+str(best_k[task][f]['bin_4']))
		plt.text(50, 150, 'Young\nk = '+str(best_k[task][f]['bin_0']), fontsize=24, color='white')
		plt.text(150, 100, 'Old\nk = '+str(best_k[task][f]['bin_4']), fontsize=24, color='white')
		plt.savefig(figurepath+'HMM/xcorr/'+fshort+'_'+task+'.png')
		
# Young Events X Old Events xcorr matrix
# Forgot to make a pattern over all splits :(
#sns.set_style("white")
dfs = {key: {key: [] for key in ROIl} for key in tasks}
for task in ['TP']: # still need to do this for DM...
	for f in ROIl:
		fig = plt.figure()
		binpat = {key: [] for key in ['Young','Old']}
		for p in binpat.keys():
			b = '0' if p == 'Young' else '4'
			k = best_k[task][f]['bin_'+b] if b == '0' else best_k[task][f]['bin_'+b]
			lstr = '/'+task+'/bin_'+b+'/k_'+str(k)
			binpat[p] = dd.io.load(f,'/'.join(['/'+task,'bin_'+str(b),'all_sub_events','k_'+str(k),'pattern'])).T
		dfs[task][f] = np.corrcoef(binpat['Young'],binpat['Old'])[:best_k[task][f]['bin_0'],best_k[task][f]['bin_0']:]
		plt.imshow(dfs[task][f],cmap='viridis')
		plt.ylabel('Young Events')
		plt.xlabel('Old Events')
		plt.colorbar()
		plt.show()
		fig.savefig(figurepath+'HMM/event_corr/'+f.split('/')[-1][:-3]+'_'+task+'.png')
		
		


