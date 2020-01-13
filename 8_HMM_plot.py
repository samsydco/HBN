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
from datetime import date,datetime
from random import shuffle
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
import matplotlib as mpl
from HMM_settings import *

ROIopts = ['YeoROIsforSRM_2020-01-03.h5','ROIs/SfN_2019/ROIs_Fig3/Fig3_','ROIs/g_diff/']
ROInow = ROIopts[0]
HMMf = HMMpath+ROInow+'/'

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
    #ax.set_xticks(time)
    #ax.set_yticks(time)
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

win_strs = [str(w) for w in win_range]
if not os.path.isfile(HMMff):
	with h5py.File(HMMff,'a') as hf:
		for fi,f in tqdm.tqdm(enumerate(ROIl)): # ROIs
			grpf = hf.create_group(f)
			for b in bins:
				grpb = grpf.create_group('bin_'+str(b))
				waa = {key: {key: [] for key in win_strs} for key in ['','_perm']}
				llsa = {key: {'mean':[],'std':[]} for key in ['tune_ll','perm_ll','mismatch_ll']}
				for i,k in enumerate(k_list):
					wa = {key: {key: [] for key in win_strs} for key in\
						  ['within_r','across_r','within_r_perm','across_r_perm']}
					lls = {key: [] for key in ['tune_ll','perm_ll','mismatch_ll']} # tune ll, perm ll, ll for other age group
					for split in range(nsplit):
						ls = dd.io.load(HMMf,'/'+f+'/bin_'+str(b)+'/split_'+str(split)+'/k_'+str(k))
						for kk in wa.keys():
							for w in win_range:
								if 'perm' in kk:
									for p in range(nshuff):
										wa[kk][str(w)].extend(ls['w_'+str(w)]['perms'][kk[:-5]+'_'+str(p)])
								else:
									wa[kk][str(w)].extend(ls['w_'+str(w)][kk])
						for kk in lls.keys():
							lls[kk].append(ls[kk])
					for wi in waa.keys():
						for w in win_strs:
							waa[wi][w].append(np.mean(wa['within_r'+wi][w]) - np.mean(wa['across_r'+wi][w]))
					for kk in llsa:
						llsa[kk]['mean'].append(np.mean(lls[kk]))
						llsa[kk]['std'].append(np.std(lls[kk]))
				for wi in waa.keys():
					grpw = grpb.create_group('_'+wi)
					for w in win_strs:
						grpw.create_dataset(w,data=waa[wi][w])
				for ll in llsa.keys():
					grpl = grpb.create_group(ll)
					for stat in llsa[ll].keys():
						grpl.create_dataset(stat,data=llsa[ll][stat])

figs = {key: plt.figure() for key in ['fig1','fig2']}
for k,v in figs.items():
	axa = figs[k].add_subplot(111)    # The big subplot
	axa.set_xlabel('# of Events',labelpad=20)
	if '1' in k:
		axa.set_ylabel('Within- minus across-boundary\npattern correlation (r)',labelpad=40)
	else:
		axa.set_ylabel('Tune log likelihood',labelpad=40)
	axa.set_yticks([],[])
	axa.set_xticks([],[])
	axa.spines["right"].set_visible(False)
	axa.spines["left"].set_visible(False)
# pick the best model parameter
best_model_id = {key: [] for key in ROIl}
best_k = {key: [] for key in ROIl}
linelist = ['-','--',':']
dashList = [(5,2),(2,5),(4,10),(3,3,2,2),(5,2,20,2)]				
for fi,f in tqdm.tqdm(enumerate(ROIl)):
	ax = {}
	for k,v in figs.items():
		ax[k] = figs[k].add_subplot(nROI,1,fi+1)
		ax[k].set_title(' '.join(f[3:].split('_')))
		ax[k].set_xticks(k_list,[])
		ax[k].set_xticklabels([])
	for b in bins:
		binfo = dd.io.load(HMMff,'/'+f+'/bin_'+str(b))
		c = '#1f77b4' if b == 0 else '#ff7f0e'
		lab = 'Ages '+str(int(round(eqbins[b])))+' - '+str(int(round(eqbins[b+1])))
		best_model_id[f].append(np.argmax(binfo['tune_ll']['mean']))
		best_k[f].append(k_list[best_model_id[f][-1]])
		for wi,w in enumerate(win_range):
			ax['fig1'].errorbar(k_list, binfo['_'][str(w)], yerr=binfo['__perm'][str(w)],\
								color = c, linestyle='--', dashes=dashList[wi],\
								label=lab+', win: '+str(np.round(w*0.8,1))+' secs')
		for lli,ll in enumerate(['tune_ll','perm_ll','mismatch_ll']):
			ax['fig2'].errorbar(k_list, binfo[ll]['mean'], yerr=binfo[ll]['std'], color=c, linestyle=linelist[lli],label=lab+' '+ll)
		ax['fig2'].axvline(x=best_k[f][-1],color='k',linestyle='--')
lgd = {}
for k,v in figs.items():
	ax[k].set_xticklabels(k_list)
	figs[k].set_size_inches(6, 6)
	figs[k].tight_layout()
	lgd[k] = ax[k].legend(loc='lower right', bbox_to_anchor=(1.55, 0))
plt.show()
figs['fig1'].savefig(figurepath+'HMM/within_r-across_r_'+HMMfdate+'.png', bbox_extra_artists=(lgd['fig1'],), bbox_inches='tight')
figs['fig2'].savefig(figurepath+'HMM/tune_ll_'+HMMfdate+'.png', bbox_extra_artists=(lgd['fig2'],), bbox_inches='tight')

TR=0.8
plt.rcParams.update({'font.size': 15})
bounds_aug = {key: {'0':[],'1':[]} for key in ROIl}
for f in ROIl: # ROIs
	plt.figure(figsize=(8,8))
	for bi,b in enumerate(bins): # For young/old group
		bstr = '/'+f+'/bin_'+str(b)+'/'
		ftf = True if b == 0 else False
		D = np.mean(dd.io.load(HMMf,bstr+'D'),axis=0)
		kb = best_k[f][bi]
		segs = dd.io.load(HMMf,'/'+f+'/bin_'+str(b)+'/all_sub_events'+'/k_'+str(kb)+'/seg_og')
		bounds = np.where(np.diff(np.argmax(segs, axis=1)))[0]
		plot_tt_similarity_matrix(plt.gca(), D, bounds, 0, 250, TR=0.8, cmin=-0.5, cmax=0.7, flip=ftf)
		bounds_aug[f][str(bi)] = [str(int(s//60))+':'+str(int(s%60)) for s in np.concatenate(([0],bounds,[750]))*0.8]
	#plt.text(150, 400, 'Young\nk = '+str(best_k[f][0]), fontsize=24, color='white')
	#plt.text(400, 200, 'Old\nk = '+str(best_k[f][1]), fontsize=24, color='white')
	plt.title(f+'\nYoung, k ='+str(best_k[f][0])+'Old, k ='+str(best_k[f][1]))
	#plt.title('Timepoint correlation (r)')
	if any(r in f for r in ['TPJ','PMC']):
		plt.savefig(figurepath+'SfN_2019/HMM/xcorr_'+f+'.png')
		
# Young Events X Old Events xcorr matrix
import pandas as pd
import seaborn as sns
sns.set_style("white")
dfs = {}
for f in ROIl:
	fig = plt.figure()
	binpat = {key: [] for key in ['Young','Old']}
	for p in binpat.keys():
		b = '0' if p == 'Young' else '4'
		k = best_k[f][0] if b == '0' else best_k[f][1]
		lstr = '/'+f+'/bin_'+b+'/all_sub_events'+'/k_'+str(k)
		binpat[p] = dd.io.load(HMMf,lstr+'/pattern').T
		#binpat[p] = pd.DataFrame(data={p+' Event '+str(e+1):binpattmp[e,:] for e in range(k)})
	dfs[f] = np.corrcoef(binpat['Young'],binpat['Old'])[:best_k[f][0],best_k[f][0]:]
	plt.imshow(dfs[f],cmap='viridis')
	plt.ylabel('Young Events')
	plt.xlabel('Old Events')
	
	plt.colorbar()
	plt.show()
	fig.savefig(figurepath+'HMM/event_corr_'+f+'.png')
		
		


