#!/usr/bin/env python3

# Make HMM plots (for each ROI):
# 1) within_r - across_r across k's, for both age groups
# 2) tune_ll across k's, for both age groups
# 3) for best fitting k, plot boundary timing for both age groups
import tqdm
import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt
from random import shuffle
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
import matplotlib as mpl
from HMM_settings import *

HMMf = HMMpath+'HMM_2019-09-12.h5'#'HMM_2019-09-12.h5'
ROIl = list(dd.io.load(HMMf).keys())
nROI = len(ROIl)

def plot_tt_similarity_matrix(ax, data_matrix, bounds, n_TRs, TR, cmin, cmax, flip=False):
	# upper or lower triangle zeroed-out
	# May need to edit k based on tril vs triu!!!!!
    cc = np.tril(np.corrcoef(data_matrix.T), k=-1) if flip else np.triu(np.corrcoef(data_matrix.T), k=1)
    
    cmap = plt.cm.viridis
    colors = Normalize(cmin, cmax, clip=True)(cc)
    colors = cmap(colors)
    colors[..., -1] = cc != 0

    im = ax.imshow(colors, extent=(-0.5, (n_TRs-0.5)*TR, (n_TRs-0.5)*TR, -0.5))

    if flip:
        plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=cmin, vmax=cmax), cmap=cmap), ax=ax)
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Time (sec)')
    ax.set_xticks(np.arange(0,n_TRs*TR+1,n_TRs*TR//3))
    ax.set_yticks(np.arange(0,n_TRs*TR+1,n_TRs*TR//3))
    # plot the boundaries 
    bounds_aug = np.concatenate(([0],bounds,[n_TRs]))*TR
    for i in range(len(bounds_aug)-1):
        if flip:
            l = mlines.Line2D([bounds_aug[i],bounds_aug[i]], [bounds_aug[i],bounds_aug[i+1]],linewidth=2,color='w')
            ax.add_line(l)
            l = mlines.Line2D([bounds_aug[i],bounds_aug[i+1]], [bounds_aug[i+1],bounds_aug[i+1]],linewidth=2,color='w')
            ax.add_line(l)
        else:
            l = mlines.Line2D([bounds_aug[i],bounds_aug[i+1]], [bounds_aug[i],bounds_aug[i]],linewidth=2,color='w')
            ax.add_line(l)
            l = mlines.Line2D([bounds_aug[i+1],bounds_aug[i+1]], [bounds_aug[i],bounds_aug[i+1]],linewidth=2,color='w')
            ax.add_line(l)
    l = mlines.Line2D(np.array([0,n_TRs*TR])+0.5, np.array([0,n_TRs*TR])+0.5,linewidth=6,color='w')
    ax.add_line(l)

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
win_strs = [str(w) for w in win_range]
for fi,f in tqdm.tqdm(enumerate(ROIl)): # ROIs
	ax = {}
	for k,v in figs.items():
		ax[k] = figs[k].add_subplot(nROI,1,fi+1)
		ax[k].set_title(' '.join(f[3:].split('_')))
		ax[k].set_xticks(k_list,[])
		ax[k].set_xticklabels([])
	for b in bins:
		c = '#1f77b4' if b == 0 else '#ff7f0e'
		lab = 'Ages '+str(int(round(eqbins[b])))+' - '+str(int(round(eqbins[b+1])))
		waa = {key: {key: [] for key in win_strs} for key in ['','_perm']}
		llsa = {key: {'mean':[],'std':[]} for key in ['tune_ll','perm_ll','mismatch_ll']}
		for i,k in enumerate(k_list):
			wa = {key: {key: [] for key in win_strs} for key in ['within_r','across_r','within_r_perm','across_r_perm']}
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
		best_model_id[f].append(np.argmax(llsa['tune_ll']['mean']))
		best_k[f].append(k_list[best_model_id[f][-1]])
		for wi,w in enumerate(win_range):
			ax['fig1'].errorbar(k_list, waa[''][str(w)], yerr=waa['_perm'][str(w)],\
								color = c, linestyle='--', dashes=dashList[wi],\
								label=lab+', win: '+str(np.round(w*0.8,1))+' secs')
		for lli,ll in enumerate(llsa.items()):
			ax['fig2'].errorbar(k_list, ll[1]['mean'], yerr=ll[1]['std'], color=c, linestyle=linelist[lli],label=lab+' '+ll[0])
		ax['fig2'].axvline(x=best_k[f][-1],color='k',linestyle='--')
lgd = {}
for k,v in figs.items():
	ax[k].set_xticklabels(k_list)
	figs[k].set_size_inches(6, 6)
	figs[k].tight_layout()
	lgd[k] = ax[k].legend(loc='lower right', bbox_to_anchor=(1.55, 0))
plt.show()
figs['fig1'].savefig(figurepath+'HMM/within_r-across_r.png', bbox_extra_artists=(lgd['fig1'],), bbox_inches='tight')
figs['fig2'].savefig(figurepath+'HMM/tune_ll.png', bbox_extra_artists=(lgd['fig2'],), bbox_inches='tight')

for fi,f in enumerate(ROIl): # ROIs
	plt.figure(figsize=(8,8))
	for bi,b in enumerate(bins): # For young/old group
		bstr = '/'+f+'/bin_'+str(b)+'/'
		ftf = True if b == 0 else False
		D = np.mean(dd.io.load(HMMf,bstr+'D'),axis=0)
		kb = best_k[f][bi]
		segs = np.zeros((nTR,kb))
		for split in range(nsplit):
			segs += dd.io.load(HMMf,bstr+'split_'+str(split)+'/k_'+str(kb)+'/seg_lo')
		segs = segs/nsplit
		bounds = np.where(np.diff(np.argmax(segs, axis=1)))[0]
		plot_tt_similarity_matrix(plt.gca(), D, bounds, nTR, TR=0.8, cmin=-0.5, cmax=0.7, flip=ftf)
		
	plt.text(150, 400, 'Young\nk = '+str(best_k[f][0]), fontsize=24, color='white')
	plt.text(400, 200, 'Old\nk = '+str(best_k[f][1]), fontsize=24, color='white')
	plt.title('Timepoint correlation (r)')
	plt.savefig(figurepath+'HMM/xcorr_'+f+'.png')

