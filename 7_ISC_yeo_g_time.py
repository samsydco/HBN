#!/usr/bin/env python3

# plot Yeo ROIs that correspond with TPJ
# determine what is going on at times with low g_diff

import tqdm
import matplotlib.pyplot as plt
from HMM_settings import *
from scipy.stats import pearsonr

figpath = figurepath+'g_diff_time/'
savedir = ISCpath+'shuff_Yeo/'
savef = ISCpath+'g_diff_time_vs_ev_conv.h5'
nTR = 750
TR = 0.8
x = np.arange(1,nTR*TR+1,TR)
xhun = [i*100 for i in range(7)]
xtxt = [str(i//60)+':'+str(i%60) for i in xhun]
task = 'DM'
ham = np.hamming(np.round(30))
ham = ham/np.sum(ham)
def hamconv(signal,ham):
	output = np.convolve(signal,ham,'same')
	return output

ROIs = [r.split('/')[-1][:-3] for r in glob.glob(savedir+'*')]
TPJ_ROIs = ['RH_DefaultA_IPL_1', 'LH_DefaultB_IPL_1', 'LH_SalVentAttnA_ParOper_1', 'RH_TempPar_3']#'RH_SalVentAttnA_ParOper_1', 
colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']

#fig,ax = plt.subplots()
TW = 10 # For circular time shuffle
nPerm = len(ev_conv)-TW*2
rs = {key:[] for key in ROIs}
ps = {key:[] for key in ROIs}
gall = []
gt = []
ISCall = []
rri=0
for ri,roi in tqdm.tqdm(enumerate(ROIs)):
	ISC = np.nanmean(dd.io.load(glob.glob(savedir+roi+'*')[0], '/'+task+'/ISC_g_time'), axis=1)
	zscoreISC = hamconv((ISC[0]-np.nanmean(ISC[1:], axis=0))/np.nanstd(ISC[1:], axis=0), ham)
	rperm = np.zeros(nPerm)
	for p in range(nPerm):
		rperm[p],_ = pearsonr(zscoreISC,ev_conv)
		ev_conv = np.concatenate((ev_conv[p+TW:],ev_conv[:p+TW]))
	rs[roi] = rperm[0]
	ps[roi] = np.sum(abs(rperm[0])<abs(rperm[1:]))/nPerm
	if ps[roi] < 0.05:
		print(roi,rs[roi],ps[roi])
		#lab = roi+', r = '+str(np.round(r,2)) if p< 0.05 else roi
		#gt.append(x[np.where(ISCall[ri]<-1)])
		#ax.plot(x,ISCall[ri],color=colors[ri],label=lab)
		#gall = np.intersect1d(gall,gt[ri]) if ri > 0 else gt[ri]
	
dd.io.save(savef,{'rs':rs,'ps':ps})

# For FLUX poster:
roi = 'RH_DefaultA_IPL_1'
ISCt = np.nanmean(dd.io.load(glob.glob(savedir+roi+'*')[0], '/'+task+'/ISC_g_time'), axis=1)[0]
fig,ax=plt.subplots(figsize=(10,1))
ax.plot(x,ISCt,'k')		
ax.set_xticks(xhun)
ax.set_xticklabels(xtxt)
ax.set_xlabel('Time [s]')
ax.set_ylabel('group ISC difference')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(figpath+'FLUX_g_diff.png',dpi=300, bbox_inches = "tight")


ax.plot(x,ev_conv*-1,'k',alpha=0.1,label='-1*Hand-Labeled Events')
ax.legend(loc='lower right')	
for e in event_list:
	ax.axvline(e*TR, color='k', linestyle='--',alpha=0.2)
ax.plot(gall,[3]*len(gall),'k*')		
ax.set_xticks(xhun)
ax.set_xticklabels(xtxt)
ax.set_xlabel('Time [s]')
ax.set_ylabel('z-scored group ISC difference')
fig.savefig(figpath+'TPJ_all.png')

fig,ax = plt.subplots()
ISC = np.mean(ISCall,axis=0)
r,p = pearsonr(ISC,ev_conv)
stars = x[np.where(ISC<-1)]
ax.plot(x,ISC,label='Avg ISC')
ax.plot(x,ev_conv*-1,'k',alpha=0.1,label='-1*Hand-Labeled Events')
ax.legend(loc='lower right')
for e in event_list:
	ax.axvline(e*TR, color='k', linestyle='--',alpha=0.2)
ax.plot(stars,[3]*len(stars),'k*')		
ax.set_xticks(xhun)
ax.set_xticklabels(xtxt)
ax.set_xlabel('Time [s]')
ax.set_ylabel('z-scored group ISC difference')
ax.set_title('TPJ vs ev_conv r = '+str(np.round(r,2))+', p = '+str(np.round(p,2)))
fig.savefig(figpath+'TPJ_mean.png')
