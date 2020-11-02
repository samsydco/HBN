#!/usr/bin/env python3

# plot Yeo ROIs that correspond with TPJ
# determine what is going on at times with low g_diff

import tqdm
import matplotlib.pyplot as plt
from HMM_settings import *
from scipy.fftpack import fft,ifft
from event_ratings import counts,hrf
from scipy.stats import pearsonr
ev_conv_perm=ev_conv

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

# shuffle phase:
def phase_shuff(signal):
	# Modified from brainiak.utils.utils.phase_randomize
	prng = np.random.RandomState(None)
	# for signals with odd number of time points only:
	pos_freq = np.arange(1, (signal.shape[0] - 1) // 2 + 1)
	neg_freq = np.arange(signal.shape[0] - 1, signal.shape[0] // 2, -1)
	phase_shifts = (prng.rand(len(pos_freq)) * 2 * np.math.pi)
	fft_data = fft(signal)
	# Shift pos and neg frequencies symmetrically, to keep signal real
	fft_data[pos_freq] *= np.exp(1j * phase_shifts)
	fft_data[neg_freq] *= np.exp(-1j * phase_shifts)
	# Inverse FFT to put data back in time domain
	signal_shuff = np.real(ifft(fft_data))
	return signal_shuff

ROIs = [r.split('/')[-1][:-3] for r in glob.glob(savedir+'*')]
TPJ_ROIs = ['RH_DefaultA_IPL_1', 'LH_DefaultB_IPL_1', 'LH_SalVentAttnA_ParOper_1', 'RH_TempPar_3']#'RH_SalVentAttnA_ParOper_1', 
colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']

#fig,ax = plt.subplots()
TW = 10 # For circular time shuffle
testdict = {key:{key:{key:{key:[] for key in ['rperm','rs','ps']} for key in ['ISC','zscore']} for key in ROIs} for key in ['permcount','circ','phase']}
for ri,roi in tqdm.tqdm(enumerate(ROIs)):
	ISC = np.nanmean(dd.io.load(glob.glob(savedir+roi+'*')[0], '/'+task+'/ISC_g_time'), axis=1)
	zscoreISC = (ISC[0]-np.nanmean(ISC[1:], axis=0))/np.nanstd(ISC[1:], axis=0)
	for test in testdict.keys():
		ev_conv=ev_conv_perm
		nPerm=2000 if test!='circ' else len(ev_conv)-TW*2
		for p in range(nPerm+1):
			rperm,_ = pearsonr(zscoreISC,ev_conv)
			testdict[test][roi]['zscore']['rperm'].append(rperm)
			rperm,_ = pearsonr(ISC[0],ev_conv)
			testdict[test][roi]['ISC']['rperm'].append(rperm)
			if test=='permcount': 
				counts = np.random.permutation(counts)
				ev_conv = np.convolve(counts,hrf,'same')
			if test=='circ': ev_conv = np.concatenate((ev_conv[p+TW:],ev_conv[:p+TW]))
			if test == 'phase': ev_conv = phase_shuff(ev_conv)
		for comp in ['ISC','zscore']:
			testdict[test][roi][comp]['rs'] = testdict[test][roi][comp]['rperm'][0]
			testdict[test][roi][comp]['ps'] = np.sum(abs(testdict[test][roi][comp]['rperm'][0])<abs(np.array(testdict[test][roi][comp]['rperm'][1:])))/nPerm
		#if testdict[test][roi]['ps'] < 0.05:
		#	print(test,roi,np.round(testdict[test][roi]['rs'],2),\
		#		  np.round(testdict[test][roi]['ps'],2))
	
dd.io.save(savef,testdict)

# For FLUX poster:
for roi in ['LH_DorsAttnB_PostC_3','RH_DefaultA_IPL_1']:
	ISC = np.nanmean(dd.io.load(glob.glob(savedir+roi+'*')[0], '/'+task+'/ISC_g_time'), axis=1)
	zscoreISC = (ISC[0]-np.nanmean(ISC[1:], axis=0))/np.nanstd(ISC[1:], axis=0)
	ISCt = np.nanmean(dd.io.load(glob.glob(savedir+roi+'*')[0], '/'+task+'/ISC_g_time'), axis=1)[0]
	fig,ax=plt.subplots(figsize=(10,1))
	#ax.plot(x,zscoreISC,'k')
	ax.plot(x,ISCt,'k')	
	#ax2 = ax.twinx()
	#ax2.plot(x,ev_conv,'k--',alpha=0.5)
	#ax2.set_ylabel('Event Confidence')
	ax.set_xticks(xhun)
	ax.set_xticklabels(xtxt)
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('group ISC difference')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	plt.tight_layout()
	fig.savefig(figpath+roi+'_g_diff.png',dpi=300, bbox_inches = "tight")

'''
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
'''

