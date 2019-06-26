#!/usr/bin/env python3

import glob
import nibabel as nib
from moviepy.editor import *
import tqdm
import numpy as np
import deepdish as dd
from scipy.stats import zscore
from datetime import date
import h5py
from settings import *

# Hemodynamic Response Function (from AFNI)
dt = np.arange(0, 15)
p = 8.6
q = 0.547
hrf = np.power(dt / (p * q), p) * np.exp(p - dt / q)

# Don't use the very beginning of the stimulus to compute correlations, to
# ensure that the story has actually started being spoken
ignore_TRs = 10

# save fMRI data in h5 file:
dataf = ISCpath+'ISC_data_'+str(date.today())+'.h5'
if os.path.exists(ISCpath+dataf):
    os.remove(ISCpath+dataf)

subs = glob.glob(prepath+'sub*.h5')
videonames = ['descme_10min_frame_samecodec.mp4','the_present_nocredits.mp4']
group_corr = []
for idx,task in enumerate(['DM','TP']):
	audio_file = videopath + videonames[idx]
	
	# Get metadata about fMRI data
	nii_template = nib.load(path+'/sub-NDARAH948UF0/func/sub-NDARAH948UF0_task-movie'+task+'_bold.nii.gz')
	TR = nii_template.header['pixdim'][4]
	nii_shape = nii_template.header.get_data_shape()
	nTR = nii_shape[3]

	# Load audio, and calculate audio envelope regressor
	print('Loading audio for %s...'%(task))
	clip = AudioFileClip(audio_file)
	Fs = clip.fps
	samples = clip.to_soundarray()
	samples = np.mean(samples, axis=1)
	T = np.floor(samples.shape[0] / Fs).astype(int)
	rms = np.zeros(T)
	for t in range(T):
		rms[t] = np.sqrt(np.mean(np.power(samples[(Fs * t):(Fs * (t + 1))], 2)))
		rms_conv = np.convolve(rms, hrf)[:T]
	rms_TRs = zscore(np.interp(np.linspace(0, (nTR - 1) * TR, nTR),
                           np.arange(0, T), rms_conv)[ignore_TRs:], ddof=1)

    # Compute correlation between rms_TRs and each voxel timecourse
	print('Calculating correlations for %s...'%(task))
	n,n_time = dd.io.load(subord[0],['/'+task+'/L'])[0].shape
	group = np.empty((n*2,n_time),dtype='float16')
	groupn = np.ones((n*2,n_time),dtype='int')*len(subord)
	for s, sub in tqdm.tqdm(enumerate(subord)):
		D = np.concatenate([dd.io.load(sub,['/'+task+'/L'])[0], dd.io.load(sub,['/'+task+'/R'])[0]], axis=0)
		group = np.nansum(np.stack((group,D)),axis=0)
		nanverts = np.argwhere(np.isnan(D))
		groupn[nanverts[:, 0],nanverts[:,1]] = groupn[nanverts[:,0],nanverts[:,1]]-1
	group = group/groupn
	
	# Compute correlations for group average timecourses
	group_corr.append(np.matmul(rms_TRs,zscore(group[:, ignore_TRs:],axis=1, ddof=1).T) / (len(rms_TRs) - 1))

A1corr = np.mean(np.stack(group_corr),axis=0)

with h5py.File(ISCpath+'A1.h5') as hf:
    hf.create_dataset('A1corr', data=A1corr)

# moved code to 5_ROI.y: 4/19/19
'''
f = h5py.File(ISCpath+'A1.h5')
A1roi={}
ROI = np.nan_to_num(f['A1corr'][:])>0.26
A1roi['L'] = ROI[:len(ROI)//2]
A1roi['R'] = ROI[len(ROI)//2:]
for sub in subs:
    with h5py.File(sub) as f:
        for task in ['DM','TP']:
            del f[task]['A1']
            data = []
            for hemi in ['L','R']:
                data.append(f[task][hemi][A1roi[hemi],:])
            f[task].create_dataset('A1', data=np.concatenate(data))
'''
            #f[task].create_dataset('A1', data=zscore(np.mean(np.concatenate(data),axis=0),ddof=1))