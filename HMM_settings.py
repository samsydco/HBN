#!/usr/bin/env python3

# HMM Settings
import os
from ISC_settings import *
from motion_check import outliers
from event_ratings import ev_conv
from sklearn.model_selection import KFold
pd.options.mode.chained_assignment = None

subord2 = [s for s in subord if s not in outliers]
agel,pcl,phenol = make_phenol(subord2)
agespan,nbinseq,eqbins,ageeq,lenageeq,minageeq = bin_split(subord2)


roidir = ISCpath+'Yeo_parcellation_outlier_'
seeds = [f[-1] for f in glob.glob(roidir+'*')]
nkdir = HMMpath+'nk_moreshuff_outlier_'
nkh5 = HMMpath+'nk_outlier_'
llh5 = HMMpath+'ll_diff_seeds_outlier.h5'
llcsv = HMMpath+'ll_diff_seeds_outlier.csv'

tasks = ['DM','TP']
TR=0.8
nTR=[750,250]

TR1 = 12 #12 sec
TR2 = 300 #300 sec (5 min)
k_list = np.unique(np.round((10*60)/np.arange(TR1,TR2,TR1))).astype(int)
nsplit = 5
bins = [0,nbinseq-1]
nshuff = 100
ll_thresh = 0.002

def FDR_p(pvals):
    # Written by Chris Baldassano (git: cbaldassano), given permission to adapt into my code on 04/18/2019 #
    # Port of AFNI mri_fdrize.c

    # Ensure p values are valid, and not exactly equal to 0 or 1
    assert np.all(pvals >= 0) and np.all(pvals <= 1)
    pvals[pvals < np.finfo(np.float_).eps] = np.finfo(np.float_).eps
    pvals[pvals == 1] = 1-np.finfo(np.float_).eps
    n = pvals.shape[0]

    # Compute q using step-down procedure
    qvals = np.zeros((n))
    sorted_ind = np.argsort(pvals)
    sorted_pvals = pvals[sorted_ind]
    qmin = 1.0
    for i in range(n-1, -1, -1):
        qval = (n * sorted_pvals[i])/(i+1)
        if qval > qmin:
            qval = qmin
        else:
            qmin = qval
        qvals[sorted_ind[i]] = qval

    # Estimate number of true positives m1 and adjust q
    if n >= 233:
        phist = np.histogram(pvals, bins=20, range=(0, 1))[0]
        sorted_phist = np.sort(phist[3:19])
        if np.sum(sorted_phist) >= 160:
            median4 = n - 20*np.dot(np.array([1, 2, 2, 1]),
                                    sorted_phist[6:10])/6
            median6 = n - 20*np.dot(np.array([1, 2, 2, 2, 2, 1]),
                                    sorted_phist[5:11])/10
            m1 = min(median4, median6)

            qfac = (n - m1)/n
            if qfac < 0.5:
                qfac = 0.25 + qfac**2
            qvals *= qfac

    return qvals


if os.path.exists(llcsv):
	df = pd.read_csv(llcsv, index_col=0)
	df=df[((df['0_2k_diff']>ll_thresh) | (df['4_2k_diff']>ll_thresh))]
	ROIl = list(df.index)

nsub= 40
y = [0]*int(np.floor(nsub/nsplit))*4+[1]*(int(np.floor(nsub/nsplit)))
kf = KFold(n_splits=nsplit, shuffle=True, random_state=2)



