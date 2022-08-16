#!/usr/bin/env python3

# HMM Settings
import os
from ISC_settings import *
from motion_check import outliers
from sklearn.model_selection import KFold
pd.options.mode.chained_assignment = None

subord2 = [s for s in subord if s not in outliers]
agel,pcl,phenol = make_phenol(subord2)
agespan,nbinseq,eqbins,ageeq,lenageeq,minageeq = bin_split(subord2)

roidir = ISCpath+'Yeo_parcellation_outlier_'
seeds = np.char.mod('%d', np.arange(4))
nkdir = HMMpath+'nk_moreshuff_outlier_'
llh5 = HMMpath+'ll_diff_seeds_outlier.h5'
llcsv = HMMpath+'ll_diff_seeds_outlier.csv'
pvals_file = ISCpath+'p_vals_seeds_outlier.h5'
HMMsavedir = HMMpath+'shuff_5bins_train04_outlier_'

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

def nearest_peak(v):
	"""Estimates location of local maximum nearest the origin
    Starting at the origin, we follow the local gradient until reaching a
    local maximum. A quadratic function is then fit to the maximum and its
    two surrounding points, and the peak of this function is used as a
    continuous-valued estimate of the location of the maximum.
    Parameters
    ----------
    v : ndarray
        Array of values from [-max_lag, max_lag] inclusive
    Returns
    -------
    float
        Location of peak of quadratic fit
    """
	lag = (len(v)-1)//2
	# Find local maximum
	while 2 <= lag <= (len(v) - 3):
		win = v[(lag-1):(lag+2)]
		if (win[1] > win[0]) and (win[1] > win[2]):
			break
		if win[0] > win[2]:
			lag -= 1
		else:
			lag += 1
	# Quadratic fit
	x = [lag-1, lag, lag+1]
	y = v[(lag-1):(lag+2)]
	denom = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
	A = (x[2] * (y[1] - y[0]) + x[1] * \
			 (y[0] - y[2]) + x[0] * (y[2] - y[1])) / denom
	B = (x[2]*x[2] * (y[0] - y[1]) + x[1]*x[1] * (y[2] - y[0]) + \
	x[0]*x[0] * (y[1] - y[2])) / denom
	max_x = (-B / (2*A))
	return min(max(max_x, 0), len(v)-1)



if os.path.exists(llcsv):
	df = pd.read_csv(llcsv, index_col=0)
	df=df[((df['0_2k_diff']>ll_thresh) | (df['4_2k_diff']>ll_thresh))]
	ROIl = list(df.index)

nsub= 40
y = [0]*int(np.floor(nsub/nsplit))*4+[1]*(int(np.floor(nsub/nsplit)))
kf = KFold(n_splits=nsplit, shuffle=True, random_state=2)



