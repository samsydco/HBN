#!/usr/bin/env python3

# HMM Settings
import os
from ISC_settings import *
from motion_check import outliers
from event_comp import ev_conv, Pro_ev_conv, child_ev_conv
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
pd.options.mode.chained_assignment = None

subord2 = [s for s in subord if s not in outliers]
agel,pcl,phenol = make_phenol(subord2)
agespan,nbinseq,eqbins,ageeq,lenageeq,minageeq = bin_split(subord2)

roidir = ISCpath+'Yeo_parcellation_outlier_'
seeds = [f[-1] for f in glob.glob(roidir+'*')]
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

def lag_pearsonr(x, y, max_lags):
    """Compute lag correlation between x and y, up to max_lags
    Parameters
    ----------
    x : ndarray
        First array of values
    y : ndarray
        Second array of values
    max_lags: int
        Largest lag (must be less than half the length of shortest array)
    Returns
    -------
    ndarray
        Array of 1 + 2*max_lags lag correlations, for x left shifted by
        max_lags to x right shifted by max_lags
    """

    assert max_lags < min(len(x), len(y)) / 2, \
        "max_lags exceeds half the length of shortest array"

    assert len(x) == len(y), "array lengths are not equal"

    lag_corrs = np.full(1 + (max_lags * 2), np.nan)

    for i in range(max_lags + 1):

        # add correlations where x is shifted to the right
        lag_corrs[max_lags + i] = pearsonr(x[:len(x) - i], y[i:len(y)])[0]

        # add correlations where x is shifted to the left
        lag_corrs[max_lags - i] = pearsonr(x[i:len(x)], y[:len(y) - i])[0]

    return lag_corrs

if os.path.exists(llcsv):
	df = pd.read_csv(llcsv, index_col=0)
	df=df[((df['0_2k_diff']>ll_thresh) | (df['4_2k_diff']>ll_thresh))]
	ROIl = list(df.index)

nsub= 40
y = [0]*int(np.floor(nsub/nsplit))*4+[1]*(int(np.floor(nsub/nsplit)))
kf = KFold(n_splits=nsplit, shuffle=True, random_state=2)



