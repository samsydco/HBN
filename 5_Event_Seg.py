#!/usr/bin/env python3

import h5py
import os
import glob
import numpy as np
from scipy.spatial.distance import squareform
from settings import *

from brainiak import isfc
EventSegf = 'EventSeg.h5'

subord = glob.glob(h5path+'sub*.h5')
subord = subord[0:5] # for testing!
ROIs = ['RSC','A1']
for task in ['DM','TP']:
    ROIdata = {}
    for roi in ROIs:
        ROIdata[roi] = []
    for subidx, sub in enumerate(subord):
        f = h5py.File(sub, 'r')
        for roi in ROIs:
            ROIdata[roi].append(f[task][roi][:])
        ROIdata[roi] = np.mean(np.stack(ROIdata[roi]),axis=0)

