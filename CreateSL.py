#!/usr/bin/env python3

import numpy as np
import deepdish as dd
import os
import h5py
from settings import *
from tqdm import tqdm

def findcols(X):
    # X is a dictionary with elements ir and jc
    # returns a list of arrays of locations of nonzero entries in each column
    cols = [None] * (len(X['jc'])-1)
    for i in range(len(cols)):
        cols[i] = X['ir'][X['jc'][i]:X['jc'][i+1]]
    return cols

np.random.seed(0)

np.random.shuffle(subord)
SLsize = round(15 / 1.4)  # 20 mm / 1.4 mm spacing
fs6 = dd.io.load(ISCpath+'fsaverage6_adj.h5')
cover_count = 3 # Number of SLs that should overlap each vertex
min_verts = 100 # Min number of valid vertices per SL

SLlist = dict()
count = dict()
for hem in ['left', 'right']:
    print(hem)
    hemi = 'L' if hem == 'left' else 'R'

    adj = findcols(fs6[hem])
    nv = len(adj)
    count[hemi] = np.zeros((nv), dtype=np.int)

    D = dict()
    for sub in subord[:10]:
		for task in ['DM','TP']:
			D[sub+task] = dd.io.load(sub, '/'+task+'/'+hemi)
    
    valid_verts = np.where(np.all(np.stack([np.all(~np.isnan(D[sub]), axis=1) for sub in list(D.keys())], axis=0), axis=0))[0]
    #valid_verts = np.where(np.power(fs6['lhsurfcoords'].T - np.array([-60, 0, 30])[np.newaxis,:], 2).sum(1) < 200)[0]
    
    # Construct all possible SLs (with all valid verts as centers)
    print('Building all SLs...')
    SL_vox = np.zeros((nv, nv), dtype=bool) # SL by vox
    for v in tqdm(valid_verts):
        toExpand = set([v])
        SL = set()
        for steps in range(SLsize):
            newNodes = set()
            for i in toExpand:
                newNodes.update(set(adj[i]))
            SL.update(toExpand)
            newNodes.difference_update(SL)
            newNodes.intersection_update(valid_verts)
            toExpand = newNodes
        SL = np.asarray(list(SL), dtype='uint64')
        if len(SL) >= min_verts:
            SL_vox[v, SL] = True
    
    # Iteratively remove SLs that drop the min voxel cover the least
    print('Pruning SLs...')
    vox_counts = SL_vox.sum(0)
    SL_min = np.zeros(nv)
    for sl in valid_verts:
        v = np.where(SL_vox[sl,:])[0]
        if len(v) > 0:
            SL_min[sl] = np.min(vox_counts[v])

    while True:
        max_min = np.max(SL_min)
        print('   ', max_min, end='\r')
        
        max_SLs = np.where(SL_min == max_min)[0]
        if max_min == cover_count:
            break
        to_remove = np.random.choice(max_SLs)

        # Update anything that may have changed
        vox_removed = np.where(SL_vox[to_remove, :])[0]
        vox_counts[vox_removed] -= 1
        SL_vox[to_remove, :] = False
        SL_min[to_remove] = 0
        for sl in np.where(SL_vox[np.ix_(np.arange(nv), vox_removed)].sum(1))[0]:
            SL_min[sl] = np.min(vox_counts[np.where(SL_vox[sl,:])[0]])

    
    count[hemi] = vox_counts.copy()
    SLlist[hemi] = []
    for SL_i in np.where(SL_vox.sum(1) > 0)[0]:
        SL = np.where(SL_vox[SL_i,:])[0]
        SLlist[hemi].append(SL)
        

dd.io.save(ISCpath+'SLlist.h5', SLlist)
dd.io.save(ISCpath+'SLcount.h5', count)
