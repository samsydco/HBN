#!/usr/bin/env python3

import glob
import pandas as pd
import numpy as np
import os
import subprocess as sp 
import datetime
from settings import *

GTdf = pd.read_csv(T1file, header=None)
compcsv = TRratingdr+'compT1.csv'
if os.path.exists(compcsv):
	compdf = pd.read_csv(compcsv)
else:
    compdf = GTdf.copy()
    compdf.columns = ['sub','GT']

if 'final' not in compdf.columns:
    compdf['final'] = [np.nan for _ in range(len(compdf))] 

for RAcsv in glob.glob(TRratingdr+'*_T1.csv'):
    RA = RAcsv.replace(TRratingdr,"").replace('_T1.csv',"")
    if RA not in compdf.columns:
        compdf[RA] = [np.nan for _ in range(len(compdf))]
    RAdf = pd.read_csv(RAcsv, header=None)
    # check if subject in compdf, add if not
    for sub in RAdf.iloc[:,0]:
        yesno = RAdf[RAdf[0].str.match(sub)][1].iloc[0]
        if not compdf[compdf['sub'].str.match(sub)].empty:
            compdf.loc[np.where(compdf['sub'].str.match(sub).values)[0][0],RA] = yesno
        else:
            compdf = compdf.append(pd.Series(), ignore_index=True)
            compdf.at[compdf.index[-1], 'sub'] = sub
            compdf.at[compdf.index[-1], RA] = yesno        

# look for ratings for which everyone has responded,
# check if there's a corresponding preprocessed file
# go from there...
plist = []
for index, row in compdf.iterrows():
    sub = row['sub']
    if not os.path.exists(fmripreppath+sub+'.html'):
        yn = [str(j) for j in [row[e] for e in compdf.columns if e not in ('sub','final')]]
        if 'nan' not in yn[1:]:
            if 'n' not in yn:
                compdf.loc[index,'final'] = 'y'
            else:
                if yn.count('m')>2:
                    compdf.loc[index,'final'] = 'y'
                if 'y' in yn:
                    print(sub,yn)
                    if str(row['final'])=='nan':
                        sp.run(["fsleyes","%s%s/anat/%s_acq-HCP_T1w.nii.gz"%(path,sub,sub)])
                        yesno = yesnofun(sub)
                        if yesno in ['y','n','m']:
                            compdf.loc[index,'final'] = yesno
                        else:
                            break
# Need to save compdf to csv with 'final' column
compdf.to_csv(compcsv, index=False)


            
    

