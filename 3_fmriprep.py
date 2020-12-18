#!/usr/bin/env python3

# Documentation for fmriprep version 1.1.4:
# https://fmriprep.org/en/1.1.4/usage.html

import subprocess as sp 
import glob
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import datetime
from settings import *
	
#site = 'Site-CBIC'
site = 'Site-RU'

compcsv = TRratingdr+'compT1_'+site+'.csv'
compdf = pd.read_csv(compcsv)

plist = []
for index, row in compdf.iterrows():
	if ((str(row['final']) != "n" and str(row['final'])!='nan') and
		not os.path.exists(fmripreppath+row['sub']+'.html')):
		#len(glob.glob(fmripreppath+sub_temp+'/figures/*sdc*'))!=2):
		plist.append(row['sub'])
plist = plist[:75]
nchunk = 4 # number of items per chunk (maybe use 10?)
pchunk = [plist[x:x+nchunk] for x in range(0,len(plist),nchunk)]
	
if __name__ == "__main__":
	# stuff only to run when not called via 'import' here

	password = input('Type the password for scohen@sophon.columbia.edu:')
	# Run participants in batches of nchunk - check what is max?:
	# Delete extra BOLD stuff
	for chunk in tqdm(pchunk):
		pstr = ' '.join(chunk)
		for task in ['DM']:
			date = str(datetime.datetime.now())[0:19].replace(' ','_')
			f = open("%sfmriprep_cmdoutput/%s_%s_%s.txt"%(path,task,date,pstr.replace(" ","_")), "w")
			# "-t" in docker instead of "-it" :)
			if site =='Site-RU':
				command = ('docker run --rm -t -u 14128:13110 -v                         /usr/local/freesurfer/license.txt:/opt/freesurfer/license.txt:ro -v '+path+site+':/data:ro -v '+outputdr+':/out -v '+tmpdr+':/scratch poldracklab/fmriprep:1.1.4 /data /out participant --ignore slicetiming fieldmaps --use-syn-sdc --fs-no-reconall --output-space template  --participant_label '+pstr+' -t movie'+task+' -w /scratch').split()
			else:
				command = ('docker run --rm -t -u 14128:13110 -v                         /usr/local/freesurfer/license.txt:/opt/freesurfer/license.txt:ro -v '+path+site+'/:/data:ro -v '+outputdr+':/out -v '+tmpdr+':/scratch poldracklab/fmriprep:1.1.4 /data /out participant --ignore=slicetiming --output-space fsaverage6 --participant_label '+pstr+' -t movie'+task+' -w /scratch').split()
			p = sp.Popen(['sudo', '-S'] + command, stdin=sp.PIPE, stderr=sp.PIPE,
			  universal_newlines=True,stdout=f)
			p.communicate(password + '\n')[1]
    

