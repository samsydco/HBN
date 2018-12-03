#!/usr/bin/env python3

import subprocess as sp 
import glob
import csv
import os

path = '/data/HBN/test2/'
codedr = 'fmriprep_code/'
subs = set(glob.glob('%ssub*'%(path))) #\
    # - set(glob.glob('%ssub*DMU'%(path))) # already running
T1file = path + codedr + 'T1.csv'
if os.path.exists(T1file):
	with open(T1file,'r') as f:
		r = csv.reader(f)
		T1dict = {rows[0]:rows[1] for rows in r}
else:
	T1dict = {}

for sub in subs:
	sub_temp = sub.replace(path,"")
	if sub_temp not in T1dict:
		# Look at T1's in fsleyes pause, do fmriprep if ok
		sp.run(["fsleyes","%s/anat/%s_acq-HCP_T1w.nii.gz"%(sub,sub_temp)])
		yesno = input("%s: Type \"y\" for \"yes\", \"n\" for \"no\", and \
			\"m\" for \"maybe\"."%(sub_temp))
		T1dict[sub_temp] = yesno
	
	if T1dict[sub_temp] != "n":
		sp.run(["screen","-dmSL",sub_temp,"-Logfile","%sfmriprep_cmdoutput/%s.txt"%(path,sub_temp),"sh"])
		# for testing
		#sp.run(["screen","-S",sub_temp,"-X","stuff","echo %s"%(sub_temp)])
		sp.run(["screen","-S",sub_temp,"-X","stuff"," \
			sudo fmriprep-docker %s \
			%sfmriprep_output/ participant \
			--ignore=slicetiming \
			--fs-license-file=/usr/local/freesurfer/license.txt \
			--output-space fsaverage6 --participant_label \
			%s -u 14128:13110 -w /tmp/scohen4/"%(path,path,sub_temp)])
		sp.run(["screen","-S",sub_temp,"-X","eval","stuff \015"])
		sp.run(["screen","-S",sub_temp,"-p","0","-X","stuff","monday1^M"])
		#sp.run(["screen","-S",sub_temp,"-X","eval","stuff \015"])
	

#T1dict = {'raymond':'red', 'rachel':'blue', 'matthew':'green'}
with open(T1file,'w') as f:
	w = csv.writer(f)
	w.writerows(T1dict.items())