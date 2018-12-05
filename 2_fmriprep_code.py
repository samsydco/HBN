#!/usr/bin/env python3

import subprocess as sp 
import glob
import csv
import os
from settings import *

# If Missingcsv exists, import info, if not, create it
if os.path.exists(T1file):
	with open(T1file,'r') as f:
		r = csv.reader(f)
		T1dict = {rows[0]:rows[1] for rows in r}
else:
	T1dict = {}

plist = []
for sub in glob.glob('%ssub*'%(path)) :
	sub_temp = sub.replace(path,"")
	if sub_temp not in T1dict:
		# Look at T1's in fsleyes pause, do fmriprep if ok
		sp.run(["fsleyes","%s/anat/%s_acq-HCP_T1w.nii.gz"%(sub,sub_temp)])
		yesno = input("%s: Type \"y\" for \"yes\", \"n\" for \"no\", and \
			\"m\" for \"maybe\"."%(sub_temp))
		T1dict[sub_temp] = yesno
		with open(T1file,'a') as f:
				w = csv.writer(f)
				# TEST THIS!
				w.writerow([sub_temp]+yesno)
	# ADD CONDITIONAL FOR IF .HTML EXISTS and change paricipant parameter
	# Note that label of HTML files might change once running sub's in batches
	if (T1dict[sub_temp] != "n" and
		not os.path.exists(fmripreppath+sub_temp+'.html')):
		plist.append(sub_temp)

# Run participants in batches of 10 - check what is max?:
pchunk = [plist[x:x+10] for x in range(0,len(plist),10)]
for ichunk,chunk in enumerate(pchunk):
	print(str(ichunk) + ' ' + ' '.join(chunk))
	chstr = str(ichunk)
	pstr = ' '.join(chunk)
	sp.run(["screen","-dmSL",chstr,"-Logfile","%sfmriprep_cmdoutput/%s.txt"%(path,chstr),"sh"])
	sp.run(["screen","-S",chstr,"-X","stuff"," \
		sudo fmriprep-docker %s \
		participant \
		--ignore=slicetiming \
		--fs-license-file=/usr/local/freesurfer/license.txt \
		--output-space fsaverage6 --participant_label \
		%s -u 14128:13110 -w /tmp/scohen4/"%(path,outputdr,pstr)])
	sp.run(["screen","-S",chstr,"-X","eval","stuff \015"])
	sp.run(["screen","-S",chstr,"-p","0","-X","stuff","monday1^M"])
	#sp.run(["screen","-S",chstr,"-X","eval","stuff \015"])