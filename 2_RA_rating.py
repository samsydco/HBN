#!/usr/bin/env python3

import subprocess as sp 
import glob
import csv
import os
from settings import *

# Right now all RAs should have already entered their information:
while True:
	initials = input("Please type your initials - should be same two letters every time you come to lab.")
	T1file = path+codedr+'T1_rating/'+initials+"_T1.csv"
	if not os.path.exists(T1file):
		print("You used different initials last time!")
		continue
	else:
		break
# If T1file exists, import info, if not, create it
if os.path.exists(T1file):
	with open(T1file,'r') as f:
		r = csv.reader(f)
		T1dict = {rows[0]:rows[1] for rows in r}
else:
	T1dict = {}

subs = glob.glob('%ssub*'%(path))
print('You have %s scans left.'%(len(subs) - len(T1dict)))
for sub in subs:
	sub_temp = sub.replace(path,"")
	if sub_temp not in T1dict:
		# Look at T1's in fsleyes pause, do fmriprep if ok
		sp.run(["fsleyes","%s/anat/%s_acq-HCP_T1w.nii.gz"%(sub,sub_temp)])
		yesno = yesnofun(sub)
		
		if yesno in ['y','n','m']:
			T1dict[sub_temp] = yesno
		else:
			break
		with open(T1file,'a') as f:
				w = csv.writer(f)
				# TEST THIS!
				w.writerow([sub_temp,yesno])





