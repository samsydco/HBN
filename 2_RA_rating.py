#!/usr/bin/env python3

import subprocess as sp 
import glob
import csv
import os
from settings import *

initials = input("Please type your initials - should be same two letters every time you come to lab.")
T1file = path+codedr+'T1_rating/'+initials+"_T1.csv"
# If T1file exists, import info, if not, create it
if os.path.exists(T1file):
	with open(T1file,'r') as f:
		r = csv.reader(f)
		T1dict = {rows[0]:rows[1] for rows in r}
else:
	T1dict = {}

for sub in glob.glob('%ssub*'%(path)):
	sub_temp = sub.replace(path,"")
	if sub_temp not in T1dict:
		# Look at T1's in fsleyes pause, do fmriprep if ok
		sp.run(["fsleyes","%s/anat/%s_acq-HCP_T1w.nii.gz"%(sub,sub_temp)])
		while True:
			try:
				yesno = input("%s: Type \"y\" for \"yes\", \"n\" for \"no\", and \
					\"m\" for \"maybe\". (Type \"break\" if you need a break)."%(sub_temp))
			except ValueError:
				print("Sorry, I didn't understand that.")
				continue

			if yesno not in ['y','n','m','break']:
				print("Sorry, your response must be \"y\", \"n\", or \"m\".")
				continue
			elif yesno=="break":
				break
			else:
				#Answer is good.
				break
		if yesno in ['y','n','m']:
			T1dict[sub_temp] = yesno
		else:
			break
		with open(T1file,'a') as f:
				w = csv.writer(f)
				# TEST THIS!
				w.writerow([sub_temp,yesno])





