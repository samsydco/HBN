#!/usr/bin/env python3

import subprocess as sp 
import glob
import re
import os
import pandas as pd
import numpy as np
import datetime
from settings import *

site = 'Site-CBIC'
#site = 'Site-RU'

'''
compcsv = TRratingdr+'compT1_'+site+'.csv'
compdf = pd.read_csv(compcsv)
fl = [item for sublist in [f.split('.txt')[0].split('_sub-NDAR')[1:] for f in glob.glob(path+'fmriprep_cmdoutput/'+'*sub*.txt')] for item in sublist]
plist = []
for i,v in {i:fl.count(i) for i in fl}.items():
    if v > 2:
        lines = []
        for file in glob.iglob(path+'sub-NDAR'+i+'/fmap/*fMRI*json'):
            for line in open(file, 'r'):
                if re.search('PhaseEncodingDirection"', line):
                    lines.append(line)
        f = outputdr+'fmriprep/'+'sub-NDAR'+i+'.html'
        if not os.path.isfile(f) and all(i in ''.join(lines) for i in ['"j"','"j-"']):
            plist.append('sub-NDAR'+i)
'''
plist = ['sub-NDARXZ902NFM']

password = input('Type the password for scohen@sophon.columbia.edu:')
# Run participants in batches of nchunk - check what is max?:
nchunk = 4 # number of items per chunk (maybe use 10?)
pchunk = [plist[x:x+nchunk] for x in range(0,len(plist),nchunk)]

# Delete extra BOLD stuff
for chunk in pchunk:
	pstr = ' '.join(chunk)
	for task in ['TP']:#,'TP']:
		date = str(datetime.datetime.now())[0:19].replace(' ','_')
		f = open("%sfmriprep_cmdoutput/%s_%s_%s.txt"%(path,task,date,pstr.replace(" ","_")), "w")
		# "-t" in docker instead of "-it" :)
		if site =='Site-RU':
			command = ('docker run --rm -t -u 14128:13110 -v                         /usr/local/freesurfer/license.txt:/opt/freesurfer/license.txt:ro -v '+path+':/data:ro -v '+outputdr+':/out -v '+tmpdr+':/scratch poldracklab/fmriprep:1.1.4 /data /out participant --ignore=slicetiming --output-space fsaverage6 --participant_label '+pstr+' -t movie'+task+' -w /scratch').split()
		else:
			command = ('docker run --rm -t -u 14128:13110 -v                         /usr/local/freesurfer/license.txt:/opt/freesurfer/license.txt:ro -v '+path+site+'/:/data:ro -v '+outputdr+':/out -v '+tmpdr+':/scratch poldracklab/fmriprep:1.1.4 /data /out participant --ignore=slicetiming --output-space fsaverage6 --participant_label '+pstr+' -t movie'+task+' -w /scratch').split()
		p = sp.Popen(['sudo', '-S'] + command, stdin=sp.PIPE, stderr=sp.PIPE,
          universal_newlines=True,stdout=f)
		p.communicate(password + '\n')[1]

'''

'docker run --rm -t -u 14128:13110 -v                         /usr/local/freesurfer/license.txt:/opt/freesurfer/license.txt:ro -v '+path+':/data:ro -v '+outputdr+':/out -v /tmp/scohen4:/scratch poldracklab/fmriprep:1.1.4 /data /out participant --ignore=slicetiming --output-space fsaverage6 --participant_label '+pstr+' -t movieDM -t movieTP -w /scratch'

# Full thing
chunk = pchunk[2]
pstr = ' '.join(chunk)
date = str(datetime.datetime.now())[0:19].replace(' ','_')
f = open("%sfmriprep_cmdoutput/FULL_THING_%s.txt"%(path,date), "w")
# "-t" in docker instead of "-it" :)
command = ('docker run --rm -t -u 14128:13110 -v /usr/local/freesurfer/license.txt:/opt/freesurfer/license.txt:ro -v '+path+':/data:ro -v '+outputdr+':/out -v /tmp/scohen4:/scratch poldracklab/fmriprep:1.1.4 /data /out participant --ignore=slicetiming --output-space fsaverage6 --participant_label '+pstr+' -w /scratch').split()
p = sp.Popen(['sudo', '-S'] + command, stdin=sp.PIPE, stderr=sp.PIPE,
          universal_newlines=True,stdout=f)
sudo_prompt = p.communicate(password + '\n')[1]

# Full thing without fieldmaps
chunk = pchunk[3]
pstr = ' '.join(chunk)
date = str(datetime.datetime.now())[0:19].replace(' ','_')
f = open("%sfmriprep_cmdoutput/FULL_THING_NO_FMAP_%s.txt"%(path,date), "w")
# "-t" in docker instead of "-it" :)
command = ('docker run --rm -t -u 14128:13110 -v /usr/local/freesurfer/license.txt:/opt/freesurfer/license.txt:ro -v '+path+':/data:ro -v '+outputdr+':/out -v /tmp/scohen4:/scratch poldracklab/fmriprep:1.1.4 /data /out participant --ignore=slicetiming --ignore=fieldmaps --output-space fsaverage6 --participant_label '+pstr+' -w /scratch').split()
p = sp.Popen(['sudo', '-S'] + command, stdin=sp.PIPE, stderr=sp.PIPE,
          universal_newlines=True,stdout=f)
sudo_prompt = p.communicate(password + '\n')[1]

for ichunk,chunk in enumerate(pchunk):
	print(str(ichunk) + ' ' + ' '.join(chunk))
	chstr = str(ichunk)
	pstr = ' '.join(chunk)
	sp.run(["screen","-dmSL",chstr,"-Logfile","%sfmriprep_cmdoutput/%s.txt"%(path,chstr),"sh"])
	sp.run(["screen","-S",chstr,"-X","stuff"," \
		sudo fmriprep-docker %s %s \
		participant \
		--ignore=slicetiming \
		--fs-license-file=/usr/local/freesurfer/license.txt \
		--output-space fsaverage6 --participant_label \
		%s -u 14128:13110 -w /tmp/scohen4/"%(path,outputdr,pstr)])
	sp.run(["screen","-S",chstr,"-X","eval","stuff \015"])
	sp.run(["screen","-S",chstr,"-p","0","-X","stuff","monday1^M"])
	#sp.run(["screen","-S",chstr,"-X","eval","stuff \015"])
'''
    

