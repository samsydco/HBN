#!/usr/bin/env python3

import subprocess as sp
import os
import glob
from settings import *

for fold in glob.glob(tmpdr+'/f*/s*'):
	#if os.path.exists(fmripreppath + 'sub-' + fold.split('_')[-2] + '.html') and not os.path.exists(fmripreppath + 'sub-' + fold.split('_')[-2] + '/log'):
		#print(fold)
		#sp.Popen("rm -r "+fold, shell=True)
	size = sp.check_output(['du','-sh', fold]).split()[0].decode('utf-8')
	if 'G' == size[-1] and any(x in size[0] for x in ['5','6']):
		sp.Popen("rm -r "+fold, shell=True)
	
	

