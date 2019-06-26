#!/usr/bin/env python3

import subprocess as sp
import os
import glob
from settings import *

fl = []
fl2 = []
for fold in glob.glob(tmpdr+'/f*/s*'):
	if os.path.exists(fmripreppath + 'sub-' + fold.split('_')[-2] + '.html') and not os.path.exists(fmripreppath + 'sub-' + fold.split('_')[-2] + '/log'):
		fl.append(fold)
		#print(fold)
		#sp.Popen("rm -r "+fold, shell=True)
	size = sp.check_output(['du','-sh', fold]).split()[0].decode('utf-8')
	#print(size)
	if 'G' == size[-1] and any(x in size[0] for x in ['5','6']):
		fl2.append(fold)
		#sp.Popen("rm -r "+fold, shell=True)
	
	

