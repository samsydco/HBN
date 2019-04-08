#!/usr/bin/env python3

import subprocess as sp
import glob
from settings import *

for fold in glob.glob(tmpdr+'/f*/s*'):
	size = sp.check_output(['du','-sh', fold]).split()[0].decode('utf-8')
	if 'G' == size[-1] and any(x in size[0] for x in ['5','6']):
		sp.Popen("rm -r "+fold, shell=True)
	

