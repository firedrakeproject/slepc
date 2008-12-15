#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#     SLEPc - Scalable Library for Eigenvalue Problem Computations
#     Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain
#
#     This file is part of SLEPc. See the README file for conditions of use
#     and additional information.
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

import os
import sys

def Load(petscdir):
  global VERSION,RELEASE
  try:
    f = open(os.sep.join([petscdir,'include','petscversion.h']))
    for l in f.readlines():
      l = l.split()
      if len(l) == 3:
        if l[1] == 'PETSC_VERSION_RELEASE':
	  RELEASE = l[2]
	if l[1] == 'PETSC_VERSION_MAJOR':
          major = l[2]
	elif l[1] == 'PETSC_VERSION_MINOR':
          minor = l[2]
	elif l[1] == 'PETSC_VERSION_SUBMINOR':
          subminor = l[2]
    f.close()
    VERSION = major + '.' + minor + '.' + subminor
  except:
    sys.exit('ERROR: file error while reading PETSC version')
