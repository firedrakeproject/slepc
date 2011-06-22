#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#     
#  SLEPc is free software: you can redistribute it and/or modify it under  the
#  terms of version 3 of the GNU Lesser General Public License as published by
#  the Free Software Foundation.
#
#  SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY 
#  WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS 
#  FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for 
#  more details.
#
#  You  should have received a copy of the GNU Lesser General  Public  License
#  along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

import os
import sys

def Load(petscdir):
  global ARCH,DIR,MAKE,SCALAR,PRECISION,ISINSTALL,DESTDIR,BFORT,TEST_RUNS,CC,CC_FLAGS,AR,AR_FLAGS,AR_LIB_SUFFIX,RANLIB,IND64
  
  if 'PETSC_ARCH' in os.environ:
    ISINSTALL = 0
    ARCH = os.environ['PETSC_ARCH']
    PETSCVARIABLES = os.sep.join([petscdir,ARCH,'conf','petscvariables'])
  else:
    ISINSTALL = 1
    ARCH = 'arch-installed-petsc'
    PETSCVARIABLES = os.sep.join([petscdir,'conf','petscvariables'])

  try:
    f = open(PETSCVARIABLES)
    for l in f.readlines():
      (k,v) = l.split('=',1)
      k = k.strip()
      v = v.strip()
      if k == 'PETSC_SCALAR':
	SCALAR = v
      elif k == 'PETSC_PRECISION':
        PRECISION = v
      elif k == 'MAKE':
	MAKE = v
      elif k == 'DESTDIR':
        DESTDIR = v
      elif k == 'BFORT':
	BFORT = v
      elif k == 'TEST_RUNS':
	TEST_RUNS = v
      elif k == 'CC':
	CC = v
      elif k == 'CC_FLAGS':
	CC_FLAGS = v
      elif k == 'AR':
	AR = v
      elif k == 'AR_FLAGS':
	AR_FLAGS = v
      elif k == 'AR_LIB_SUFFIX':
	AR_LIB_SUFFIX = v
      elif k == 'RANLIB':
	RANLIB = v
    f.close()
  except:
    sys.exit('ERROR: PETSc is not configured for architecture ' + ARCH)

  PETSCCONF_H = os.sep.join([petscdir,ARCH,'include','petscconf.h'])
  IND64 = 0
  try:
    f = open(PETSCCONF_H)
    for l in f.readlines():
      l = l.split()
      if len(l)==3 and l[0]=='#define' and l[1]=='PETSC_USE_64BIT_INDICES' and l[2]=='1':
	IND64 = 1
    f.close()
  except:
    sys.exit('ERROR: cannot open petscconf.h')

