#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2009, Universidad Politecnica de Valencia, Spain
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
  global ARCH,DIR,MAKE,SCALAR,PRECISION,ISINSTALL,INSTALL_DIR
  
  if 'PETSC_ARCH' in os.environ:
    ISINSTALL = 0
    ARCH = os.environ['PETSC_ARCH']
    PETSCVARIABLES = os.sep.join([petscdir,ARCH,'conf','petscvariables'])
  else:
    ISINSTALL = 1
    ARCH = 'unknown'
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
      elif k == 'INSTALL_DIR':
        INSTALL_DIR = v
      elif k == 'PETSC_ARCH_NAME':
        ARCH = v
    f.close()
  except:
    sys.exit('ERROR: PETSc is not configured for architecture ' + ARCH)

  if ISINSTALL and ARCH == 'unknown':
    sys.exit('ERROR: PETSc architecture name is not defined')
    
