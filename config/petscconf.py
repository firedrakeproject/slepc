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
  global ARCH,DIR,MAKE,SCALAR,PRECISION,MPIUNI,ISINSTALL,INSTALL_DIR
  
  if 'PETSC_ARCH' in os.environ:
    ISINSTALL = 0
    ARCH = os.environ['PETSC_ARCH']
    PETSCVARIABLES = os.sep.join([petscdir,ARCH,'conf','petscvariables'])
  else:
    ISINSTALL = 1
    ARCH = 'unknown'
    PETSCVARIABLES = os.sep.join([petscdir,'conf','petscvariables'])

  MPIUNI = 0
  
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
      elif k == 'MPI_INCLUDE' and v.endswith('mpiuni'):
        MPIUNI = 1
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
    
