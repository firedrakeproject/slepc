import os
import sys

def Load(petscdir):
  global ARCH,DIR,MAKE,SCALAR,PRECISION,MPIUNI
  
  if 'PETSC_ARCH' in os.environ:
    ARCH = os.environ['PETSC_ARCH']
  else:
    try:
      f = open(os.sep.join([petscdir,'bmake','petscconf']))
      ARCH = ''
      for l in f.readlines():
	if l.startswith('PETSC_ARCH='):
	  ARCH = l.split('=')[1].rstrip()
	  f.close()
	  break
      f.close()
    except:
      sys.exit('ERROR: PETSc must be configured first')
    if not petscarch:
      sys.exit('ERROR: please set enviroment variable PETSC_ARCH')

  MPIUNI = 0
  
  try:
    f = open(os.sep.join([petscdir,'bmake',ARCH,'petscconf']))
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
    f.close()
  except:
    sys.exit('ERROR: PETSc is not configured for architecture ' + ARCH)
