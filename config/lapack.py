import os
import sys

from check import *

def checkLapack(conf,scalar,precision):

  functions = ['laev2','gehrd','lanhs','lange','getri','hseqr','trexc','trevc','steqr','geevx','ggevx']

  if scalar == 'real':
    functions += ['orghr','syevr','sygvd']
    if precision == 'double':
      prefix = 'd'
    else:
      prefix = 's'
  else:
    functions += ['unghr','heevr','hegvd']
    if precision == 'double':
      prefix = 'z'
    else:
      prefix = 'c'
  
  missing = []
  conf.write('SLEPC_MISSING_LAPACK =')
  for i in functions:
    f =  '#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_UNDERSCORE)\n'
    f += prefix + i + '_\n'
    f += '#elif defined(PETSC_HAVE_FORTRAN_CAPS)\n'
    f += prefix.upper() + i.upper() + '\n'
    f += '#else\n'
    f += prefix + i + '\n'
    f += '#endif\n'
   
    if not checkLink([f],[],[]):
      missing.append(prefix + i)
      conf.write(' -DSLEPC_MISSING_LAPACK_' + i.upper())
  conf.write('\n')
  return missing