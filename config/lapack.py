import os
import sys

import petscconf
import log
import check

def Check(conf):

  functions = ['laev2','gehrd','lanhs','lange','getri','hseqr','trexc','trevc','geevx','ggevx','gelqf']

  if petscconf.SCALAR == 'real':
    functions += ['orghr','syevr','sygvd','ormlq']
    if petscconf.PRECISION == 'single':
      prefix = 's'
    else:
      prefix = 'd'
  else:
    functions += ['unghr','heevr','hegvd','unmlq']
    if petscconf.PRECISION == 'single':
      prefix = 'c'
    else:
      prefix = 'z'
  
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
   
    log.Write('=== Checking LAPACK '+prefix+i+' function...')
    if not check.Link([f],[],[]):
      missing.append(prefix + i)
      conf.write(' -DSLEPC_MISSING_LAPACK_' + i.upper())


  if petscconf.PRECISION == 'single':
    functions = ['slamch','sstevr']
  else:
    functions = ['dlamch','dstevr']

  for i in functions:
    f =  '#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_UNDERSCORE)\n'
    f += i + '_\n'
    f += '#elif defined(PETSC_HAVE_FORTRAN_CAPS)\n'
    f += i.upper() + '\n'
    f += '#else\n'
    f += i + '\n'
    f += '#endif\n'
   
    log.Write('=== Checking LAPACK '+i+' function...')
    if not check.Link([f],[],[]):
      missing.append(i)
      conf.write(' -DSLEPC_MISSING_LAPACK_' + i.upper())
  
  conf.write('\n')
  return missing
