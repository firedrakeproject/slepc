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

import petscconf
import log
import check

def Check(conf):
  log.Write('='*80)
  log.Println('Checking LAPACK library...')

  functions = ['laev2','gehrd','lanhs','lange','getri','hseqr','trexc','trevc','geevx','ggevx','gelqf','gesdd']

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


  functions = ['stevr','bdsdc']
  if petscconf.PRECISION == 'single':
    prefix = 's'
  else:
    prefix = 'd'

  for i in functions:
    f =  '#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_UNDERSCORE)\n'
    f += prefix + i + '_\n'
    f += '#elif defined(PETSC_HAVE_FORTRAN_CAPS)\n'
    f += prefix.upper() + i.upper() + '\n'
    f += '#else\n'
    f += prefix + i + '\n'
    f += '#endif\n'
   
    log.Write('=== Checking LAPACK '+i+' function...')
    if not check.Link([f],[],[]):
      missing.append(i)
      conf.write(' -DSLEPC_MISSING_LAPACK_' + i.upper())
  
  conf.write('\n')
  return missing
