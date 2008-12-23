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

  # LAPACK standard functions
  l = ['laev2','gehrd','lanhs','lange','getri','hseqr','trexc','trevc','geevx','ggevx','gelqf','gesdd']

  # LAPACK functions with different real and complex versions
  if petscconf.SCALAR == 'real':
    l += ['orghr','syevr','sygvd','ormlq']
    if petscconf.PRECISION == 'single':
      prefix = 's'
    else:
      prefix = 'd'
  else:
    l += ['unghr','heevr','hegvd','unmlq','ungtr','hetrd']
    if petscconf.PRECISION == 'single':
      prefix = 'c'
    else:
      prefix = 'z'

  # add prefix to LAPACK names  
  functions = []
  for i in l:
    functions.append(prefix + i)

  # LAPACK functions which are always used in real version 
  if petscconf.PRECISION == 'single':
    functions += ['sstevr','sbdsdc','ssteqr','sorgtr','ssytrd']
  else:
    functions += ['dstevr','dbdsdc','dsteqr','dorgtr','dsytrd']
   
  # check for all functions at once
  all = [] 
  for i in functions:
    f =  '#if defined(PETSC_BLASLAPACK_UNDERSCORE)\n'
    f += i + '_\n'
    f += '#elif defined(PETSC_BLASLAPACK_CAPS) || defined(PETSC_BLASLAPACK_STDCALL)\n'
    f += i.upper() + '\n'
    f += '#else\n'
    f += i + '\n'
    f += '#endif\n'
    all.append(f)

  log.Write('=== Checking all LAPACK functions...')
  if check.Link(all,[],[]):
    return []

  # check functions one by one
  missing = []
  conf.write('SLEPC_MISSING_LAPACK =')
  for i in functions:
    f =  '#if defined(PETSC_BLASLAPACK_UNDERSCORE)\n'
    f += i + '_\n'
    f += '#elif defined(PETSC_BLASLAPACK_CAPS) || defined(PETSC_BLASLAPACK_STDCALL)\n'
    f += i.upper() + '\n'
    f += '#else\n'
    f += i + '\n'
    f += '#endif\n'
  
    log.Write('=== Checking LAPACK '+i+' function...')
    if not check.Link([f],[],[]):
      missing.append(i)
      conf.write(' -DSLEPC_MISSING_LAPACK_' + i[1:].upper())

  conf.write('\n')
  return missing
