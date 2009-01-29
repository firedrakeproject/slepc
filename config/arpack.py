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
import check

def Check(conf,directory,libs):

  if petscconf.SCALAR == 'real':
    if petscconf.PRECISION == 'single':
      functions = ['psnaupd','psneupd','pssaupd','psseupd']
    else:
      functions = ['pdnaupd','pdneupd','pdsaupd','pdseupd']
  else:
    if petscconf.PRECISION == 'single':
      functions = ['pcnaupd','pcneupd']
    else:
      functions = ['pznaupd','pzneupd']

  if libs:
    libs = [libs]
  else:
    libs = [['-lparpack','-larpack'],['-lparpack_MPI','-larpack'],['-lparpack_MPI-LINUX','-larpack_LINUX'],['-lparpack_MPI-SUN4','-larpack_SUN4']]

  if directory:
    dirs = [directory]
  else:
    dirs = check.GenerateGuesses('Arpack')
    
  return check.FortranLib(conf,'ARPACK',dirs,libs,functions)
