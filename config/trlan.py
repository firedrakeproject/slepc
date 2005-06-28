import os
import sys

import petscconf
import check

def Check(conf,directory,libs):

  if petscconf.SCALAR == 'complex':
    sys.exit('ERROR: TRLAN does not support complex numbers.') 

  if petscconf.PRECISION == 'single':
    sys.exit('ERROR: TRLAN does not support single precision.') 

  functions = ['trlan77']
  if libs:
    libs = [libs]
  else:
    if petscconf.MPIUNI:
      libs = [['-ltrlan']]
    else:
      libs = [['-ltrlan_mpi']]

  if directory:
    dirs = [directory]
  else:
    dirs = check.GenerateGuesses('TRLan')
    
  return check.FortranLib(conf,'TRLAN',dirs,libs,functions)
