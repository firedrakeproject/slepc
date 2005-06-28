import os
import sys

import petscconf
import check

def Check(conf,directory,libs):

  if petscconf.SCALAR == 'complex':
    sys.exit('ERROR: PLANSO does not support complex numbers.') 

  if petscconf.PRECISION == 'single':
    sys.exit('ERROR: PLANSO does not support single precision.') 

  functions = ['plandr2']
  if libs:
    libs = [libs]
  else:
    libs = [['-lplan','-llanso']]

  if directory:
    dirs = [directory]
  else:
    dirs = check.GenerateGuesses('Plan')

  return check.FortranLib(conf,'PLANSO',dirs,libs,functions,['op','opm'])
