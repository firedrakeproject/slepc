import os
import sys

from check import *

def checkTrlan(conf,directory,libs,scalar,precision,uniprocessor):

  if scalar == 'complex':
    sys.exit('ERROR: TRLAN does not support complex numbers.') 

  if precision == 'single':
    sys.exit('ERROR: TRLAN does not support single precision.') 

  functions = ['trlan77']
  if uniprocessor:
    if libs:
      libs = [libs]
    else:
      libs = [['-ltrlan']]
  else:
    if libs:
      libs = [libs]
    else:
      libs = [['-ltrlan_mpi']]

  if directory:
    dirs = [directory]
  else:
    dirs = generateGuesses('TRLan')
    
  return checkFortranLib(conf,'TRLAN',dirs,libs,functions)
