import os
import sys

from check import *

def checkPlanso(conf,directory,libs,scalar,precision,uniprocessor):

  if scalar == 'complex':
    sys.exit('ERROR: PLANSO does not support complex numbers.') 

  if precision == 'single':
    sys.exit('ERROR: PLANSO does not support single precision.') 

  functions = ['plandr2']
  if libs:
    libs = [libs]
  else:
    libs = [['-lplan','-llanso']]

  if directory:
    dirs = [directory]
  else:
    dirs = generateGuesses('Plan')

  return checkFortranLib(conf,'PLANSO',dirs,libs,functions,['op','opm'])
