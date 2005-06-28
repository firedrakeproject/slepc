import os
import sys

import petscconf
import check

def Check(conf,directory,libs):

  if petscconf.SCALAR == 'complex':
    sys.exit('ERROR: BLZPACK does not support complex numbers.') 
  
  if petscconf.PRECISION == 'double':
    functions = ['blzdrd']
  else:
    functions = ['blzdrs']

  if libs:
    libs = [libs]
  else:
    libs = [['-lblzpack']]

  if directory:
    dirs = [directory]
  else:
    dirs = check.GenerateGuesses('Blzpack')

  return check.FortranLib(conf,'BLZPACK',dirs,libs,functions)
