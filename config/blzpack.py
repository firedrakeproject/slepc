import os
import sys

from check import *

def checkBlzpack(conf,directory,libs,scalar,precision,uniprocessor):

  if scalar == 'complex':
    sys.exit('ERROR: BLZPACK does not support complex numbers.') 
  
  if precision == 'double':
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
    dirs = generateGuesses('Blzpack')

  return checkFortranLib(conf,'BLZPACK',dirs,libs,functions)
