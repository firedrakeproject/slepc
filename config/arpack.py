import os
import sys

from check import *

def checkArpack(conf,directory,libs,scalar,precision,uniprocessor):

  if uniprocessor:

    if scalar == 'real':
      if precision == 'double':
	functions = ['dnaupd','dneupd','dsaupd','dseupd']
      else:
	functions = ['snaupd','sneupd','dsaupd','dseupd']
    else:
      if precision == 'double':
	functions = ['znaupd','zneupd']
      else:
	functions = ['cnaupd','cneupd']

    if libs:
      libs = [libs]
    else:
      libs = [['-larpack'],['-larpack_LINUX'],['-larpack_SUN4']]

  else:

    if scalar == 'real':
      if precision == 'double':
	functions = ['pdnaupd','pdneupd','pdsaupd','pdseupd']
      else:
	functions = ['psnaupd','psneupd','pdsaupd','pdseupd']
    else:
      if precision == 'double':
	functions = ['pznaupd','pzneupd']
      else:
	functions = ['pcnaupd','pcneupd']

    if libs:
      libs = [libs]
    else:
      libs = [['-lparpack','-larpack'],['-lparpack_LINUX','-larpack_LINUX'],['-lparpack_SUN4','-larpack_SUN4']]

  if directory:
    dirs = [directory]
  else:
    dirs = generateGuesses('Arpack')
    
  return checkFortranLib(conf,'ARPACK',dirs,libs,functions)
