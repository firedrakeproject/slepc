import os
import sys

import petscconf
import check

def Check(conf,directory,libs):

  if petscconf.MPIUNI:

    if petscconf.SCALAR == 'real':
      if petscconf.PRECISION == 'double':
	functions = ['dnaupd','dneupd','dsaupd','dseupd']
      else:
	functions = ['snaupd','sneupd','dsaupd','dseupd']
    else:
      if petscconf.PRECISION == 'double':
	functions = ['znaupd','zneupd']
      else:
	functions = ['cnaupd','cneupd']

    if libs:
      libs = [libs]
    else:
      libs = [['-larpack'],['-larpack_LINUX'],['-larpack_SUN4']]

  else:

    if petscconf.SCALAR == 'real':
      if petscconf.PRECISION == 'double':
	functions = ['pdnaupd','pdneupd','pdsaupd','pdseupd']
      else:
	functions = ['psnaupd','psneupd','pdsaupd','pdseupd']
    else:
      if petscconf.PRECISION == 'double':
	functions = ['pznaupd','pzneupd']
      else:
	functions = ['pcnaupd','pcneupd']

    if libs:
      libs = [libs]
    else:
      libs = [['-lparpack','-larpack'],['-lparpack_MPI','-larpack'],['-lparpack_MPI-LINUX','-larpack_LINUX'],['-lparpack_MPI-SUN4','-larpack_SUN4']]

  if directory:
    dirs = [directory]
  else:
    dirs = check.GenerateGuesses('Arpack')
    
  return check.FortranLib(conf,'ARPACK',dirs,libs,functions)
