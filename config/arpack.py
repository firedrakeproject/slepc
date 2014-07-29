#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#
#  SLEPc is free software: you can redistribute it and/or modify it under  the
#  terms of version 3 of the GNU Lesser General Public License as published by
#  the Free Software Foundation.
#
#  SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
#  WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
#  FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
#  more details.
#
#  You  should have received a copy of the GNU Lesser General  Public  License
#  along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

import os
import sys

import petscconf
import log
import check

def Check(conf,vars,cmake,tmpdir,directory,libs):

  if (petscconf.PRECISION != 'single') & (petscconf.PRECISION != 'double'):
    log.Exit('ERROR: ARPACK is supported only in single or double precision.')

  if petscconf.IND64:
    log.Exit('ERROR: cannot use external packages with 64-bit indices.')

  if petscconf.MPIUNI:
    if petscconf.SCALAR == 'real':
      if petscconf.PRECISION == 'single':
        functions = ['snaupd','sneupd','ssaupd','sseupd']
      else:
        functions = ['dnaupd','dneupd','dsaupd','dseupd']
    else:
      if petscconf.PRECISION == 'single':
        functions = ['cnaupd','cneupd']
      else:
        functions = ['znaupd','zneupd']
  else:
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
    if petscconf.MPIUNI:
      libs = [['-larpack'],['-larpack_LINUX'],['-larpack_SUN4']]
    else:
      libs = [['-lparpack','-larpack'],['-lparpack_MPI','-larpack'],['-lparpack_MPI-LINUX','-larpack_LINUX'],['-lparpack_MPI-SUN4','-larpack_SUN4']]

  if directory:
    dirs = [directory]
  else:
    dirs = check.GenerateGuesses('Arpack')

  return check.FortranLib(tmpdir,conf,vars,cmake,'ARPACK',dirs,libs,functions)
