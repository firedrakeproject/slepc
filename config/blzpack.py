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

  if petscconf.SCALAR == 'complex':
    log.Exit('ERROR: BLZPACK does not support complex numbers.')

  if (petscconf.PRECISION != 'single') & (petscconf.PRECISION != 'double'):
    log.Exit('ERROR: BLZPACK is supported only in single or double precision.')

  if petscconf.IND64:
    log.Exit('ERROR: cannot use external packages with 64-bit indices.')

  if petscconf.PRECISION == 'single':
    functions = ['blzdrs']
  else:
    functions = ['blzdrd']

  if libs:
    libs = [libs]
  else:
    libs = [['-lblzpack']]

  if directory:
    dirs = [directory]
  else:
    dirs = check.GenerateGuesses('Blzpack')

  return check.FortranLib(tmpdir,conf,vars,cmake,'BLZPACK',dirs,libs,functions)
