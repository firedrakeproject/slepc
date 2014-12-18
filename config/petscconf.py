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

def Load(petscdir):
  global ARCH,DIR,MAKE,SCALAR,PRECISION,ISINSTALL,DESTDIR,BFORT,TEST_RUNS,CC,CC_FLAGS,FC,AR,AR_FLAGS,AR_LIB_SUFFIX,RANLIB,IND64,BUILD_USING_CMAKE,MAKE_IS_GNUMAKE,MPIUNI,SINGLELIB,SLFLAG,LANGUAGE,DEBUG

  if 'PETSC_ARCH' in os.environ and os.environ['PETSC_ARCH']:
    ISINSTALL = 0
    ARCH = os.environ['PETSC_ARCH']
    PETSCVARIABLES = os.sep.join([petscdir,ARCH,'conf','petscvariables'])
    PETSCCONF_H = os.sep.join([petscdir,ARCH,'include','petscconf.h'])
  else:
    ISINSTALL = 1
    PETSCVARIABLES = os.sep.join([petscdir,'conf','petscvariables'])
    PETSCCONF_H = os.sep.join([petscdir,'include','petscconf.h'])

  BUILD_USING_CMAKE = 0
  MAKE_IS_GNUMAKE = 0
  SINGLELIB = 0
  LANGUAGE = 'c'
  try:
    f = open(PETSCVARIABLES)
    for l in f.readlines():
      r = l.split('=',1)
      if len(r)!=2: continue
      k = r[0].strip()
      v = r[1].strip()
      if k == 'PETSC_SCALAR':
        SCALAR = v
      elif k == 'PETSC_PRECISION':
        PRECISION = v
      elif k == 'MAKE':
        MAKE = v
      elif k == 'DESTDIR':
        DESTDIR = v
      elif k == 'BFORT':
        BFORT = v
      elif k == 'TEST_RUNS':
        TEST_RUNS = v
      elif k == 'CC':
        CC = v
      elif k == 'CC_FLAGS':
        CC_FLAGS = v
      elif k == 'FC' and not v=='':
        FC = v
      elif k == 'AR':
        AR = v
      elif k == 'AR_FLAGS':
        AR_FLAGS = v
      elif k == 'AR_LIB_SUFFIX':
        AR_LIB_SUFFIX = v
      elif k == 'CC_LINKER_SLFLAG':
        SLFLAG = v
      elif k == 'RANLIB':
        RANLIB = v
      elif k == 'PETSC_BUILD_USING_CMAKE':
        BUILD_USING_CMAKE = v
      elif k == 'MAKE_IS_GNUMAKE':
        MAKE_IS_GNUMAKE = v
      elif k == 'PETSC_LANGUAGE' and v=='CXXONLY':
        LANGUAGE = 'c++'
    f.close()
  except:
    sys.exit('ERROR: cannot process file ' + PETSCVARIABLES)

  IND64 = 0
  MPIUNI = 0
  DEBUG = 0
  try:
    f = open(PETSCCONF_H)
    for l in f.readlines():
      l = l.split()
      if len(l)==3 and l[0]=='#define' and l[1]=='PETSC_USE_64BIT_INDICES' and l[2]=='1':
        IND64 = 1
      elif len(l)==3 and l[0]=='#define' and l[1]=='PETSC_HAVE_MPIUNI' and l[2]=='1':
        MPIUNI = 1
      elif len(l)==3 and l[0]=='#define' and l[1]=='PETSC_USE_DEBUG' and l[2]=='1':
        DEBUG = 1
      elif len(l)==3 and l[0]=='#define' and l[1]=='PETSC_USE_SINGLE_LIBRARY' and l[2]=='1':
        SINGLELIB = 1
    f.close()
  except:
    if ISINSTALL:
      sys.exit('ERROR: cannot process file ' + PETSCCONF_H + ', maybe you forgot to set PETSC_ARCH')
    else:
      sys.exit('ERROR: cannot process file ' + PETSCCONF_H)

  # empty PETSC_ARCH, guess an arch name
  if ISINSTALL:
    ARCH = 'arch-' + sys.platform.replace('cygwin','mswin')+ '-' + LANGUAGE
    if DEBUG:
      ARCH += '-debug'
    else:
      ARCH += '-opt'
    if not 'real' in SCALAR:
      ARCH += '-' + SCALAR

