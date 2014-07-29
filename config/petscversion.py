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
import commands

def Load(petscdir):
  global VERSION,RELEASE,LVERSION,ISREPO,GITREV,GITDATE,BRANCH
  try:
    f = open(os.sep.join([petscdir,'include','petscversion.h']))
    for l in f.readlines():
      l = l.split()
      if len(l) == 3:
        if l[1] == 'PETSC_VERSION_RELEASE':
          RELEASE = l[2]
        if l[1] == 'PETSC_VERSION_MAJOR':
          major = l[2]
        elif l[1] == 'PETSC_VERSION_MINOR':
          minor = l[2]
        elif l[1] == 'PETSC_VERSION_SUBMINOR':
          subminor = l[2]
        elif l[1] == 'PETSC_VERSION_PATCH':
          patchlevel = l[2]
    f.close()
    VERSION = major + '.' + minor
    LVERSION = major + '.' + minor + '.' + subminor
  except:
    sys.exit('ERROR: file error while reading PETSC version')

  # Check whether this is a working copy of the repository
  ISREPO = 0
  if os.path.exists(os.sep.join([petscdir,'.git'])):
    (status, output) = commands.getstatusoutput('cd '+petscdir+';git rev-parse')
    if not status:
      ISREPO = 1
      (status, GITREV) = commands.getstatusoutput('cd '+petscdir+';git log -1 --pretty=format:%H')
      (status, GITDATE) = commands.getstatusoutput('cd '+petscdir+';git log -1 --pretty=format:%ci')
      (status, BRANCH) = commands.getstatusoutput('cd '+petscdir+';git describe --contains --all HEAD')

