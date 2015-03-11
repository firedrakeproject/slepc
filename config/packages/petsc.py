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

import package, os, commands

class PETSc(package.Package):

  def __init__(self,argdb,log):
    self.packagename  = 'petsc'
    self.downloadable = False
    self.log          = log

  def Check(self):
    self.havepackage = self.Link([],[],[])

  def LoadVersion(self,petscdir):
    try:
      f = open(os.path.join(petscdir,'include','petscversion.h'))
      for l in f.readlines():
        l = l.split()
        if len(l) == 3:
          if l[1] == 'PETSC_VERSION_RELEASE':
            self.release = l[2]
          if l[1] == 'PETSC_VERSION_MAJOR':
            major = l[2]
          elif l[1] == 'PETSC_VERSION_MINOR':
            minor = l[2]
          elif l[1] == 'PETSC_VERSION_SUBMINOR':
            subminor = l[2]
          elif l[1] == 'PETSC_VERSION_PATCH':
            patchlevel = l[2]
      f.close()
      self.version = major + '.' + minor
      self.lversion = major + '.' + minor + '.' + subminor
    except:
      self.log.Exit('ERROR: File error while reading PETSc version')

    # Check whether this is a working copy of the repository
    self.isrepo = False
    if os.path.exists(os.path.join(petscdir,'.git')):
      (status, output) = commands.getstatusoutput('cd '+petscdir+';git rev-parse')
      if not status:
        self.isrepo = True
        (status, self.gitrev) = commands.getstatusoutput('cd '+petscdir+';git log -1 --pretty=format:%H')
        (status, self.gitdate) = commands.getstatusoutput('cd '+petscdir+';git log -1 --pretty=format:%ci')
        (status, self.branch) = commands.getstatusoutput('cd '+petscdir+';git describe --contains --all HEAD')

