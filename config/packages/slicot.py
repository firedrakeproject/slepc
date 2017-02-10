#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain
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

import os, commands
import log, package

class Slicot(package.Package):

  def __init__(self,argdb,log):
    package.Package.__init__(self,argdb,log)
    self.packagename    = 'slicot'
    self.installable    = True
    self.downloadable   = True
    self.version        = '4.5'
    self.url            = 'http://slicot.org/objects/software/shared/slicot45.tar.gz'
    self.archive        = 'slicot45.tar.gz'
    self.dirname        = 'slicot'
    self.supportsscalar = ['real']
    self.fortran        = True
    self.ProcessArgs(argdb)


  def Check(self,conf,vars,cmake,petsc):
    functions = ['sb03od']
    if self.packagelibs:
      libs = [self.packagelibs]
    else:
      libs = [['-lslicot']]

    if self.packagedir:
      dirs = [self.packagedir]
    else:
      dirs = self.GenerateGuesses('slicot')

    self.FortranLib(conf,vars,cmake,dirs,libs,functions)


  def Install(self,conf,vars,cmake,petsc,archdir):
    externdir = os.path.join(archdir,'externalpackages')
    builddir  = os.path.join(externdir,self.dirname)
    self.Download(externdir,builddir)
    libname = 'libslicot.a'

    # Configure
    g = open(os.path.join(builddir,'make.inc'),'w')
    g.write('FORTRAN   = '+petsc.fc+'\n')
    g.write('OPTS      = '+petsc.fc_flags.replace('-Wall','').replace('-Wshadow','')+'\n')
    g.write('ARCH      = '+petsc.ar+'\n')
    g.write('ARCHFLAGS = '+petsc.ar_flags+'\n')
    g.write('SLICOTLIB = ../'+libname+'\n')
    g.close()

    # Build package
    target = 'lib'
    result,output = commands.getstatusoutput('cd '+builddir+'&&'+petsc.make+' clean &&'+petsc.make+' '+target)
    self.log.write(output)
    if result:
      self.log.Exit('ERROR: installation of SLICOT failed.')

    # Move files
    libDir = os.path.join(archdir,'lib')
    os.rename(os.path.join(builddir,libname),os.path.join(libDir,libname))

    # Check build
    functions = ['sb03od']
    libs = [['-lslicot']]
    libDir = os.path.join(archdir,'lib')
    dirs = [libDir]
    self.FortranLib(conf,vars,cmake,dirs,libs,functions)
    self.havepackage = True

