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

import os, sys, commands
import log, package

class Sowing(package.Package):

  def __init__(self,argdb,log):
    package.Package.__init__(self,argdb,log)
    self.packagename  = 'sowing'
    self.downloadable = True
    self.url          = 'https://bitbucket.org/petsc/pkg-sowing.git'
    self.ProcessArgs(argdb)

  def Install(self,archdir,make):
    name = self.packagename.upper()
    self.log.NewSection('Installing '+name+'...')

    # Create externalpackages directory
    externdir = os.path.join(archdir,'externalpackages')
    if not os.path.exists(externdir):
      try:
        os.mkdir(externdir)
      except:
        self.log.Exit('ERROR: cannot create directory ' + externdir)

    # Check if source is already available
    builddir = os.path.join(externdir,'pkg-sowing')
    if os.path.exists(builddir):
      self.log.write('Using '+builddir)
    else: # clone Sowing repo
      url = self.packageurl
      if url=='':
        url = self.url
      try:
        result,output = commands.getstatusoutput('cd '+externdir+'&& git clone '+url)
        self.log.write(output)
      except RuntimeError, e:
        self.log.Exit('Error cloning '+url+': '+str(e))

    # Configure, build and install package
    result,output = commands.getstatusoutput('cd '+builddir+'&& ./configure --prefix='+archdir+'&&'+make+'&&'+make+' install')
    self.log.write(output)

    self.havepackage = True
    return os.path.join(archdir,'bin','bfort')
