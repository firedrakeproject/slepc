#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
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
      except RuntimeError as e:
        self.log.Exit('Error cloning '+url+': '+str(e))

    # Configure, build and install package
    result,output = commands.getstatusoutput('cd '+builddir+'&& ./configure --prefix='+archdir+'&&'+make+'&&'+make+' install')
    self.log.write(output)

    self.havepackage = True
    return os.path.join(archdir,'bin','bfort')
