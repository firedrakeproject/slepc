#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2020, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

import os, sys, log, package

class Sowing(package.Package):

  def __init__(self,argdb,log):
    package.Package.__init__(self,argdb,log)
    self.packagename  = 'sowing'
    self.downloadable = True
    self.version      = '1.1.26'
    self.url          = 'https://bitbucket.org/petsc/pkg-sowing/get/v'+self.version+'-p1.tar.gz'
    self.archive      = 'sowing-'+self.version+'-p1.tar.gz'
    self.dirname      = 'sowing-'+self.version+'-p1'
    self.ProcessArgs(argdb)

  def ShowHelp(self):
    wd = package.Package.wd
    print('  --download-sowing[=<fname>]'.ljust(wd)+': Download and install SOWING')
    print('  The latter is needed (for Fortran) only if using a git version of SLEPc and a non-git version of PETSc')

  def DownloadAndInstall(self,slepc,petsc,archdir):
    name = self.packagename.upper()
    self.log.NewSection('Installing '+name+'...')

    # Get package
    externdir = slepc.CreateDir(archdir,'externalpackages')
    builddir  = self.Download(externdir,slepc.downloaddir)

    # Configure, build and install package
    (result,output) = self.RunCommand('cd '+builddir+'&& ./configure --prefix='+archdir+'&&'+petsc.make+'&&'+petsc.make+' install')

    self.havepackage = True
    return os.path.join(archdir,'bin','bfort')
