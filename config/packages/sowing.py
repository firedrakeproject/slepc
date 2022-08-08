#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

from __future__ import print_function
import os, sys, log, package

class Sowing(package.Package):

  def __init__(self,argdb,log):
    package.Package.__init__(self,argdb,log)
    self.packagename  = 'sowing'
    self.downloadable = True
    self.version      = '1.1.26-p5'
    self.url          = 'https://bitbucket.org/petsc/pkg-sowing/get/v'+self.version+'.tar.gz'
    self.archive      = 'sowing-'+self.version+'.tar.gz'
    self.ProcessArgs(argdb)

  def ShowHelp(self):
    wd = package.Package.wd
    print('  --download-sowing[=<fname>]'.ljust(wd)+': Download and install SOWING')
    print('  The latter is needed (for Fortran) only if using a git version of SLEPc and a non-git version of PETSc')

  def DownloadAndInstall(self,slepc,petsc,archdir):
    name = self.packagename.upper()
    self.log.NewSection('Installing '+name+'...')

    # Get package
    externdir = slepc.GetExternalPackagesDir(archdir)
    builddir  = self.Download(externdir,slepc.downloaddir)

    # Configure, build and install package
    (result,output) = self.RunCommand('cd '+builddir+'&& ./configure --prefix='+archdir+'&&'+petsc.make+'&&'+petsc.make+' install')

    self.havepackage = True
    return os.path.join(archdir,'bin','bfort')

  def Process(self,slepcconf,slepcvars,slepcrules,slepc,petsc,archdir=''):
    # Download sowing if requested and make Fortran stubs if necessary
    bfort = petsc.bfort
    if self.downloadpackage:
      bfort = self.DownloadAndInstall(slepc,petsc,archdir)
    if slepc.isrepo and petsc.fortran:
      try:
        if not os.path.exists(bfort):
          bfort = os.path.join(archdir,'bin','bfort')
        if not os.path.exists(bfort):
          bfort = self.DownloadAndInstall(slepc,petsc,archdir)
        self.log.NewSection('Generating Fortran stubs...')
        self.log.write('Using BFORT='+bfort)
        sys.path.insert(0, os.path.abspath(os.path.join('lib','slepc','bin','maint')))
        import generatefortranstubs
        generatefortranstubs.main(slepc.dir,bfort,os.getcwd(),0)
        generatefortranstubs.processf90interfaces(slepc.dir,0)
      except:
        self.log.Exit('Try configuring with --download-sowing or use a git version of PETSc')
    if bfort != petsc.bfort:
      slepcvars.write('BFORT = '+bfort+'\n')

