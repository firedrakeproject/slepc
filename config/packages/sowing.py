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
    self.inplace      = False
    #self.gitcommit    = '3a410fa51a7bb531676f16deb8bc0c1ded8293c3'
    self.version      = '1.1.26.12'
    obj = self.version if hasattr(self,'version') else self.gitcommit
    self.url          = 'https://bitbucket.org/petsc/pkg-sowing/get/'+('v'+obj if hasattr(self,'version') else obj)+'.tar.gz'
    self.archive      = 'sowing-'+obj+'.tar.gz'
    self.ProcessArgs(argdb)

  def ProcessArgs(self,argdb):
    value,found = argdb.PopBool('with-fortran-bindings-inplace')
    if found:
      self.inplace = value
    package.Package.ProcessArgs(self,argdb)

  def ShowHelp(self):
    wd = package.Package.wd
    print('  --download-sowing[=<fname>]'.ljust(wd)+': Download and install SOWING')
    print('  The latter is needed (for Fortran) only if using a Git version of SLEPc and a non-Git version of PETSc')
    print('  --with-fortran-bindings-inplace=<bool>: Generate Fortran bindings in SLEPc source tree')

  def DownloadAndInstall(self,slepc,petsc,archdir):
    name = self.packagename.upper()
    if hasattr(self,'version') and self.packageurl=='':
      self.log.NewSection('Installing '+name+' version '+self.version+'...')
    else:
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
        generatefortranstubs.main(slepc.dir,petsc.dir,'' if self.inplace else petsc.archname,bfort,slepc.dir,0)
        generatefortranstubs.processf90interfaces(slepc.dir,'' if self.inplace else petsc.archname,0)
      except:
        self.log.Exit('Try configuring with --download-sowing or use a Git version of PETSc')
    if bfort != petsc.bfort:
      slepcvars.write('BFORT = '+bfort+'\n')

