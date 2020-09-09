#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2020, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

import os, log, package

class Trlan(package.Package):

  def __init__(self,argdb,log):
    package.Package.__init__(self,argdb,log)
    self.packagename    = 'trlan'
    self.installable    = True
    self.downloadable   = True
    self.version        = '201009'
    self.url            = 'https://codeforge.lbl.gov/frs/download.php/210/trlan-'+self.version+'.tar.gz'
    self.archive        = 'trlan-'+self.version+'.tar.gz'
    self.dirname        = 'trlan-'+self.version
    self.supportsscalar = ['real']
    self.fortran        = True
    self.ProcessArgs(argdb)

  def Check(self,slepcconf,slepcvars,petsc,archdir):
    functions = ['trlan77']
    if self.packagelibs:
      libs = [self.packagelibs]
    else:
      if petsc.mpiuni:
        libs = [['-ltrlan']]
      else:
        libs = [['-ltrlan_mpi']]

    if self.packagedir:
      dirs = [os.path.join(self.packagedir,'lib'),self.packagedir]
    else:
      dirs = self.GenerateGuesses('TRLan',archdir)

    self.FortranLib(slepcconf,slepcvars,dirs,libs,functions)


  def DownloadAndInstall(self,slepcconf,slepcvars,slepc,petsc,archdir,prefixdir):
    externdir = slepc.CreateDir(archdir,'externalpackages')
    builddir  = self.Download(externdir,slepc.downloaddir)
    extra_fcflags = ' -fallow-argument-mismatch' if petsc.isGfortran100plus() else ''

    # Makefile
    cont  = 'FC     = '+petsc.fc+'\n'
    cont += 'F90    = '+petsc.fc+'\n'
    cont += 'FFLAGS = '+petsc.fc_flags.replace('-Wall','').replace('-Wshadow','')+extra_fcflags+'\n'
    cont += 'SHELL  = /bin/sh\n'
    self.log.write('Using makefile definitions:\n')
    self.log.write(cont)
    mfile = open(os.path.join(builddir,'Make.inc'),'w')
    mfile.write(cont)
    mfile.close()

    # Build package
    if petsc.mpiuni:
      target = 'lib'
    else:
      target = 'plib'
    (result,output) = self.RunCommand('cd '+builddir+'&&'+petsc.make+' clean &&'+petsc.make+' '+target)
    if result:
      self.log.Exit('Installation of TRLAN failed')

    # Move files
    incdir,libdir = slepc.CreatePrefixDirs(prefixdir)
    if petsc.mpiuni:
      libName = 'libtrlan.a'
    else:
      libName = 'libtrlan_mpi.a'
    os.rename(os.path.join(builddir,libName),os.path.join(libdir,libName))

    # Check build
    functions = ['trlan77']
    if petsc.mpiuni:
      libs = [['-ltrlan']]
    else:
      libs = [['-ltrlan_mpi']]
    dirs = [libdir]
    self.FortranLib(slepcconf,slepcvars,dirs,libs,functions)

