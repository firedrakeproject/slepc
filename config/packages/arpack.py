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

class Arpack(package.Package):

  def __init__(self,argdb,log):
    package.Package.__init__(self,argdb,log)
    self.packagename    = 'arpack'
    self.installable    = True
    self.downloadable   = True
    self.version        = '3.7.0'
    self.url            = 'https://github.com/opencollab/arpack-ng/archive/'+self.version+'.tar.gz'
    self.archive        = 'arpack-ng-'+self.version+'.tar.gz'
    self.dirname        = 'arpack-ng-'+self.version
    self.supportssingle = True
    self.fortran        = True
    self.hasheaders     = True   # the option --with-arpack-include=... is simply ignored
    self.ProcessArgs(argdb)

  def Functions(self,petsc):
    if petsc.mpiuni or petsc.msmpi:
      if petsc.scalar == 'real':
        if petsc.precision == 'single':
          functions = ['snaupd','sneupd','ssaupd','sseupd']
        else:
          functions = ['dnaupd','dneupd','dsaupd','dseupd']
      else:
        if petsc.precision == 'single':
          functions = ['cnaupd','cneupd']
        else:
          functions = ['znaupd','zneupd']
    else:
      if petsc.scalar == 'real':
        if petsc.precision == 'single':
          functions = ['psnaupd','psneupd','pssaupd','psseupd']
        else:
          functions = ['pdnaupd','pdneupd','pdsaupd','pdseupd']
      else:
        if petsc.precision == 'single':
          functions = ['pcnaupd','pcneupd']
        else:
          functions = ['pznaupd','pzneupd']
    return functions


  def Check(self,slepcconf,slepcvars,petsc,archdir):
    functions = self.Functions(petsc)
    if self.packagelibs:
      libs = [self.packagelibs]
    else:
      if petsc.mpiuni or petsc.msmpi:
        libs = [['-larpack'],['-larpack_LINUX'],['-larpack_SUN4']]
      else:
        libs = [['-lparpack','-larpack'],['-lparpack_MPI','-larpack'],['-lparpack_MPI-LINUX','-larpack_LINUX'],['-lparpack_MPI-SUN4','-larpack_SUN4']]

    if self.packagedir:
      dirs = [os.path.join(self.packagedir,'lib'),self.packagedir,os.path.join(self.packagedir,'lib64')]
    else:
      dirs = self.GenerateGuesses('Arpack',archdir)
    self.FortranLib(slepcconf,slepcvars,dirs,libs,functions)


  def DownloadAndInstall(self,slepcconf,slepcvars,slepc,petsc,archdir,prefixdir):
    externdir = slepc.CreateDir(archdir,'externalpackages')
    builddir  = self.Download(externdir,slepc.downloaddir)

    # Check for autoreconf
    (result,output) = self.RunCommand('autoreconf --help')
    if result:
      self.log.Exit('--download-arpack requires that the command autoreconf is available on your PATH')

    # Build package
    extra_fcflags = ' -fallow-argument-mismatch' if petsc.isGfortran100plus() else ''
    confopt = '--prefix='+prefixdir+' CC="'+petsc.cc+'" CFLAGS="'+petsc.cc_flags+'" F77="'+petsc.fc+'" FFLAGS="'+petsc.fc_flags.replace('-Wall','').replace('-Wshadow','')+extra_fcflags+'" FC="'+petsc.fc+'" FCFLAGS="'+petsc.fc_flags.replace('-Wall','').replace('-Wshadow','')+extra_fcflags+'" LIBS="'+petsc.blaslapack_lib+'"'
    if not petsc.mpiuni and not petsc.msmpi:
      confopt = confopt+' --enable-mpi MPICC="'+petsc.cc+'" MPIF77="'+petsc.fc+'" MPIFC="'+petsc.fc+'"'
    if not petsc.buildsharedlib:
      confopt = confopt+' --disable-shared'
    if petsc.ind64:
      if not petsc.blaslapackint64:
        self.log.Exit('To install ARPACK with 64-bit integers you also need a BLAS with 64-bit integers')
      confopt = confopt+' INTERFACE64=1'
    (result,output) = self.RunCommand('cd '+builddir+'&& sh bootstrap && ./configure '+confopt+' && '+petsc.make+' && '+petsc.make+' install')
    if result:
      self.log.Exit('Installation of ARPACK failed')

    # Check build
    functions = self.Functions(petsc)
    if petsc.mpiuni or petsc.msmpi:
      libs = [['-larpack']]
    else:
      libs = [['-lparpack','-larpack']]
    dirs = [os.path.join(prefixdir,'lib'),os.path.join(prefixdir,'lib64')]
    self.FortranLib(slepcconf,slepcvars,dirs,libs,functions)

