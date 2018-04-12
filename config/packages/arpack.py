#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

import os, commands
import log, package

class Arpack(package.Package):

  def __init__(self,argdb,log):
    package.Package.__init__(self,argdb,log)
    self.packagename    = 'arpack'
    self.installable    = True
    self.downloadable   = True
    self.version        = '3.3.0'
    self.url            = 'https://github.com/opencollab/arpack-ng/archive/'+self.version+'.tar.gz'
    self.archive        = 'arpack-ng-'+self.version+'.tar.gz'
    self.dirname        = 'arpack-ng-'+self.version
    self.supportssingle = True
    self.fortran        = True
    self.ProcessArgs(argdb)

  def Functions(self,petsc):
    if petsc.mpiuni:
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


  def Check(self,conf,vars,cmake,petsc):
    functions = self.Functions(petsc)
    if self.packagelibs:
      libs = [self.packagelibs]
    else:
      if petsc.mpiuni:
        libs = [['-larpack'],['-larpack_LINUX'],['-larpack_SUN4']]
      else:
        libs = [['-lparpack','-larpack'],['-lparpack_MPI','-larpack'],['-lparpack_MPI-LINUX','-larpack_LINUX'],['-lparpack_MPI-SUN4','-larpack_SUN4']]

    if self.packagedir:
      dirs = [self.packagedir]
    else:
      dirs = self.GenerateGuesses('Arpack')

    self.FortranLib(conf,vars,cmake,dirs,libs,functions)


  def Install(self,conf,vars,cmake,petsc,archdir):
    externdir = os.path.join(archdir,'externalpackages')
    builddir  = os.path.join(externdir,self.dirname)
    self.Download(externdir,builddir)

    # Check for autoreconf
    result,output = commands.getstatusoutput('autoreconf --help')
    if result:
      self.log.Exit('ERROR: --download-arpack requires that the command autoreconf is available on your PATH.')

    # Build package
    confopt = '--prefix='+archdir+' F77="'+petsc.fc+'" FFLAGS="'+petsc.fc_flags.replace('-Wall','').replace('-Wshadow','')+'"'
    if not petsc.mpiuni:
      confopt = confopt+' --enable-mpi'
    if not petsc.buildsharedlib:
      confopt = confopt+' --disable-shared'
    result,output = commands.getstatusoutput('cd '+builddir+'&& sh bootstrap && ./configure '+confopt+' && '+petsc.make+' && '+petsc.make+' install')
    self.log.write(output)
    if result:
      self.log.Exit('ERROR: installation of ARPACK failed.')

    # Check build
    functions = self.Functions(petsc)
    if petsc.mpiuni:
      libs = [['-larpack']]
    else:
      libs = [['-lparpack','-larpack']]
    libDir = os.path.join(archdir,'lib')
    dirs = [libDir]
    self.FortranLib(conf,vars,cmake,dirs,libs,functions)
    self.havepackage = True

