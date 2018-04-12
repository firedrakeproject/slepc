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

  def Check(self,conf,vars,cmake,petsc):
    functions = ['trlan77']
    if self.packagelibs:
      libs = [self.packagelibs]
    else:
      if petsc.mpiuni:
        libs = [['-ltrlan']]
      else:
        libs = [['-ltrlan_mpi']]

    if self.packagedir:
      dirs = [self.packagedir]
    else:
      dirs = self.GenerateGuesses('TRLan')

    self.FortranLib(conf,vars,cmake,dirs,libs,functions)


  def Install(self,conf,vars,cmake,petsc,archdir):
    externdir = os.path.join(archdir,'externalpackages')
    builddir  = os.path.join(externdir,self.dirname)
    self.Download(externdir,builddir)

    # Configure
    g = open(os.path.join(builddir,'Make.inc'),'w')
    g.write('FC     = '+petsc.fc+'\n')
    g.write('F90    = '+petsc.fc+'\n')
    g.write('FFLAGS = '+petsc.fc_flags.replace('-Wall','').replace('-Wshadow','')+'\n')
    g.write('SHELL  = /bin/sh\n')
    g.close()

    # Build package
    if petsc.mpiuni:
      target = 'lib'
    else:
      target = 'plib'
    result,output = commands.getstatusoutput('cd '+builddir+'&&'+petsc.make+' clean &&'+petsc.make+' '+target)
    self.log.write(output)
    if result:
      self.log.Exit('ERROR: installation of TRLAN failed.')

    # Move files
    libDir = os.path.join(archdir,'lib')
    if petsc.mpiuni:
      libName = 'libtrlan.a'
    else:
      libName = 'libtrlan_mpi.a'
    os.rename(os.path.join(builddir,libName),os.path.join(libDir,libName))

    # Check build
    functions = ['trlan77']
    if petsc.mpiuni:
      libs = [['-ltrlan']]
    else:
      libs = [['-ltrlan_mpi']]
    libDir = os.path.join(archdir,'lib')
    dirs = [libDir]
    self.FortranLib(conf,vars,cmake,dirs,libs,functions)
    self.havepackage = True

