#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2019, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

import os, shutil, log, package

class Blopex(package.Package):

  def __init__(self,argdb,log):
    package.Package.__init__(self,argdb,log)
    self.packagename  = 'blopex'
    self.downloadable = True
    self.url          = 'https://github.com/lobpcg/blopex/archive/master.tar.gz'
    self.archive      = 'blopex-master.tar.gz'
    self.dirname      = 'blopex-master'
    self.hasheaders   = True
    self.ProcessArgs(argdb)

  def DownloadAndInstall(self,conf,vars,slepc,petsc,archdir,prefixdir):
    externdir = os.path.join(archdir,'externalpackages')
    builddir  = os.path.join(externdir,self.dirname,'blopex_abstract')
    self.Download(externdir,builddir)

    # Configure
    g = open(os.path.join(builddir,'Makefile.inc'),'w')
    g.write('CC          = '+petsc.cc+'\n')
    g.write('CFLAGS      = '+petsc.cc_flags.replace('-Wall','').replace('-Wshadow','')+'\n')
    g.write('AR          = '+petsc.ar+' '+petsc.ar_flags+'\n')
    g.write('AR_LIB_SUFFIX = '+petsc.ar_lib_suffix+'\n')
    g.write('RANLIB      = '+petsc.ranlib+'\n')
    g.write('TARGET_ARCH = \n')
    g.close()

    # Build package
    result,output = self.RunCommand('cd '+builddir+'&&'+petsc.make+' clean &&'+petsc.make)
    self.log.write(output)
    if result:
      self.log.Exit('ERROR: installation of BLOPEX failed.')

    # Move files
    incdir,libDir = self.CreatePrefixDirs(prefixdir)
    incblopexdir = os.path.join(incdir,'blopex')
    if not os.path.exists(incblopexdir):
      try:
        os.mkdir(incblopexdir)
      except:
        self.log.Exit('ERROR: Cannot create directory: '+incblopexdir)
    os.rename(os.path.join(builddir,'lib','libBLOPEX.'+petsc.ar_lib_suffix),os.path.join(libDir,'libBLOPEX.'+petsc.ar_lib_suffix))
    for root, dirs, files in os.walk(os.path.join(builddir,'include')):
      for name in files:
        shutil.copyfile(os.path.join(builddir,'include',name),os.path.join(incblopexdir,name))

    if petsc.buildsharedlib:
      l = petsc.slflag + libDir + ' -L' + libDir + ' -lBLOPEX'
    else:
      l = '-L' + libDir + ' -lBLOPEX'
    f = '-I' + incdir + ' -I' + incblopexdir

    # Check build
    if petsc.scalar == 'real':
      functions = ['lobpcg_solve_double']
    else:
      functions = ['lobpcg_solve_complex']
    if not self.Link(functions,[],[l]+[f]):
      self.log.Exit('\nERROR: Unable to link with downloaded BLOPEX')

    # Write configuration files
    conf.write('#define SLEPC_HAVE_BLOPEX 1\n')
    vars.write('BLOPEX_LIB = ' + l + '\n')
    vars.write('BLOPEX_INCLUDE = ' + f + '\n')

    self.havepackage = True
    self.packageflags = [l] + [f]

