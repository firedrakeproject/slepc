#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

import os, commands, shutil
import log, package

class Blopex(package.Package):

  def __init__(self,argdb,log):
    package.Package.__init__(self,argdb,log)
    self.packagename  = 'blopex'
    self.downloadable = True
    self.version      = '1.1.2'
    self.url          = 'http://slepc.upv.es/download/external/blopex-'+self.version+'.tar.gz'
    self.archive      = 'blopex.tar.gz'
    self.dirname      = 'blopex-'+self.version
    self.ProcessArgs(argdb)

  def Install(self,conf,vars,cmake,petsc,archdir):
    externdir = os.path.join(archdir,'externalpackages')
    builddir  = os.path.join(externdir,self.dirname)
    self.Download(externdir,builddir)

    # Configure
    g = open(os.path.join(builddir,'Makefile.inc'),'w')
    g.write('CC          = '+petsc.cc+'\n')
    if petsc.ind64: blopexint = ' -DBlopexInt="long long" '
    else: blopexint = ''
    g.write('CFLAGS      = '+petsc.cc_flags.replace('-Wall','').replace('-Wshadow','')+blopexint+'\n')
    g.write('AR          = '+petsc.ar+' '+petsc.ar_flags+'\n')
    g.write('AR_LIB_SUFFIX = '+petsc.ar_lib_suffix+'\n')
    g.write('RANLIB      = '+petsc.ranlib+'\n')
    g.write('TARGET_ARCH = \n')
    g.close()

    # Build package
    result,output = commands.getstatusoutput('cd '+builddir+'&&'+petsc.make+' clean &&'+petsc.make)
    self.log.write(output)
    if result:
      self.log.Exit('ERROR: installation of BLOPEX failed.')

    # Move files
    incDir = os.path.join(archdir,'include')
    libDir = os.path.join(archdir,'lib')
    os.rename(os.path.join(builddir,'lib','libBLOPEX.'+petsc.ar_lib_suffix),os.path.join(libDir,'libBLOPEX.'+petsc.ar_lib_suffix))
    for root, dirs, files in os.walk(os.path.join(builddir,'include')):
      for name in files:
        shutil.copyfile(os.path.join(builddir,'include',name),os.path.join(incDir,name))

    if petsc.buildsharedlib:
      l = petsc.slflag + libDir + ' -L' + libDir + ' -lBLOPEX'
    else:
      l = '-L' + libDir + ' -lBLOPEX'
    f = '-I' + incDir

    # Check build
    if petsc.scalar == 'real':
      functions = ['lobpcg_solve_double']
    else:
      functions = ['lobpcg_solve_complex']
    if not self.Link(functions,[],[l]+[f]):
      self.log.Exit('\nERROR: Unable to link with downloaded BLOPEX')

    # Write configuration files
    conf.write('#ifndef SLEPC_HAVE_BLOPEX\n#define SLEPC_HAVE_BLOPEX 1\n#endif\n\n')
    vars.write('BLOPEX_LIB = ' + l + '\n')
    cmake.write('set (SLEPC_HAVE_BLOPEX YES)\n')
    cmake.write('find_library (BLOPEX_LIB BLOPEX HINTS '+ libDir +')\n')

    self.havepackage = True
    self.packageflags = [l] + [f]

