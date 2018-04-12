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

class Primme(package.Package):

  def __init__(self,argdb,log):
    package.Package.__init__(self,argdb,log)
    self.packagename    = 'primme'
    self.installable    = True
    self.downloadable   = True
    self.version        = '2.1'
    self.url            = 'https://github.com/primme/primme/tarball/release-2.1'
    self.archive        = 'primme-'+self.version+'.tar.gz'
    self.dirname        = 'PRIMME'
    self.supportssingle = True
    self.supports64bint = True
    self.ProcessArgs(argdb)

  def Check(self,conf,vars,cmake,petsc):
    functions_base = ['primme_set_method','primme_free','primme_initialize']
    if self.packagedir:
      dirs = [os.path.join(self.packagedir,'lib')]
    else:
      dirs = self.GenerateGuesses('Primme')

    libs = self.packagelibs
    if not libs:
      libs = ['-lprimme']
    if petsc.scalar == 'real':
      if petsc.precision == 'single':
        functions = functions_base + ['sprimme']
      else:
        functions = functions_base + ['dprimme']
    else:
      if petsc.precision == 'single':
        functions = functions_base + ['cprimme']
      else:
        functions = functions_base + ['zprimme']

    for d in dirs:
      if d:
        if petsc.buildsharedlib:
          l = [petsc.slflag + d] + ['-L' + d] + libs
        else:
          l = ['-L' + d] + libs
        f = ['-I' + os.path.join(os.path.dirname(d),'include')]
      else:
        l =  libs
        f = []
      if self.Link(functions,[],l+f):
        conf.write('#ifndef SLEPC_HAVE_PRIMME\n#define SLEPC_HAVE_PRIMME 1\n#endif\n\n')
        vars.write('PRIMME_LIB = ' + ' '.join(l) + '\n')
        vars.write('PRIMME_FLAGS = ' + ' '.join(f) + '\n')
        cmake.write('set (SLEPC_HAVE_PRIMME YES)\n')
        cmake.write('find_library (PRIMME_LIB primme HINTS '+ d +')\n')
        cmake.write('find_path (PRIMME_INCLUDE primme.h ' + d + '/include)\n')
        self.havepackage = True
        self.packageflags = l+f
        return

    self.log.Println('\nERROR: Unable to link with PRIMME library')
    self.log.Println('ERROR: In directories '+' '.join(dirs))
    self.log.Println('ERROR: With flags '+' '.join(libs))
    self.log.Println('NOTE: make sure PRIMME version is 2.0 at least')
    self.log.Exit('')


  def Install(self,conf,vars,cmake,petsc,archdir):
    externdir = os.path.join(archdir,'externalpackages')
    builddir  = os.path.join(externdir,self.dirname)
    self.Download(externdir,builddir,'primme-')

    # Configure
    g = open(os.path.join(builddir,'Make_flags'),'w')
    g.write('LIBRARY     = libprimme.'+petsc.ar_lib_suffix+'\n')
    g.write('SOLIBRARY   = libprimme.'+petsc.sl_suffix+'\n')
    g.write('CC          = '+petsc.cc+'\n')
    if hasattr(petsc,'fc'):
      g.write('F77         = '+petsc.fc+'\n')
    g.write('DEFINES     = ')
    if petsc.blaslapackunderscore:
      g.write('-DF77UNDERSCORE ')
    if petsc.blaslapackint64:
      g.write('-DPRIMME_BLASINT_SIZE=64')
    g.write('\n')
    g.write('INCLUDE     = \n')
    g.write('CFLAGS      = '+petsc.cc_flags.replace('-Wall','').replace('-Wshadow','')+'\n')
    g.write('RANLIB      = '+petsc.ranlib+'\n')
    g.close()

    # Build package
    result,output = commands.getstatusoutput('cd '+builddir+'&&'+petsc.make+' clean &&'+petsc.make)
    self.log.write(output)
    if result:
      self.log.Exit('ERROR: installation of PRIMME failed.')

    # Move files
    incDir = os.path.join(archdir,'include')
    libDir = os.path.join(archdir,'lib')
    os.rename(os.path.join(builddir,'lib','libprimme.'+petsc.ar_lib_suffix),os.path.join(libDir,'libprimme.'+petsc.ar_lib_suffix))
    for root, dirs, files in os.walk(os.path.join(builddir,'include')):
      for name in files:
        shutil.copyfile(os.path.join(builddir,'include',name),os.path.join(incDir,name))

    if petsc.buildsharedlib:
      l = petsc.slflag + libDir + ' -L' + libDir + ' -lprimme'
    else:
      l = '-L' + libDir + ' -lprimme'
    f = '-I' + incDir

    # Check build
    functions_base = ['primme_set_method','primme_free','primme_initialize']
    if petsc.scalar == 'real':
      if petsc.precision == 'single':
        functions = functions_base + ['sprimme']
      else:
        functions = functions_base + ['dprimme']
    else:
      if petsc.precision == 'single':
        functions = functions_base + ['cprimme']
      else:
        functions = functions_base + ['zprimme']
    if not self.Link(functions,[],[l]+[f]):
      self.log.Exit('\nERROR: Unable to link with downloaded PRIMME')

    # Write configuration files
    conf.write('#ifndef SLEPC_HAVE_PRIMME\n#define SLEPC_HAVE_PRIMME 1\n#endif\n\n')
    vars.write('PRIMME_LIB = ' + l + '\n')
    cmake.write('set (SLEPC_HAVE_PRIMME YES)\n')
    cmake.write('find_library (PRIMME_LIB primme HINTS '+ libDir +')\n')
    cmake.write('find_path (PRIMME_INCLUDE primme.h ' + incDir + ')\n')

    self.havepackage = True
    self.packageflags = [l] + [f]

