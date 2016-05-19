#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#
#  SLEPc is free software: you can redistribute it and/or modify it under  the
#  terms of version 3 of the GNU Lesser General Public License as published by
#  the Free Software Foundation.
#
#  SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
#  WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
#  FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
#  more details.
#
#  You  should have received a copy of the GNU Lesser General  Public  License
#  along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

import os, commands, shutil
import log, package

class Primme(package.Package):

  def __init__(self,argdb,log):
    package.Package.__init__(self,argdb,log)
    self.packagename  = 'primme'
    self.installable  = True
    self.downloadable = True
    self.version      = '1.2.2'
    self.url          = 'https://github.com/primme/primme/tarball/release-'+self.version
    self.archive      = 'primme-'+self.version+'.tar.gz'
    self.dirname      = 'PRIMME'
    self.ProcessArgs(argdb)

  def Check(self,conf,vars,cmake,petsc):
    functions_base = ['primme_set_method','primme_Free','primme_initialize']
    if self.packagedir:
      dirs = [self.packagedir]
    else:
      dirs = self.GenerateGuesses('Primme')

    libs = self.packagelibs
    if not libs:
      libs = ['-lprimme']
    if petsc.scalar == 'real':
      functions = functions_base + ['dprimme']
    else:
      functions = functions_base + ['zprimme']

    for d in dirs:
      if d:
        if 'rpath' in petsc.slflag:
          l = [petsc.slflag + d] + ['-L' + d] + libs
        else:
          l = ['-L' + d] + libs
        f = ['-I' + os.path.join(d,'PRIMMESRC','COMMONSRC')]
      else:
        l =  libs
        f = []
      if self.Link(functions,[],l+f):
        conf.write('#ifndef SLEPC_HAVE_PRIMME\n#define SLEPC_HAVE_PRIMME 1\n#endif\n\n')
        vars.write('PRIMME_LIB = ' + ' '.join(l) + '\n')
        vars.write('PRIMME_FLAGS = ' + ' '.join(f) + '\n')
        cmake.write('set (SLEPC_HAVE_PRIMME YES)\n')
        cmake.write('find_library (PRIMME_LIB primme HINTS '+ d +')\n')
        cmake.write('find_path (PRIMME_INCLUDE primme.h ' + d + '/PRIMMESRC/COMMONSRC)\n')
        self.havepackage = True
        self.packageflags = l+f
        return

    self.log.Println('\nERROR: Unable to link with PRIMME library')
    self.log.Println('ERROR: In directories '+' '.join(dirs))
    self.log.Println('ERROR: With flags '+' '.join(libs))
    self.log.Exit('')


  def Install(self,conf,vars,cmake,petsc,archdir):
    externdir = os.path.join(archdir,'externalpackages')
    builddir  = os.path.join(externdir,self.dirname)
    self.Download(externdir,builddir,'primme-')

    # Configure
    g = open(os.path.join(builddir,'Make_flags'),'w')
    g.write('LIBRARY     = libprimme.a\n')
    g.write('DLIBRARY    = libdprimme.a\n')
    g.write('ZLIBRARY    = libzprimme.a\n')
    g.write('CC          = '+petsc.cc+'\n')
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
    os.rename(os.path.join(builddir,'libprimme.'+petsc.ar_lib_suffix),os.path.join(libDir,'libprimme.'+petsc.ar_lib_suffix))
    for name in ['primme.h','primme_f77.h','Complexz.h']:
      shutil.copyfile(os.path.join(builddir,'PRIMMESRC','COMMONSRC',name),os.path.join(incDir,name))

    if 'rpath' in petsc.slflag:
      l = petsc.slflag + libDir + ' -L' + libDir + ' -lprimme'
    else:
      l = '-L' + libDir + ' -lprimme'
    f = '-I' + incDir

    # Check build
    functions_base = ['primme_set_method','primme_Free','primme_initialize']
    if petsc.scalar == 'real':
      functions = functions_base + ['dprimme']
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

