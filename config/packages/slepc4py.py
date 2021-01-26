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

class Slepc4py(package.Package):

  def __init__(self,argdb,log):
    package.Package.__init__(self,argdb,log)
    self.packagename     = 'slepc4py'
    self.installable     = True
    self.ProcessArgs(argdb)

  def ProcessArgs(self,argdb,petscpackages=''):
    url,flag,found = argdb.PopUrl('download-slepc4py')
    if found:
      self.log.Exit('--download-slepc4py has been renamed to --with-slepc4py\nUse -h for help')
    value,found = argdb.PopBool('with-'+self.packagename)
    if found:
      self.requested = value

  def ShowHelp(self):
    wd = package.Package.wd
    print('  --with-slepc4py=<bool>'.ljust(wd)+': Build Python bindings (default: no)')

  def ShowInfo(self):
    if self.havepackage:
      self.log.Println('Python bindings (slepc4py) will be built after SLEPc')

  def Process(self,slepcconf,slepcvars,slepcrules,slepc,petsc,archdir=''):
    if not self.requested:
      self.SkipInstall(slepcrules)
      return
    self.log.NewSection('Processing slepc4py...')

    # Check petsc4py module
    try:
      from petsc4py import PETSc
    except ImportError:
      self.log.Exit('Cannot import petsc4py, make sure your PYTHONPATH is set correctly')
    # Check for cython
    try:
      import Cython
    except ImportError:
      self.log.Exit('--with-slepc4py requires that cython is installed on your system')

    builddir = os.path.join(slepc.dir,'src','binding','slepc4py')
    destdir  = os.path.join(slepc.prefixdir,'lib')

    # add makefile rules
    envvars = 'PETSC_ARCH="" SLEPC_DIR=${SLEPC_INSTALLDIR}' if slepc.isinstall else ''
    confdir = os.path.join(destdir,'slepc','conf')
    rule =  'slepc4pybuild:\n'
    rule += '\t@echo "*** Building slepc4py ***"\n'
    rule += '\t@${RM} -f '+os.path.join(confdir,'slepc4py.errorflg')+'\n'
    rule += '\t@(cd '+builddir+' && \\\n'
    rule += '\t   %s ${PYTHON} setup.py clean --all && \\\n' % envvars
    rule += '\t   %s ${PYTHON} setup.py build ) > ' % envvars
    rule += os.path.join(confdir,'slepc4py.log')+' 2>&1 || \\\n'
    rule += '\t   (echo "**************************ERROR*************************************" && \\\n'
    rule += '\t   echo "Error building slepc4py. Check '+os.path.join(confdir,'slepc4py.log')+'" && \\\n'
    rule += '\t   echo "********************************************************************" && \\\n'
    rule += '\t   touch '+os.path.join(confdir,'slepc4py.errorflg')+' && \\\n'
    rule += '\t   exit 1)\n\n'
    slepcrules.write(rule)

    rule =  'slepc4pyinstall:\n'
    rule += '\t@echo "*** Installing slepc4py ***"\n'
    rule += '\t@(cd '+builddir+' && \\\n'
    rule += '\t   %s ${PYTHON} setup.py install --install-lib=%s) >> ' % (envvars,destdir)
    rule += os.path.join(confdir,'slepc4py.log')+' 2>&1 || \\\n'
    rule += '\t   (echo "**************************ERROR*************************************" && \\\n'
    rule += '\t   echo "Error building slepc4py. Check '+os.path.join(confdir,'slepc4py.log')+'" && \\\n'
    rule += '\t   echo "********************************************************************" && \\\n'
    rule += '\t   exit 1)\n'
    rule += '\t@echo "====================================="\n'
    rule += '\t@echo "To use slepc4py, add '+destdir+' to PYTHONPATH"\n'
    rule += '\t@echo "====================================="\n\n'
    slepcrules.write(rule)

    if slepc.isinstall:
      slepcrules.write('slepc4py-build:\n')
      slepcrules.write('slepc4py-install: slepc4pybuild slepc4pyinstall\n')
    else:
      slepcrules.write('slepc4py-build: slepc4pybuild slepc4pyinstall\n')
      slepcrules.write('slepc4py-install:\n')

    slepcconf.write('#define SLEPC_HAVE_SLEPC4PY 1\n')
    slepcconf.write('#define SLEPC4PY_INSTALL_PATH %s\n' % destdir)
    self.havepackage = True


  def SkipInstall(self,slepcrules):
    # add emtpy rules
    slepcrules.write('slepc4py-build:\n')
    slepcrules.write('slepc4py-install:\n')

