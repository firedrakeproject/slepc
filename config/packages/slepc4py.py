#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

from __future__ import print_function
import sys, os, log, package

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
      if not 'PYTHONPATH' in os.environ:
        pythonpath = os.path.join(petsc.dir,'lib') if petsc.isinstall else os.path.join(petsc.dir,petsc.arch,'lib')
        sys.path.insert(0,pythonpath)
      else:
        pythonpath = os.environ['PYTHONPATH']
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
    envvars = 'PYTHONPATH=%s ' % pythonpath
    if slepc.isinstall:
      envvars += 'PETSC_ARCH="" SLEPC_DIR=${DESTDIR}${SLEPC_INSTALLDIR} '
    confdir = os.path.join(destdir,'slepc','conf')
    rule =  'slepc4pybuild:\n'
    rule += '\t@echo "*** Building slepc4py ***"\n'
    rule += '\t@${RM} -f '+os.path.join(confdir,'slepc4py.errorflg')+'\n'
    rule += '\t@cd '+builddir+' && \\\n'
    rule += '\t   %s ${PYTHON} setup.py build 2>&1 || \\\n' % envvars
    rule += '\t   (echo "**************************ERROR*************************************" && \\\n'
    rule += '\t   echo "Error building slepc4py." && \\\n'
    rule += '\t   echo "********************************************************************" && \\\n'
    rule += '\t   touch '+os.path.join(confdir,'slepc4py.errorflg')+' && \\\n'
    rule += '\t   exit 1)\n\n'
    slepcrules.write(rule)

    rule =  'slepc4pyinstall:\n'
    rule += '\t@echo "*** Installing slepc4py ***"\n'
    rule += '\t@cd '+builddir+' && \\\n'
    rule += '\t   %s ${PYTHON} setup.py install --install-lib=%s \\\n' % (envvars,destdir)
    rule += '\t      $(if $(DESTDIR),--root=\'$(DESTDIR)\')'
    rule += '\t   2>&1 || \\\n'
    rule += '\t   (echo "**************************ERROR*************************************" && \\\n'
    rule += '\t   echo "Error installing slepc4py" && \\\n'
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

    rule =  'slepc4pytest:\n'
    rule += '\t@echo "*** Testing slepc4py ***"\n'
    rule += '\t@PYTHONPATH=%s:%s ${PYTHON} %s --verbose\n' % (destdir,pythonpath,os.path.join('src','binding','slepc4py','test','runtests.py'))
    rule += '\t@echo "====================================="\n\n'
    slepcrules.write(rule)

    slepcconf.write('#define SLEPC_HAVE_SLEPC4PY 1\n')
    slepcconf.write('#define SLEPC4PY_INSTALL_PATH %s\n' % destdir)
    self.havepackage = True


  def SkipInstall(self,slepcrules):
    # add emtpy rules
    slepcrules.write('slepc4py-build:\n')
    slepcrules.write('slepc4py-install:\n')
    slepcrules.write('slepc4pytest:\n')

