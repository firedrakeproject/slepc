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
    self.downloadable    = True
    self.url             = 'https://gitlab.com/slepc/slepc4py.git'
    self.builtafterslepc = True
    self.ProcessArgs(argdb)


  def DownloadOnly(self,slepcconf,slepcvars,slepcrules,slepc,petsc,archdir,prefixdir):
    externdir = slepc.CreateDir(archdir,'externalpackages')
    destdir   = os.path.join(prefixdir,'lib')

    # Check if source is already available
    builddir = os.path.join(externdir,'slepc4py')
    if os.path.exists(builddir):
      self.log.write('Using '+builddir)
    else: # clone slepc4py repo
      url = self.packageurl
      if url=='':
        url = self.url
      try:
        (result,output) = self.RunCommand('cd '+externdir+'&& git clone '+url)
      except RuntimeError as e:
        self.log.Exit('Cannot clone '+url+': '+str(e))

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

