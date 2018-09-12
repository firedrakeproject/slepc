from __future__ import print_function
#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

import argdb, os, sys, commands

class SLEPc:

  def __init__(self,argdb,log):
    self.log       = log
    self.clean     = argdb.PopBool('with-clean')
    self.cmake     = argdb.PopBool('with-cmake')
    self.prefixdir = argdb.PopPath('prefix')[0]
    self.isinstall = not self.prefixdir==''
    self.datadir   = argdb.PopPath('DATAFILESPATH')[0]

  def ShowHelp(self):
    print('''SLEPc:
  --with-clean=<bool>          : Delete prior build files including externalpackages
  --with-cmake=<bool>          : Enable builds with CMake (disabled by default)
  --prefix=<dir>               : Specify location to install SLEPc (e.g., /usr/local)
  --DATAFILESPATH=<dir>        : Specify location of datafiles (for SLEPc developers)''')

  def InitDir(self):
    if 'SLEPC_DIR' in os.environ:
      self.dir = os.environ['SLEPC_DIR']
      if not os.path.exists(self.dir) or not os.path.exists(os.path.join(self.dir,'config')):
        sys.exit('ERROR: SLEPC_DIR enviroment variable is not valid')
      if os.path.realpath(os.getcwd()) != os.path.realpath(self.dir):
        sys.exit('ERROR: SLEPC_DIR is not the current directory')
    else:
      self.dir = os.getcwd()
      if not os.path.exists(os.path.join(self.dir,'config')):
        sys.exit('ERROR: Current directory is not valid')

  def LoadVersion(self):
    try:
      f = open(os.path.join(self.dir,'include','slepcversion.h'))
      for l in f.readlines():
        l = l.split()
        if len(l) == 3:
          if l[1] == 'SLEPC_VERSION_RELEASE':
            self.release = l[2]
          if l[1] == 'SLEPC_VERSION_MAJOR':
            major = l[2]
          elif l[1] == 'SLEPC_VERSION_MINOR':
            minor = l[2]
          elif l[1] == 'SLEPC_VERSION_SUBMINOR':
            subminor = l[2]
      f.close()
      self.version = major + '.' + minor
      self.lversion = major + '.' + minor + '.' + subminor
      self.nversion = int(major)*100 + int(minor)
    except:
      self.log.Exit('ERROR: file error while reading SLEPc version')

    # Check whether this is a working copy of the repository
    self.isrepo = False
    if os.path.exists(os.path.join(self.dir,'src','docs')) and os.path.exists(os.path.join(self.dir,'.git')):
      (status, output) = commands.getstatusoutput('git rev-parse')
      if status:
        print('WARNING: SLEPC_DIR appears to be a git working copy, but git is not found in PATH')
      else:
        self.isrepo = True
        (status, self.gitrev) = commands.getstatusoutput('git describe')
        if not self.gitrev:
          (status, self.gitrev) = commands.getstatusoutput('git log -1 --pretty=format:%H')
        (status, self.gitdate) = commands.getstatusoutput('git log -1 --pretty=format:%ci')
        (status, self.branch) = commands.getstatusoutput('git describe --contains --all HEAD')

