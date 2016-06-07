#!/usr/bin/env python
import os, sys, shutil, commands

try:
  WindowsError
except NameError:
  WindowsError = None

class Installer:
  def __init__(self, args = None):
    if len(args)<6:
      print '********************************************************************'
      print 'Installation script error - not enough arguments:'
      print './config/install.py SLEPC_DIR PETSC_DIR PETSC_ARCH DESTDIR LIB_SUFFIX RANLIB'
      print '********************************************************************'
      sys.exit(1)
    self.rootDir     = args[0]
    self.petscDir    = args[1]
    self.destDir     = os.path.abspath(args[2])
    self.arch        = args[3]
    self.arLibSuffix = args[4]
    self.ranlib      = ' '.join(args[5:])
    self.copies = []
    return

  def setupDirectories(self):
    self.installDir        = self.destDir
    self.archDir           = os.path.join(self.rootDir, self.arch)
    self.rootIncludeDir    = os.path.join(self.rootDir, 'include')
    self.archIncludeDir    = os.path.join(self.rootDir, self.arch, 'include')
    self.rootConfDir       = os.path.join(self.rootDir, 'lib','slepc','conf')
    self.archConfDir       = os.path.join(self.rootDir, self.arch, 'lib','slepc','conf')
    self.rootBinDir        = os.path.join(self.rootDir, 'bin')
    self.archBinDir        = os.path.join(self.rootDir, self.arch, 'bin')
    self.archLibDir        = os.path.join(self.rootDir, self.arch, 'lib')
    self.destIncludeDir    = os.path.join(self.destDir, 'include')
    self.destConfDir       = os.path.join(self.destDir, 'lib','slepc','conf')
    self.destLibDir        = os.path.join(self.destDir, 'lib')
    self.destBinDir        = os.path.join(self.destDir, 'bin')
    self.installIncludeDir = os.path.join(self.installDir, 'include')
    self.installBinDir     = os.path.join(self.installDir, 'bin')
    self.rootShareDir      = os.path.join(self.rootDir, 'share')
    self.destShareDir      = os.path.join(self.destDir, 'share')
    return

  def checkDestdir(self):
    if os.path.exists(self.destDir):
      if os.path.samefile(self.destDir, self.rootDir):
        print '********************************************************************'
        print 'Incorrect prefix usage. Specified destDir same as current SLEPC_DIR'
        print '********************************************************************'
        sys.exit(1)
      if os.path.samefile(self.destDir, os.path.join(self.rootDir,self.arch)):
        print '********************************************************************'
        print 'Incorrect prefix usage. Specified destDir same as current SLEPC_DIR/PETSC_ARCH'
        print '********************************************************************'
        sys.exit(1)
      if not os.path.isdir(os.path.realpath(self.destDir)):
        print '********************************************************************'
        print 'Specified destDir', self.destDir, 'is not a directory. Cannot proceed!'
        print '********************************************************************'
        sys.exit(1)
      if not os.access(self.destDir, os.W_OK):
        print '********************************************************************'
        print 'Unable to write to ', self.destDir, 'Perhaps you need to do "sudo make install"'
        print '********************************************************************'
        sys.exit(1)
    return

  def copytree(self, src, dst, symlinks = False, copyFunc = shutil.copy2, exclude = []):
    """Recursively copy a directory tree using copyFunc, which defaults to shutil.copy2().

       The copyFunc() you provide is only used on the top level, lower levels always use shutil.copy2

    The destination directory must not already exist.
    If exception(s) occur, an shutil.Error is raised with a list of reasons.

    If the optional symlinks flag is true, symbolic links in the
    source tree result in symbolic links in the destination tree; if
    it is false, the contents of the files pointed to by symbolic
    links are copied.
    """
    copies = []
    names  = os.listdir(src)
    if not os.path.exists(dst):
      os.makedirs(dst)
    elif not os.path.isdir(dst):
      raise shutil.Error, 'Destination is not a directory'
    errors = []
    for name in names:
      srcname = os.path.join(src, name)
      dstname = os.path.join(dst, name)
      try:
        if symlinks and os.path.islink(srcname):
          linkto = os.readlink(srcname)
          os.symlink(linkto, dstname)
        elif os.path.isdir(srcname):
          copies.extend(self.copytree(srcname, dstname, symlinks,exclude = exclude))
        elif not (os.path.basename(srcname) in exclude or os.path.splitext(os.path.basename(srcname))[1]=='.html'):
          copyFunc(srcname, dstname)
          copies.append((srcname, dstname))
        # XXX What about devices, sockets etc.?
      except (IOError, os.error), why:
        errors.append((srcname, dstname, str(why)))
      # catch the Error from the recursive copytree so that we can
      # continue with other files
      except shutil.Error, err:
        errors.extend((srcname,dstname,str(err.args[0])))
    try:
      shutil.copystat(src, dst)
    except OSError, e:
      if WindowsError is not None and isinstance(e, WindowsError):
        # Copying file access times may fail on Windows
        pass
      else:
        errors.extend((src, dst, str(e)))
    if errors:
      raise shutil.Error, errors
    return copies


  def fixConfFile(self, src):
    lines   = []
    oldFile = open(src, 'r')
    for line in oldFile.readlines():
      # paths generated by configure could be different link-path than whats used by user, so fix both
      line = line.replace(os.path.join(self.rootDir, self.arch), self.installDir)
      line = line.replace(os.path.realpath(os.path.join(self.rootDir, self.arch)), self.installDir)
      line = line.replace(os.path.join(self.rootDir, 'bin'), self.installBinDir)
      line = line.replace(os.path.realpath(os.path.join(self.rootDir, 'bin')), self.installBinDir)
      line = line.replace(os.path.join(self.rootDir, 'include'), self.installIncludeDir)
      line = line.replace(os.path.realpath(os.path.join(self.rootDir, 'include')), self.installIncludeDir)
      # remove SLEPC_DIR/PETSC_ARCH variables from conf-makefiles. They are no longer necessary
      line = line.replace('${SLEPC_DIR}/${PETSC_ARCH}', self.installDir)
      line = line.replace('PETSC_ARCH=${PETSC_ARCH}', '')
      line = line.replace('${SLEPC_DIR}', self.installDir)
      lines.append(line)
    oldFile.close()
    newFile = open(src, 'w')
    newFile.write(''.join(lines))
    newFile.close()
    return

  def fixConf(self):
    import shutil
    for file in ['slepc_rules', 'slepc_variables','slepcrules', 'slepcvariables']:
      self.fixConfFile(os.path.join(self.destConfDir,file))
    self.fixConfFile(os.path.join(self.destLibDir,'pkgconfig','SLEPc.pc'))
    return

  def createUninstaller(self):
    uninstallscript = os.path.join(self.destConfDir, 'uninstall.py')
    f = open(uninstallscript, 'w')
    # Could use the Python AST to do this
    f.write('#!'+sys.executable+'\n')
    f.write('import os\n')

    f.write('copies = '+repr(self.copies).replace(self.destDir,self.installDir))
    f.write('''
for src, dst in copies:
  try:
    os.remove(dst)
  except:
    pass
''')
    #TODO: need to delete libXXX.YYY.dylib.dSYM directory on Mac
    dirs = [os.path.join('include','slepc','finclude'),os.path.join('include','slepc','private'),os.path.join('lib','slepc','conf')]
    newdirs = []
    for dir in dirs: newdirs.append(os.path.join(self.installDir,dir))
    f.write('dirs = '+str(newdirs))
    f.write('''
for dir in dirs:
  import shutil
  try:
    shutil.rmtree(dir)
  except:
    pass
''')
    f.close()
    os.chmod(uninstallscript,0744)
    return

  def installIncludes(self):
    # TODO: should exclude slepc/finclude except for fortran builds
    self.copies.extend(self.copytree(self.rootIncludeDir, self.destIncludeDir,exclude = ['makefile']))
    self.copies.extend(self.copytree(self.archIncludeDir, self.destIncludeDir))
    return

  def installConf(self):
    self.copies.extend(self.copytree(self.rootConfDir, self.destConfDir, exclude = ['gmakegen.py','install.py']))
    self.copies.extend(self.copytree(self.archConfDir, self.destConfDir))
    return

  def installBin(self):
    #if os.path.exists(self.rootBinDir):
    #  self.copies.extend(self.copytree(self.rootBinDir, self.destBinDir))
    #if os.path.exists(self.archBinDir):
    #  self.copies.extend(self.copytree(self.archBinDir, self.destBinDir))
    return

  def installShare(self):
    self.copies.extend(self.copytree(self.rootShareDir, self.destShareDir))
    return

  def copyLib(self, src, dst):
    '''Run ranlib on the destination library if it is an archive. Also run install_name_tool on dylib on Mac'''
    # Symlinks (assumed local) are recreated at dst
    if os.path.islink(src):
      linkto = os.readlink(src)
      try:
        os.remove(dst)            # In case it already exists
      except OSError:
        pass
      os.symlink(linkto, dst)
      return
    # Do not install object files
    if not os.path.splitext(src)[1] == '.o':
      shutil.copy2(src, dst)
    if os.path.splitext(dst)[1] == '.'+self.arLibSuffix:
      (result, output) = commands.getstatusoutput(self.ranlib+' '+dst)
    if os.path.splitext(dst)[1] == '.dylib' and os.path.isfile('/usr/bin/install_name_tool'):
      (result, output) = commands.getstatusoutput('otool -D '+src)
      oldname = output[output.find("\n")+1:]
      installName = oldname.replace(self.archDir, self.installDir)
      (result, output) = commands.getstatusoutput('/usr/bin/install_name_tool -id ' + installName + ' ' + dst)
    # preserve the original timestamps - so that the .a vs .so time order is preserved
    shutil.copystat(src,dst)
    return

  def installLib(self):
    self.copies.extend(self.copytree(self.archLibDir, self.destLibDir, copyFunc = self.copyLib, exclude = ['.DIR']))
    return


  def outputInstallDone(self):
    print '''\
====================================
Install complete.
Now to check if the libraries are working do (in current directory):
make SLEPC_DIR=%s PETSC_DIR=%s PETSC_ARCH="" test
====================================\
''' % (self.installDir,self.petscDir)
    return

  def outputDestDirDone(self):
    print '''\
====================================
Copy to DESTDIR %s is now complete.
Before use - please copy/install over to specified prefix: %s
====================================\
''' % (self.destDir,self.installDir)
    return

  def runsetup(self):
    self.setupDirectories()
    self.checkDestdir()
    return

  def runcopy(self):
    if self.destDir == self.installDir:
      print '*** Installing SLEPc at prefix location:',self.destDir, ' ***'
    else:
      print '*** Copying SLEPc to DESTDIR location:',self.destDir, ' ***'
    if not os.path.exists(self.destDir):
      try:
        os.makedirs(self.destDir)
      except:
        print '********************************************************************'
        print 'Unable to create', self.destDir, 'Perhaps you need to do "sudo make install"'
        print '********************************************************************'
        sys.exit(1)
    self.installIncludes()
    self.installConf()
    self.installBin()
    self.installLib()
    self.installShare()
    return

  def runfix(self):
    self.fixConf()
    return

  def rundone(self):
    self.createUninstaller()
    if self.destDir == self.installDir:
      self.outputInstallDone()
    else:
      self.outputDestDirDone()
    return

  def run(self):
    self.runsetup()
    self.runcopy()
    self.runfix()
    self.rundone()
    return

if __name__ == '__main__':
  Installer(sys.argv[1:]).run()
