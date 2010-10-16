#!/usr/bin/env python

"""
SLEPc: Scalable Library for Eigenvalue Problem Computations
===========================================================

SLEPc is a software library for the solution of large scale sparse
eigenvalue problems on parallel computers. It is an extension of PETSc
and can be used for either standard or generalized eigenproblems, with
real or complex arithmetic. It can also be used for computing a
partial SVD of a large, sparse, rectangular matrix, and to solve
quadratic eigenvalue problems
"""

import sys, os
from distutils.core import setup
from distutils.util import get_platform
from distutils.spawn import find_executable
from distutils.command.build import build as _build
if 'setuptools' in sys.modules:
    from setuptools.command.install import install as _install
else:
    from distutils.command.install import install as _install
from distutils.command.sdist import sdist as _sdist
from distutils import log

init_py = """\
# Author:  SLEPc Team
# Contact: slepc-maint@grycap.upv.es

def get_slepc_dir():
    import os
    return os.path.dirname(__file__)

def get_config():
    conf = {}
    conf['SLEPC_DIR'] = get_slepc_dir()
    return conf
"""

metadata = {
    'provides' : ['slepc'],
    'requires' : [],
}

def bootstrap():
    # Set SLEPC_DIR
    SLEPC_DIR  = os.path.abspath(os.getcwd())
    os.environ['SLEPC_DIR']  = SLEPC_DIR
    if not os.environ.get('PETSC_DIR'):
        try: del os.environ['PETSC_ARCH']
        except KeyError: pass
    # Generate package __init__.py file
    from distutils.dir_util import mkpath
    pkgdir = os.path.join('config', 'pypi')
    pkgfile = os.path.join(pkgdir, '__init__.py')
    if not os.path.exists(pkgdir):
        mkpath(pkgdir)
    if not os.path.exists(pkgfile):
        open(pkgfile, 'wt').write(init_py)
    if not os.environ.get('PETSC_DIR'):
        if (('distribute' in sys.modules) or
            ('setuptools' in sys.modules)):
            metadata['install_requires']= ['petsc']
    if 'setuptools' in sys.modules:
        metadata['zip_safe'] = False

def get_petsc_dir():
    PETSC_DIR = os.environ.get('PETSC_DIR')
    if not PETSC_DIR:
        try:
            import petsc
            PETSC_DIR = petsc.get_petsc_dir()
        except ImportError:
            log.warn("PETSC_DIR not specified")
            PETSC_DIR = os.path.join(os.path.sep, 'usr', 'local', 'petsc')
    return PETSC_DIR

def get_petsc_arch():
    PETSC_ARCH = os.environ.get('PETSC_ARCH') or 'installed-petsc'
    return PETSC_ARCH

def config(dry_run=False):
    log.info('SLEPc: configure')
    if dry_run: return
    # PETSc
    # Run SLEPc configure
    os.environ['PETSC_DIR'] = get_petsc_dir()
    status = os.system(" ".join((
            find_executable('python'),
            os.path.join('config', 'configure.py'),
            )))
    if status != 0: raise RuntimeError(status)

def build(dry_run=False):
    log.info('SLEPc: build')
    if dry_run: return
    # Run SLEPc build
    status = os.system(" ".join((
            find_executable('make'),
            'PETSC_DIR='+get_petsc_dir(),
            'PETSC_ARCH='+get_petsc_arch(),
            'all',
            )))
    if status != 0: raise RuntimeError

def install(dest_dir, prefix=None, dry_run=False):
    log.info('SLEPc: install')
    if dry_run: return
    if prefix is None:
        prefix = dest_dir
    PETSC_ARCH = get_petsc_arch()
    # Run SLEPc install
    status = os.system(" ".join((
            find_executable('make'),
            'PETSC_DIR='+get_petsc_dir(),
            'PETSC_ARCH='+get_petsc_arch(),
            'SLEPC_DESTDIR='+dest_dir,
            'install',
            )))
    if status != 0: raise RuntimeError
    slepcvariables = os.path.join(dest_dir, 'conf', 'slepcvariables')
    open(slepcvariables, 'a').write('SLEPC_DESTDIR=%s\n' % prefix)

class context:
    def __init__(self):
        self.sys_argv = sys.argv[:]
        self.wdir = os.getcwd()
    def enter(self):
        del sys.argv[1:]
        pdir = os.environ['SLEPC_DIR']
        os.chdir(pdir)
        return self
    def exit(self):
        sys.argv[:] = self.sys_argv
        os.chdir(self.wdir)

class cmd_build(_build):

    def initialize_options(self):
        _build.initialize_options(self)
        PETSC_ARCH = os.environ.get('PETSC_ARCH', 'installed-petsc')
        self.build_base = os.path.join(PETSC_ARCH, 'build-python')

    def run(self):
        _build.run(self)
        ctx = context().enter()
        try:
            config(self.dry_run)
            build(self.dry_run)
        finally:
            ctx.exit()

class cmd_install(_install):

    def initialize_options(self):
        _install.initialize_options(self)
        self.optimize = 1

    def run(self):
        root_dir = self.install_platlib
        dest_dir = os.path.join(root_dir, 'slepc')
        bdist_base = self.get_finalized_command('bdist').bdist_base
        if dest_dir.startswith(bdist_base):
            prefix = dest_dir[len(bdist_base)+1:]
            prefix = prefix[prefix.index(os.path.sep):]
        else:
            prefix = dest_dir
        dest_dir = os.path.abspath(dest_dir)
        prefix   = os.path.abspath(prefix)
        #
        _install.run(self)
        ctx = context().enter()
        try:
            install(dest_dir, prefix, self.dry_run)
        finally:
            ctx.exit()

class cmd_sdist(_sdist):

    def initialize_options(self):
        _sdist.initialize_options(self)
        self.force_manifest = 1
        self.template = os.path.join('config', 'manifest.in')

def version():
    import re
    version_re = {
        'major'  : re.compile(r"#define\s+SLEPC_VERSION_MAJOR\s+(\d+)"),
        'minor'  : re.compile(r"#define\s+SLEPC_VERSION_MINOR\s+(\d+)"),
        'micro'  : re.compile(r"#define\s+SLEPC_VERSION_SUBMINOR\s+(\d+)"),
        'patch'  : re.compile(r"#define\s+SLEPC_VERSION_PATCH\s+(\d+)"),
        'release': re.compile(r"#define\s+SLEPC_VERSION_RELEASE\s+(\d+)"),
        }
    slepcversion_h = os.path.join('include','slepcversion.h')
    data = open(slepcversion_h, 'rt').read()
    major = int(version_re['major'].search(data).groups()[0])
    minor = int(version_re['minor'].search(data).groups()[0])
    micro = int(version_re['micro'].search(data).groups()[0])
    patch = int(version_re['patch'].search(data).groups()[0])
    release = int(version_re['release'].search(data).groups()[0])
    if release:
        v = "%d.%d" % (major, minor)
        if micro > 0:
            v += ".%d" % micro
        if patch > 0:
            v += ".post%d" % patch
    else:
        v = "%d.%d.dev%d" % (major, minor+1, 0)
    return v

def tarball():
    VERSION = version()
    if '.dev' in VERSION:
        return None
    if '.post' not in VERSION:
        VERSION = VERSION + '.post0'
    VERSION = VERSION.replace('.post', '-p')
    return ('http://www.grycap.upv.es/slepc/download/distrib/' +
            'slepc-%s.tar.gz' % VERSION)

description = __doc__.split('\n')[1:-1]; del description[1:3]
classifiers = """
License :: Public Domain
Operating System :: POSIX
Intended Audience :: Developers
Intended Audience :: Science/Research
Programming Language :: C
Programming Language :: C++
Programming Language :: Fortran
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries
"""

bootstrap()
setup(name='slepc',
      version=version(),
      description=description.pop(0),
      long_description='\n'.join(description),
      classifiers= classifiers.split('\n')[1:-1],
      keywords = ['SLEPc','PETSc', 'MPI'],
      platforms=['POSIX'],
      license='LGPL',

      url='http://www.grycap.upv.es/slepc/',
      download_url=tarball(),

      author='SLEPc Team',
      author_email='slepc-maint@grycap.upv.es',
      maintainer='Lisandro Dalcin',
      maintainer_email='dalcinl@gmail.com',

      packages = ['slepc'],
      package_dir = {'slepc': 'config/pypi'},
      cmdclass={
        'build': cmd_build,
        'install': cmd_install,
        'sdist': cmd_sdist,
        },
      **metadata)
