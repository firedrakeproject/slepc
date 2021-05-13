#!/usr/bin/env python
# Author:  Lisandro Dalcin
# Contact: dalcinl@gmail.com

"""
SLEPc for Python
"""

import sys
import os
import re

try:
    import setuptools
except ImportError:
    setuptools = None

pyver = sys.version_info[:2]
if pyver < (2, 6) or (3, 0) <= pyver < (3, 2):
    raise RuntimeError("Python version 2.6, 2.7 or >= 3.2 required")
if pyver == (2, 6) or pyver == (3, 2):
    sys.stderr.write(
        "WARNING: Python %d.%d is not supported.\n" % pyver)

# --------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------

topdir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, topdir)

from conf.metadata import metadata

def name():
    return 'slepc4py'

def version():
    with open(os.path.join(topdir, 'src', '__init__.py')) as f:
        m = re.search(r"__version__\s*=\s*'(.*)'", f.read())
        return m.groups()[0]

def description():
    with open(os.path.join(topdir, 'DESCRIPTION.rst')) as f:
        return f.read()

name     = name()
version  = version()

url      = 'https://gitlab.com/slepc/slepc'
pypiroot = 'https://pypi.io/packages/source/%s/%s/' % (name[0], name)
download = pypiroot + '%(name)s-%(version)s.tar.gz' % vars()

devstat  = ['Development Status :: 5 - Production/Stable']
keywords = ['SLEPc', 'PETSc', 'MPI']

metadata['name'] = name
metadata['version'] = version
metadata['description'] = __doc__.strip()
metadata['long_description'] = description()
metadata['keywords'] += keywords
metadata['classifiers'] += devstat
metadata['url'] = url
metadata['download_url'] = download

metadata['provides'] = ['slepc4py']
metadata['requires'] = ['petsc4py']

# --------------------------------------------------------------------
# Extension modules
# --------------------------------------------------------------------

def get_ext_modules(Extension):
    from os import walk
    from glob import glob
    from os.path import join
    glob_join = lambda *args: glob(join(*args))
    depends = []
    for pth, dirs, files in walk('src'):
        depends += glob_join(pth, '*.h')
        depends += glob_join(pth, '*.c')
    if 'SLEPC_DIR' in os.environ:
        pd = os.environ['SLEPC_DIR']
        pa = os.environ.get('PETSC_ARCH', '')
        depends += glob_join(pd, 'include', '*.h')
        depends += glob_join(pd, 'include', 'slepc', 'private', '*.h')
        depends += glob_join(pd, pa, 'include', 'slepcconf.h')
    if 'PETSC_DIR' in os.environ:
        pd = os.environ['PETSC_DIR']
        pa = os.environ.get('PETSC_ARCH', '')
        depends += glob_join(pd, 'include', '*.h')
        depends += glob_join(pd, 'include', 'petsc', 'private', '*.h')
        depends += glob_join(pd, pa, 'include', 'petscconf.h')
    try:
        import petsc4py
        petsc4py_includes = [petsc4py.get_include()]
    except ImportError:
        petsc4py_includes = []
    return [Extension('slepc4py.lib.SLEPc',
                      sources=['src/SLEPc.c',],
                      include_dirs=['src/include',
                                    ] + petsc4py_includes,
                      depends=depends)]

# --------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------

from conf.slepcconf import setup, Extension
from conf.slepcconf import config, build, build_src, build_ext, install
from conf.slepcconf import clean, test, sdist

CYTHON = '0.24'

def run_setup():
    setup_args = metadata.copy()
    if setuptools:
        setup_args['zip_safe'] = False
        vstr = setup_args['version'].split('.')[:2]
        x, y = int(vstr[0]), int(vstr[1])
        PETSC = SLEPC = ">=%s.%s,<%s.%s" % (x, y, x, y+1)
        setup_args['install_requires'] = ['petsc4py'+PETSC]
        SLEPC_DIR = os.environ.get('SLEPC_DIR')
        if not (SLEPC_DIR and os.path.isdir(SLEPC_DIR)):
            setup_args['install_requires'] += ['slepc'+SLEPC]
    if setuptools:
        src = os.path.join('src', 'slepc4py.SLEPc.c')
        has_src = os.path.exists(os.path.join(topdir, src))
        has_git = os.path.isdir(os.path.join(topdir, '.git'))
        has_hg  = os.path.isdir(os.path.join(topdir, '.hg'))
        suffix = os.path.join('src', 'binding', 'slepc4py')
        in_slepc = topdir.endswith(os.path.sep + suffix)
        if not has_src or has_git or has_hg or in_slepc:
            setup_args['setup_requires'] = ['Cython>='+CYTHON]
    #
    setup(packages     = ['slepc4py',
                          'slepc4py.lib',],
          package_dir  = {'slepc4py'     : 'src',
                          'slepc4py.lib' : 'src/lib'},
          package_data = {'slepc4py'     : ['include/slepc4py/*.h',
                                            'include/slepc4py/*.i',
                                            'include/slepc4py/*.pxd',
                                            'include/slepc4py/*.pxi',
                                            'include/slepc4py/*.pyx',
                                            'SLEPc.pxd',],
                          'slepc4py.lib' : ['slepc.cfg'],},
          ext_modules  = get_ext_modules(Extension),
          cmdclass     = {'config'     : config,
                          'build'      : build,
                          'build_src'  : build_src,
                          'build_ext'  : build_ext,
                          'install'    : install,
                          'clean'      : clean,
                          'test'       : test,
                          'sdist'      : sdist,
                          },
          **setup_args)

def chk_cython(VERSION):
    from distutils import log
    from distutils.version import LooseVersion
    from distutils.version import StrictVersion
    warn = lambda msg='': sys.stderr.write(msg+'\n')
    #
    try:
        import Cython
    except ImportError:
        warn("*"*80)
        warn()
        warn(" You need to generate C source files with Cython!!")
        warn(" Download and install Cython <https://cython.org>")
        warn()
        warn("*"*80)
        return False
    #
    try:
        CYTHON_VERSION = Cython.__version__
    except AttributeError:
        from Cython.Compiler.Version import version as CYTHON_VERSION
    REQUIRED = VERSION
    m = re.match(r"(\d+\.\d+(?:\.\d+)?).*", CYTHON_VERSION)
    if m:
        Version = StrictVersion
        AVAILABLE = m.groups()[0]
    else:
        Version = LooseVersion
        AVAILABLE = CYTHON_VERSION
    if (REQUIRED is not None and
        Version(AVAILABLE) < Version(REQUIRED)):
        warn("*"*80)
        warn()
        warn(" You need to install Cython %s (you have version %s)"
             % (REQUIRED, CYTHON_VERSION))
        warn(" Download and install Cython <https://cython.org>")
        warn()
        warn("*"*80)
        return False
    #
    return True

def run_cython(source, target=None,
               depends=(), includes=(),
               destdir_c=None, destdir_h=None,
               wdir=None, force=False, VERSION=None):
    from glob import glob
    from distutils import log
    from distutils import dep_util
    from distutils.errors import DistutilsError
    if target is None:
        target = os.path.splitext(source)[0]+'.c'
    cwd = os.getcwd()
    try:
        if wdir: os.chdir(wdir)
        alldeps = [source]
        for dep in depends:
            alldeps += glob(dep)
        if not (force or dep_util.newer_group(alldeps, target)):
            log.debug("skipping '%s' -> '%s' (up-to-date)",
                      source, target)
            return
    finally:
        os.chdir(cwd)
    if not chk_cython(VERSION):
        raise DistutilsError("requires Cython>=%s" % VERSION)
    log.info("cythonizing '%s' -> '%s'", source, target)
    from conf.cythonize import cythonize
    err = cythonize(source, target,
                    includes=includes,
                    destdir_c=destdir_c,
                    destdir_h=destdir_h,
                    wdir=wdir)
    if err:
        raise DistutilsError(
            "Cython failure: '%s' -> '%s'" % (source, target))

def build_sources(cmd):
    from os.path import exists, isdir, join

    # slepc4py.SLEPc
    source = 'slepc4py.SLEPc.pyx'
    target = 'slepc4py.SLEPc.c'
    depends = ["include/*/*.pxd",
               "SLEPc/*.pyx",
               "SLEPc/*.pxi",]
    import petsc4py
    includes = ['include', petsc4py.get_include()]
    destdir_h = os.path.join('include', 'slepc4py')
    run_cython(source, target,
               depends=depends, includes=includes,
               destdir_c=None, destdir_h=destdir_h, wdir='src',
               force=cmd.force, VERSION=CYTHON)

build_src.run = build_sources

def run_testsuite(cmd):
    from distutils.errors import DistutilsError
    sys.path.insert(0, 'test')
    try:
        from runtests import main
    finally:
        del sys.path[0]
    if cmd.dry_run:
        return
    args = cmd.args[:] or []
    if cmd.verbose < 1:
        args.insert(0,'-q')
    if cmd.verbose > 1:
        args.insert(0,'-v')
    err = main(args)
    if err:
        raise DistutilsError("test")

test.run = run_testsuite

# --------------------------------------------------------------------

def main():
    run_setup()

if __name__ == '__main__':
    main()

# --------------------------------------------------------------------
