# Author:  Lisandro Dalcin
# Contact: dalcinl@gmail.com
import os
import sys
import optparse
import unittest

__unittest = True

components = [
    'PETSc',
    'SLEPc',
]


def getoptionparser():
    parser = optparse.OptionParser()

    parser.add_option("-q", "--quiet",
                      action="store_const", const=0, dest="verbose", default=1,
                      help="do not print status messages to stdout")
    parser.add_option("-v", "--verbose",
                      action="store_const", const=2, dest="verbose", default=1,
                      help="print status messages to stdout")
    parser.add_option("-i", "--include", type="string",
                      action="append",  dest="include", default=[],
                      help="include tests matching PATTERN", metavar="PATTERN")
    parser.add_option("-e", "--exclude", type="string",
                      action="append", dest="exclude", default=[],
                      help="exclude tests matching PATTERN", metavar="PATTERN")
    parser.add_option("-k", "--pattern", type="string",
                      action="append", dest="patterns", default=[],
                      help="only run tests which match the given substring")
    parser.add_option("-f", "--failfast",
                      action="store_true", dest="failfast", default=False,
                      help="Stop on first failure")
    parser.add_option("--no-builddir",
                      action="store_false", dest="builddir", default=True,
                      help="disable testing from build directory")
    parser.add_option("--path", type="string",
                      action="append", dest="path", default=[],
                      help="prepend PATH to sys.path", metavar="PATH")
    parser.add_option("--arch", type="string",
                      action="store", dest="arch", default=None,
                      help="use PETSC_ARCH",
                      metavar="PETSC_ARCH")
    parser.add_option("-s","--summary",
                      action="store_true", dest="summary", default=0,
                      help="print PETSc log summary")
    parser.add_option("--no-memdebug",
                      action="store_false", dest="memdebug", default=True,
                      help="Do not use PETSc memory debugging")
    return parser


def getbuilddir():
    try:
        try:
            from setuptools.dist import Distribution
        except ImportError:
            from distutils.dist import Distribution
        try:
            from setuptools.command.build import build
        except ImportError:
            from distutils.command.build import build
        cmd_obj = build(Distribution())
        cmd_obj.finalize_options()
        return cmd_obj.build_platlib
    except Exception:
        return None


def getprocessorinfo():
    try:
        name = os.uname()[1]
    except:
        import platform
        name = platform.uname()[1]
    from petsc4py.PETSc import COMM_WORLD
    rank = COMM_WORLD.getRank()
    return (rank, name)


def getlibraryinfo(name):
    modname = "%s4py.%s" % (name.lower(), name)
    module = __import__(modname, fromlist=[name])
    (major, minor, micro), devel = module.Sys.getVersion(devel=True)
    r = not devel
    if r: release = 'release'
    else: release = 'development'
    arch = module.__arch__
    return (
        "%s %d.%d.%d %s (conf: '%s')" %
        (name, major, minor, micro, release, arch)
    )


def getpythoninfo():
    x, y, z = sys.version_info[:3]
    return ("Python %d.%d.%d (%s)" % (x, y, z, sys.executable))


def getpackageinfo(pkg):
    try:
        pkg = __import__(pkg)
    except ImportError:
        return None
    name = pkg.__name__
    version = pkg.__version__
    path = pkg.__path__[0]
    return ("%s %s (%s)" % (name, version, path))


def setup_python(options):
    rootdir = os.path.dirname(os.path.dirname(__file__))
    builddir = os.path.join(rootdir, getbuilddir())
    if options.builddir and os.path.exists(builddir):
        sys.path.insert(0, builddir)
    if options.path:
        path = options.path[:]
        path.reverse()
        for p in path:
            sys.path.insert(0, p)


def setup_unittest(options):
    from unittest import TestSuite
    try:
        from unittest.runner import _WritelnDecorator
    except ImportError:
        from unittest import _WritelnDecorator
    #
    writeln_orig = _WritelnDecorator.writeln
    def writeln(self, message=''):
        try: self.stream.flush()
        except: pass
        writeln_orig(self, message)
        try: self.stream.flush()
        except: pass
    _WritelnDecorator.writeln = writeln


def import_package(options, pkgname):
    args = [sys.argv[0]]
    if options.memdebug:
        args.append('-malloc_debug')
        args.append('-malloc_dump')
    if options.summary:
        args.append('-log_view')
    package = __import__(pkgname)
    package.init(args, arch=options.arch)


def print_banner(options):
    r, n = getprocessorinfo()
    prefix = "[%d@%s]" % (r, n)

    def writeln(message='', endl='\n'):
        if message is None:
            return
        from petsc4py.PETSc import Sys
        message = "%s %s" % (prefix, message)
        Sys.syncPrint(message, endl=endl, flush=True)

    if options.verbose:
        writeln(getpythoninfo())
        writeln(getpackageinfo('numpy'))
        for entry in components:
            writeln(getlibraryinfo(entry))
            writeln(getpackageinfo('%s4py' % entry.lower()))


def load_tests(options, args):
    from glob import glob
    import re
    testsuitedir = os.path.dirname(__file__)
    sys.path.insert(0, testsuitedir)
    pattern = 'test_*.py'
    wildcard = os.path.join(testsuitedir, pattern)
    testfiles = glob(wildcard)
    testfiles.sort()
    testsuite = unittest.TestSuite()
    testloader = unittest.TestLoader()
    if options.patterns:
        testloader.testNamePatterns = [
            ('*%s*' % p) if ('*' not in p) else p
            for p in options.patterns
        ]
    include = exclude = None
    if options.include:
        include = re.compile('|'.join(options.include)).search
    if options.exclude:
        exclude = re.compile('|'.join(options.exclude)).search
    for testfile in testfiles:
        filename = os.path.basename(testfile)
        testname = os.path.splitext(filename)[0]
        if ((exclude and exclude(testname)) or
            (include and not include(testname))):
            continue
        module = __import__(testname)
        for arg in args:
            try:
                cases = testloader.loadTestsFromNames((arg,), module)
                testsuite.addTests(cases)
            except AttributeError:
                pass
        if not args:
            cases = testloader.loadTestsFromModule(module)
            testsuite.addTests(cases)
    return testsuite


def run_tests(options, testsuite, runner=None):
    if runner is None:
        runner = unittest.TextTestRunner(verbosity=options.verbose)
        runner.failfast = options.failfast
    result = runner.run(testsuite)
    return result.wasSuccessful()



def abort(code=1):
    os.abort()


def shutdown(success):
    pass


def main(args=None):
    pkgname = '%s4py' % components[-1].lower()
    parser = getoptionparser()
    (options, args) = parser.parse_args(args)
    setup_python(options)
    setup_unittest(options)
    import_package(options, pkgname)
    print_banner(options)
    testsuite = load_tests(options, args)
    success = run_tests(options, testsuite)
    if not success and options.failfast: abort()
    shutdown(success)
    return not success


if __name__ == '__main__':
    import sys
    sys.dont_write_bytecode = True
    sys.exit(main())
