#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

slepcMansecs    = ["eps","lme","mfn","nep","pep","svd","sys"]
slepcClassIdMap = {
  "_p_BV *"     : "BV_CLASSID",
  "_p_DS *"     : "DS_CLASSID",
  "_p_FN *"     : "FN_CLASSID",
  "_p_RG *"     : "RG_CLASSID",
  "_p_ST *"     : "ST_CLASSID",
  "_p_EPS *"    : "EPS_CLASSID",
  "_p_PEP *"    : "PEP_CLASSID",
  "_p_NEP *"    : "NEP_CLASSID",
  "_p_SVD *"    : "SVD_CLASSID",
  "_p_MFN *"    : "MFN_CLASSID",
  "_p_LME *"    : "LME_CLASSID",
}

def main(slepcDir,petscDir,petscArch,clangDir=None,clangLib=None,verbose=False,multiproc=True,maxWorkers=0,mansecs=slepcMansecs,checkFunctionFilter=None,applyPatches=False):
  extraCompilerFlags = [ '-I'+os.path.join(slepcDir,'include'), '-I'+os.path.join(slepcDir,petscArch,'include') ]
  extraHeaderIncludes = []
  mansecimpls = [m+"impl.h" for m in slepcMansecs]+["slepcimpl.h","vecimplslepc.h"]
  for headerFile in os.listdir(os.path.join(slepcDir,"include","slepc","private")):
    if headerFile in mansecimpls:
      extraHeaderIncludes.append("#include <slepc/private/{}>".format(headerFile))
  petscClangLinter.main(petscDir,petscArch,altBaseDir=slepcDir,clangDir=clangDir,clangLib=clangLib,verbose=verbose,multiproc=multiproc,maxWorkers=maxWorkers,mansecs=mansecs,checkFunctionFilter=checkFunctionFilter,applyPatches=applyPatches,extraCompilerFlags=extraCompilerFlags,extraHeaderIncludes=extraHeaderIncludes)


if __name__ == "__main__":
  import sys,argparse

  try:
    slepcDir = os.environ["SLEPC_DIR"]
  except KeyError:
    slepcDir = None
  try:
    petscDir = os.environ["PETSC_DIR"]
  except KeyError:
    petscDir = None
  try:
    petscArch = os.environ["PETSC_ARCH"]
  except KeyError:
    petscArch = None

  sys.path.insert(0, os.path.join(petscDir,'lib','petsc','bin','maint'))
  import petscClangLinter
  clangDir = petscClangLinter.tryToFindLibclangDir()
  petscClangLinter.classIdMap.update(slepcClassIdMap)

  parser = argparse.ArgumentParser(description="set options for clang static analysis tool",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  grouplibclang = parser.add_argument_group(title="libclang location settings")
  group = grouplibclang.add_mutually_exclusive_group(required=False)
  group.add_argument("--clang_dir",nargs="?",help="directory containing libclang.[so|dylib|dll], if not given attempts to automatically detect it via llvm-config",default=clangDir,dest="clangdir")
  group.add_argument("--clang_lib",nargs="?",help="direct location of libclang.[so|dylib|dll], overrides clang directory if set",dest="clanglib")
  grouppetsc = parser.add_argument_group(title="petsc/slepc location settings")
  grouppetsc.add_argument("--SLEPC_DIR",required=False,default=slepcDir,help="if this option is unused defaults to environment variable $SLEPC_DIR",dest="slepcdir")
  grouppetsc.add_argument("--PETSC_DIR",required=False,default=petscDir,help="if this option is unused defaults to environment variable $PETSC_DIR",dest="petscdir")
  grouppetsc.add_argument("--PETSC_ARCH",required=False,default=petscArch,help="if this option is unused defaults to environment variable $PETSC_ARCH",dest="petscarch")
  parser.add_argument("--verbose",required=False,action="store_true",help="verbose progress printed to screen")
  parser.add_argument("--show-warnings",required=False,action="store_true",help="show ast matching warnings",dest="warn")
  filterFuncChoices = ", ".join(list(petscClangLinter.checkFunctionMap.keys()))
  parser.add_argument("--filter-functions",required=False,nargs="+",choices=list(petscClangLinter.checkFunctionMap.keys()),metavar="FUNCTIONNAME",help="filter to display errors only related to list of provided function names, default is all functions. Choose from available function names: "+filterFuncChoices,dest="filterfunc")
  mansecChoices = ", ".join(slepcMansecs)
  parser.add_argument("--filter-mansec",required=False,nargs="+",default=slepcMansecs,choices=slepcMansecs,metavar="MANSEC",help="run only over specified mansecs (defaults to all), choose from: "+mansecChoices,dest="filtermansec")
  parser.add_argument("--no-multiprocessing",required=False,action="store_false",help="use multiprocessing",dest="multiproc")
  parser.add_argument("--jobs",required=False,type=int,default=0,nargs="?",help="number of multiprocessing jobs, 0 defaults to number of processors on machine")
  parser.add_argument("--apply-patches",required=False,action="store_true",help="apply patches automatically instead of saving to file",dest="apply")
  args = parser.parse_args()

  if args.slepcdir is None:
    raise RuntimeError("Could not determine SLEPC_DIR from environment, please set via options")
  if args.petscdir is None:
    raise RuntimeError("Could not determine PETSC_DIR from environment, please set via options")
  if args.petscarch is None:
    raise RuntimeError("Could not determine PETSC_ARCH from environment, please set via options")

  if args.clanglib:
    args.clangdir = None

  if args.verbose:
    args.warn = True
  ret = main(args.slepcdir,args.petscdir,args.petscarch,clangDir=args.clangdir,clangLib=args.clanglib,verbose=args.verbose,multiproc=args.multiproc,maxWorkers=args.jobs,mansecs=args.filtermansec,checkFunctionFilter=args.filterfunc,applyPatches=args.apply)
  sys.exit(ret)
