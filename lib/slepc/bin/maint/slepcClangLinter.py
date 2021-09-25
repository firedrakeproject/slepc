#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

slepcMansecs    = ["eps","lme","mfn","nep","pep","svd","sys"]
slepcAuxMansecs = ["bv","ds","fn","rg","st"]
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

def main(slepcDir,petscDir,petscArch,clangDir=None,clangLib=None,verbose=False,werror=False,maxWorkers=-1,checkFunctionFilter=None,applyPatches=False):
  extraCompilerFlags = [ '-I'+os.path.join(slepcDir,'include'), '-I'+os.path.join(slepcDir,petscArch,'include') ]
  with open(os.path.join(slepcDir,petscArch,"lib","slepc","conf","slepcvariables"),"r") as sv:
    line = sv.readline()
    while line:
      if line.find("INCLUDE")>-1:
        for inc in line.split("=",1)[1].split():
          extraCompilerFlags.append(inc)
      line = sv.readline()

  extraHeaderIncludes = []
  mansecimpls = [m+"impl.h" for m in slepcMansecs+slepcAuxMansecs]+["slepcimpl.h","vecimplslepc.h"]
  for headerFile in os.listdir(os.path.join(slepcDir,"include","slepc","private")):
    if headerFile in mansecimpls:
      extraHeaderIncludes.append("#include <slepc/private/{}>".format(headerFile))
  ret = petscClangLinter.main(petscDir,petscArch,srcDir=os.path.join(slepcDir,"src"),clangDir=clangDir,clangLib=clangLib,verbose=verbose,werror=werror,workers=maxWorkers,checkFunctionFilter=checkFunctionFilter,patchDir=os.path.join(slepcDir,"slepcLintPatches"),applyPatches=applyPatches,extraCompilerFlags=extraCompilerFlags,extraHeaderIncludes=extraHeaderIncludes)
  return ret


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
  parser.add_argument("-v","--verbose",required=False,action="store_true",help="verbose progress printed to screen")
  parser.add_argument("--werror",required=False,action="store_true",help="treat all warnings as errors")
  filterFuncChoices = ", ".join(list(petscClangLinter.checkFunctionMap.keys()))
  parser.add_argument("-f","--functions",required=False,nargs="+",choices=list(petscClangLinter.checkFunctionMap.keys()),metavar="FUNCTIONNAME",help="filter to display errors only related to list of provided function names, default is all functions. Choose from available function names: "+filterFuncChoices,dest="funcs")
  parser.add_argument("-j","--jobs",required=False,type=int,default=-1,nargs="?",help="number of multiprocessing jobs, -1 means number of processors on machine")
  parser.add_argument("-a","--apply-patches",required=False,action="store_true",help="automatically apply patches that are saved to file",dest="apply")
  args = parser.parse_args()

  if args.slepcdir is None:
    raise RuntimeError("Could not determine SLEPC_DIR from environment, please set via options")
  if args.petscdir is None:
    raise RuntimeError("Could not determine PETSC_DIR from environment, please set via options")
  if args.petscarch is None:
    raise RuntimeError("Could not determine PETSC_ARCH from environment, please set via options")

  if args.clanglib:
    args.clangdir = None

  ret = main(args.slepcdir,args.petscdir,args.petscarch,clangDir=args.clangdir,clangLib=args.clanglib,verbose=args.verbose,maxWorkers=args.jobs,checkFunctionFilter=args.funcs,applyPatches=args.apply)
  sys.exit(ret)
