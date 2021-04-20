#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:05:39 2021

@author: jacobfaibussowitsch
"""
import os,sys
try:
  import clang.cindex as clx
except ModuleNotFoundError as mnfe:
  if mnfe.name == "clang":
    raise RuntimeError("Must run e.g. 'pip install clang' to use linter") from mnfe

"""
clang.cindex.TranslationUnit does not have all latest flags, but we prefix
with P_ just in case

see: https://clang.llvm.org/doxygen/group__CINDEX__TRANSLATION__UNIT.html#gab1e4965c1ebe8e41d71e90203a723fe9
"""
P_CXTranslationUnit_None                                 = 0x0
P_CXTranslationUnit_DetailedPreprocessingRecord          = 0x01
P_CXTranslationUnit_Incomplete                           = 0x02
P_CXTranslationUnit_PrecompiledPreamble                  = 0x04
P_CXTranslationUnit_CacheCompletionResults               = 0x08
P_CXTranslationUnit_ForSerialization                     = 0x10
P_CXTranslationUnit_SkipFunctionBodies                   = 0x40
P_CXTranslationUnit_IncludeBriefCommentsInCodeCompletion = 0x80
P_CXTranslationUnit_CreatePreambleOnFirstParse           = 0x100
P_CXTranslationUnit_KeepGoing                            = 0x200
P_CXTranslationUnit_SingleFileParse                      = 0x400
P_CXTranslationUnit_LimitSkipFunctionBodiesToPreamble    = 0x800
P_CXTranslationUnit_IncludeAttributedTypes               = 0x1000
P_CXTranslationUnit_VisitImplicitAttributes              = 0x2000
P_CXTranslationUnit_IgnoreNonErrorsFromIncludedFiles     = 0x4000
P_CXTranslationUnit_RetainExcludedConditionalBlocks      = 0x8000

slepcmansecs    = ["eps","lme","mfn","nep","pep","svd","sys"]
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

def main(slepcDir,petscDir,petscArch,clangDir=None,clangLib=None,verbose=False,multiproc=True,mansecFilter=None,checkFunctionFilter=None,printWarnings=False,maxWorkers=0,applyPatches=False):
  if not clx.conf.loaded:
    clx.conf.set_compatibility_check(True)
    if clangLib:
      clangLib = os.path.abspath(os.path.expanduser(os.path.expandvars(clangLib)))
      clx.conf.set_library_file(clangLib)
    elif clangDir:
      clangDir = os.path.abspath(os.path.expanduser(os.path.expandvars(clangDir)))
      clx.conf.set_library_path(clangDir)
    else:
      raise RuntimeError("Must supply either clang directory path or clang library path")

  rootPrintPrefix  = "[ROOT]"
  pchClangOptions  = (P_CXTranslationUnit_CreatePreambleOnFirstParse |
                      P_CXTranslationUnit_Incomplete |
                      P_CXTranslationUnit_ForSerialization)
  baseClangOptions = (P_CXTranslationUnit_PrecompiledPreamble |
                      P_CXTranslationUnit_SkipFunctionBodies |
                      P_CXTranslationUnit_LimitSkipFunctionBodiesToPreamble)
  miscFlags        = ["-x","c++","-Wno-nullability-completeness"]
  sysincludes      = petscClangLinter.getClangSysIncludes()
  extraIncludes    = petscClangLinter.getPetscExtraIncludes(petscDir,petscArch)
  extraIncludesSle = [ '-I'+os.path.join(slepcDir,'include'), '-I'+os.path.join(slepcDir,petscArch,'include') ]
  compilerFlags    = sysincludes+miscFlags+extraIncludes+extraIncludesSle
  if verbose: print("\n".join([rootPrintPrefix+" Compile flags:",*compilerFlags]))

  # create a precompiled header from petsc.h, and all of the major "impl" headers,
  # this saves a lot of time since this includes almost every sub-header in petsc.
  # Including petsc.h first should define almost everything we need so no side effects
  # from including headers in the wrong order below
  mansecimpls     = [m+"impl.h" for m in petscClangLinter.mansecs]+["isimpl.h","dtimpl.h","dmpleximpl.h","petscfeimpl.h","dmlabelimpl.h","sfimpl.h","viewerimpl.h","characteristicimpl.h"]
  megaHeaderLines = ["#include <petscastfix.hpp>","#include <petsc.h>"]
  for headerFile in os.listdir(os.path.join(petscDir,"include","petsc","private")):
    if headerFile in mansecimpls or headerFile.startswith(("hash","pc")):
      megaHeaderLines.append("#include <petsc/private/{}>".format(headerFile))
  slepcmansecimpls = [m+"impl.h" for m in slepcmansecs]+["slepcimpl.h","vecimplslepc.h"]
  for headerFile in os.listdir(os.path.join(slepcDir,"include","slepc","private")):
    if headerFile in slepcmansecimpls:
      megaHeaderLines.append("#include <slepc/private/{}>".format(headerFile))
  index = clx.Index.create()
  megaHeader = "\n".join(megaHeaderLines)+"\n" # extra newline for last line
  petscPrecompiledHeader = os.path.join(slepcDir,"include","slepc_ast_precompile.h.pch")
  if verbose:
    print("\n".join([rootPrintPrefix+" Mega header:",megaHeader]))
    print(rootPrintPrefix,"Creating precompiled header",petscPrecompiledHeader)
  tu = index.parse("megaHeader.hpp",args=compilerFlags,unsaved_files=[("megaHeader.hpp",megaHeader)],options=pchClangOptions)
  if tu.diagnostics:
    print("\n".join(map(str,tu.diagnostics)))
    raise clx.LibclangError("Warnings generated when creating the precompiled header. This usually means that the provided libclang is faulty")
  petscClangLinter.osRemoveSilent(petscPrecompiledHeader)
  tu.save(petscPrecompiledHeader)
  compilerFlags.extend(["-include-pch",petscPrecompiledHeader])

  if multiproc:
    import multiprocessing as mp
    # -1 since num workers+root = numCpu
    if not maxWorkers: maxWorkers = max(mp.cpu_count()-1,1)
    if maxWorkers == 1:
      multiproc = False
      print(rootPrintPrefix,"Number of processes ({}) too small. Not using multiprocessing".format(maxWorkers))
  # need a second "if multiproc" since the above might turn multiproc off. it might have
  # been a great place to use "goto" to jump to the else but since thats apparently far
  # too complex a construct according to python devs we do this stupid song and dance
  if multiproc:
    # get the library file to pass to subprocesses
    clangLib = clx.conf.get_filename()
    fileProcessorQueue = mp.JoinableQueue(3*maxWorkers)
    exceptionSignalQueue = mp.Queue()
    dataQueue = mp.Queue()
    fileProcessorLock = mp.Lock()
    workerArgs = (clangLib,checkFunctionFilter,os.path.join(slepcDir,"include"),slepcClassIdMap,compilerFlags,baseClangOptions,verbose,printWarnings,exceptionSignalQueue,dataQueue,fileProcessorQueue,fileProcessorLock,)
    for i in range(maxWorkers):
      workerName = "[{}]".format(i)
      worker = mp.Process(target=petscClangLinter.queueMain,args=workerArgs,name=workerName,daemon=True)
      worker.start()
    # need these later for error printing
    errBars = "".join(["[ERROR]",85*"-","[ERROR]\n"])
    errBars = [errBars,errBars]
  else:
    # apply the filters if we aren't using workers
    updateCheckFunctionMap(checkFunctionFilter)
    updatePetscClassIdMap(slepcDir)

  # always update mansecs
  petscClangLinter.updateMansecs(mansecFilter)
  # exclude these directories
  excludeDirs = {"f90-mod","f90-src","f90-custom","output","input","python","fsrc","ftn-auto","ftn-custom","f2003-src","ftn-kernels","tests","tutorials"}
  excludeDirSuffixes = (".dSYM",)
  # allow these file suffixes
  allowFileSuffixes = (".c",".cpp",".cxx",".cu")
  warnings,errorsLeft,diffs = [],[],[]
  for mansec in slepcmansecs:
    workdir = os.path.join(slepcDir,'src',mansec)
    for root,dirs,files in os.walk(workdir):
      if verbose: print(rootPrintPrefix,"Processing directory",root)
      dirs[:] = [d for d in dirs if d not in excludeDirs]
      dirs[:] = [d for d in dirs if not d.endswith(excludeDirSuffixes)]
      files   = [os.path.join(root,f) for f in files if f.endswith(allowFileSuffixes)]
      if multiproc:
        for filename in files:
          fileProcessorQueue.put(filename)
      else:
        for filename in files:
          if verbose: print(rootPrintPrefix,"Processing file     ",filename)
          tu = index.parse(filename,args=compilerFlags,options=baseClangOptions)
          if tu.diagnostics and (verbose or printWarnings):
            diags = {" ".join([rootPrintPrefix,d]) for d in map(str,tu.diagnostics)}
            print("\n".join(diags))
          with petscClangLinter.BadSource(rootPrintPrefix,printWarningMessages=printWarnings) as badSource:
            for func,parent in petscClangLinter.findFunctionCallExpr(tu,petscClangLinter.checkFunctionMap.keys()):
              petscClangLinter.checkFunctionMap[func.spelling](badSource,func,parent)
            diffs.extend(badSource.coalesceDiffs())
            errorsLeft.append(badSource.getErrorsLeft())
            warnings.append(badSource.getAllWarnings())
    if multiproc:
      stopMultiproc = False
      # join here to colocate error messages to a mansec
      fileProcessorQueue.join()
      while not exceptionSignalQueue.empty():
        exception = exceptionSignalQueue.get()
        errMess   = str(exception).join(errBars)
        print(errMess)
        stopMultiproc = True
      if stopMultiproc: raise RuntimeError("Error in child process detected")
  if multiproc:
    # send stop-signal to child processes
    for _ in range(maxWorkers):
      fileProcessorQueue.put(petscClangLinter.QueueSignal.EXIT_QUEUE)
    fileProcessorQueue.close()
    # wait for queue to close
    fileProcessorQueue.join()
    exceptionSignalQueue.close()
    while not dataQueue.empty():
      signal,returnData = dataQueue.get()
      if signal == petscClangLinter.QueueSignal.ERRORS_LEFT:
        errorsLeft.append(returnData)
      elif signal == petscClangLinter.QueueSignal.UNIFIED_DIFF:
        diffs.extend(returnData)
      elif signal == petscClangLinter.QueueSignal.WARNING:
        warnings.append(returnData)
    dataQueue.close()
  if verbose: print(rootPrintPrefix,"Deleting precompiled header",petscPrecompiledHeader)
  petscClangLinter.osRemoveSilent(petscPrecompiledHeader)
  errorsLeft = [e for e in errorsLeft if e] # remove any None's
  warnings   = [w for w in warnings if w]
  diffs      = [d for d in diffs if d]
  if diffs:
    import time

    srcDir,patchDir = os.path.join(slepcDir,"src"),os.path.join(slepcDir,"slepcLintPatches")
    try:
      os.mkdir(patchDir)
    except FileExistsError:
      pass
    manglePostfix = "".join(["_",str(int(time.time())),".patch"])
    for filename,diff in diffs:
      filename    = filename.replace(srcDir,"").replace(os.path.sep,"_")[1:]
      mangledFile = filename.split(".")[0]+manglePostfix
      mangledFile = os.path.join(patchDir,mangledFile)
      if verbose: print(rootPrintPrefix,"Writing patch to file",mangledFile)
      with open(mangledFile,"w") as fd:
        fd.write(diff)
    if applyPatches:
      import subprocess,glob

      if verbose: print(rootPrintPrefix,"Applying patches from patch directory",patchDir)
      rootDir   = "".join(["-d",os.path.abspath(os.path.sep)])
      patchGlob = "".join([patchDir,os.path.sep,"*",manglePostfix])
      for patch in glob.iglob(patchGlob):
        if verbose: print(rootPrintPrefix,"Applying patch",patch)
        if sys.version_info >= (3,7):
          output = subprocess.run(["patch",rootDir,"-p0","--unified","-i",patch],check=True,universal_newlines=True,capture_output=True)
        else:
          output = subprocess.run(["patch",rootDir,"-p0","--unified","-i",patch],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,universal_newlines=True)
        if verbose: print(output.stdout)
  returnCode = 0
  if errorsLeft or (warnings and verbose):
    print(rootPrintPrefix,27*"=","UNCORRECTABLE SOURCE BEGIN",30*"=")
    if warnings and verbose:
      print("\n".join(warnings))
    if errorsLeft:
      print("\n".join(errorsLeft))
      returnCode = 1
    print(rootPrintPrefix,27*"=","UNCORRECTABLE SOURCE END",32*"=")
    print(rootPrintPrefix,"Some errors or warnings could not be automatically corrected via the patch files, see above")
  elif diffs:
    if applyPatches:
      print(rootPrintPrefix,"All errors or warnings successfully patched")
    else:
      print(rootPrintPrefix,"All errors fixable via patch files written to",patchDir)
      returnCode = 2
  return returnCode


if __name__ == "__main__":
  import argparse

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
  petscClangLinter.petscClassIdMap.update(slepcClassIdMap)

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
  mansecChoices = ", ".join(slepcmansecs)
  parser.add_argument("--filter-mansec",required=False,nargs="+",choices=slepcmansecs,metavar="MANSEC",help="run only over specified mansecs (defaults to all), choose from: "+mansecChoices,dest="filtermansec")
  parser.add_argument("--no-multiprocessing",required=False,action="store_false",help="use multiprocessing",dest="multiproc")
  parser.add_argument("--jobs",required=False,type=int,default=0,nargs="?",help="number of multiprocessing jobs, 0 defaults to number of processors on machine")
  parser.add_argument("--apply-patches",required=False,action="store_true",help="apply patches automatically instead of saving to file",dest="apply")
  args = parser.parse_args()

  if args.petscdir is None:
    raise RuntimeError("Could not determine PETSC_DIR from environment, please set via options")
  if args.petscarch is None:
    raise RuntimeError("Could not determine PETSC_ARCH from environment, please set via options")

  if args.clanglib:
    args.clangdir = None

  if args.verbose:
    args.warn = True
  ret = main(args.slepcdir,args.petscdir,args.petscarch,clangDir=args.clangdir,clangLib=args.clanglib,verbose=args.verbose,multiproc=args.multiproc,mansecFilter=args.filtermansec,checkFunctionFilter=args.filterfunc,printWarnings=args.warn,maxWorkers=args.jobs,applyPatches=args.apply)
  sys.exit(ret)
