/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Tests SlepcHasExternalPackage().\n\n";

#include <slepcsys.h>

int main(int argc,char **argv)
{
  char           pkg[128] = "arpack";
  PetscBool      has,flg;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-pkg",pkg,sizeof(pkg),NULL));
  PetscCall(SlepcHasExternalPackage(pkg,&has));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "SLEPc has %s? %s\n",pkg,PetscBools[has]));
  PetscCall(PetscStrcmp(pkg,"arpack",&flg));
#if defined(SLEPC_HAVE_ARPACK)
  PetscCheck(!flg || has,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"SlepcHasExternalPackage() says ARPACK is not configured but SLEPC_HAVE_ARPACK is defined");
#else
  PetscCheck(!flg || !has,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"SlepcHasExternalPackage() says ARPACK is configured but SLEPC_HAVE_ARPACK is undefined");
#endif
  PetscCall(PetscStrcmp(pkg,"primme",&flg));
#if defined(SLEPC_HAVE_PRIMME)
  PetscCheck(!flg || has,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"SlepcHasExternalPackage() says PRIMME is not configured but SLEPC_HAVE_PRIMME is defined");
#else
  PetscCheck(!flg || !has,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"SlepcHasExternalPackage() says PRIMME is configured but SLEPC_HAVE_PRIMME is undefined");
#endif
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: arpack
      args: -pkg arpack
      requires: arpack
   test:
      suffix: no-arpack
      args: -pkg arpack
      requires: !arpack
   test:
      suffix: primme
      args: -pkg primme
      requires: primme
   test:
      suffix: no-primme
      args: -pkg primme
      requires: !primme

TEST*/
