/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

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
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetString(NULL,NULL,"-pkg",pkg,sizeof(pkg),NULL);CHKERRQ(ierr);
  ierr = SlepcHasExternalPackage(pkg,&has);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "SLEPc has %s? %s\n",pkg,PetscBools[has]);CHKERRQ(ierr);
  ierr = PetscStrcmp(pkg,"arpack",&flg);CHKERRQ(ierr);
#if defined(SLEPC_HAVE_ARPACK)
  if (flg && !has) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"SlepcHasExternalPackage() says ARPACK is not configured but SLEPC_HAVE_ARPACK is defined");
#else
  if (flg && has)  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"SlepcHasExternalPackage() says ARPACK is configured but SLEPC_HAVE_ARPACK is undefined");
#endif
  ierr = PetscStrcmp(pkg,"primme",&flg);CHKERRQ(ierr);
#if defined(SLEPC_HAVE_PRIMME)
  if (flg && !has) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"SlepcHasExternalPackage() says PRIMME is not configured but SLEPC_HAVE_PRIMME is defined");
#else
  if (flg && has)  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"SlepcHasExternalPackage() says PRIMME is configured but SLEPC_HAVE_PRIMME is undefined");
#endif
  ierr = SlepcFinalize();
  return ierr;
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
