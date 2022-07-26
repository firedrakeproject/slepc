/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Tests SlepcInitialize() after PetscInitialize().\n\n";

#include <slepcsys.h>

int main(int argc,char **argv)
{
  PetscBool pInitialized,sInitialized,pFinalized,sFinalized,skip_petsc_finalize;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscInitialized(&pInitialized));
  PetscCall(SlepcInitialized(&sInitialized));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"PetscInitialized=%d, SlepcInitialized=%d.\n",(int)pInitialized,(int)sInitialized));
  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscInitialized(&pInitialized));
  PetscCall(SlepcInitialized(&sInitialized));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"PetscInitialized=%d, SlepcInitialized=%d.\n",(int)pInitialized,(int)sInitialized));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-skip_petsc_finalize",&skip_petsc_finalize));
  if (!skip_petsc_finalize) {
    PetscCall(PetscFinalize());
    PetscCall(PetscFinalized(&pFinalized));
    if (!pFinalized) printf("Unexpected value: PetscFinalized() returned False after PetscFinalize()\n");
  }
  PetscCall(SlepcFinalized(&sFinalized));
  if (sFinalized) printf("Unexpected value: SlepcFinalized() returned True before SlepcFinalize()\n");
  PetscCall(SlepcFinalize());
  PetscCall(SlepcFinalized(&sFinalized));
  if (!sFinalized) printf("Unexpected value: SlepcFinalized() returned False after SlepcFinalize()\n");
  if (skip_petsc_finalize) PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      output_file: output/test4_1.out
      test:
         suffix: 1
      test:
         suffix: 2
         args: -skip_petsc_finalize

TEST*/
