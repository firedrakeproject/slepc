/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Demonstrates SlepcInitializeNoArguments()
*/

#include <slepcsys.h>

int main(int argc,char **argv)
{
  PetscBool      isInitialized,isFinalized;

  PetscCall(SlepcInitialized(&isInitialized));
  if (!isInitialized) {
    PetscCall(SlepcInitializeNoArguments());
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Initialize SLEPc.\n"));
    PetscCall(SlepcInitialized(&isInitialized));
    PetscCall(SlepcFinalized(&isFinalized));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"SlepcInitialized=%d, SlepcFinalized=%d.\n",(int)isInitialized,(int)isFinalized));
  } else PetscCall(PetscPrintf(PETSC_COMM_WORLD,"SLEPc was already initialized.\n"));
  PetscCall(SlepcFinalize());
  PetscCall(SlepcFinalized(&isFinalized));
  if (!isFinalized) printf("Unexpected value: SlepcFinalized() returned False after SlepcFinalize()\n");
  return 0;
}

/*TEST

   test:
      suffix: 1

TEST*/
