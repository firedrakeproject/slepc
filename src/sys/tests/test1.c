/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

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

  CHKERRQ(SlepcInitialized(&isInitialized));
  if (!isInitialized) {
    CHKERRQ(SlepcInitializeNoArguments());
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Initialize SLEPc.\n"));
    CHKERRQ(SlepcInitialized(&isInitialized));
    CHKERRQ(SlepcFinalized(&isFinalized));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"SlepcInitialized=%d, SlepcFinalized=%d.\n",(int)isInitialized,(int)isFinalized));
  } else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"SLEPc was already initialized.\n"));
  CHKERRQ(SlepcFinalize());
  CHKERRQ(SlepcFinalized(&isFinalized));
  if (!isFinalized) printf("Unexpected value: SlepcFinalized() returned False after SlepcFinalize()\n");
  return 0;
}

/*TEST

   test:
      suffix: 1

TEST*/
