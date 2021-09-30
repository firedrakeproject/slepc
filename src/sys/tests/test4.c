/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Tests SlepcInitialize() after PetscInitialize().\n\n";

#include <slepcsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscBool      pInitialized,sInitialized,pFinalized,sFinalized,skip_petsc_finalize;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscInitialized(&pInitialized);CHKERRQ(ierr);
  ierr = SlepcInitialized(&sInitialized);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"PetscInitialized=%d, SlepcInitialized=%d.\n",pInitialized,sInitialized);CHKERRQ(ierr);
  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = PetscInitialized(&pInitialized);CHKERRQ(ierr);
  ierr = SlepcInitialized(&sInitialized);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"PetscInitialized=%d, SlepcInitialized=%d.\n",pInitialized,sInitialized);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-skip_petsc_finalize",&skip_petsc_finalize);CHKERRQ(ierr);
  if (!skip_petsc_finalize) {
    ierr = PetscFinalize();if (ierr) return ierr;
    ierr = PetscFinalized(&pFinalized);if (ierr) return ierr;
    if (!pFinalized) printf("Unexpected value: PetscFinalized() returned False after PetscFinalize()\n");
  }
  ierr = SlepcFinalized(&sFinalized);if (ierr) return ierr;
  if (sFinalized) printf("Unexpected value: SlepcFinalized() returned True before SlepcFinalize()\n");
  ierr = SlepcFinalize();
  if (ierr) printf("SlepcFinalize() returned with error code %d\n",ierr);
  ierr = SlepcFinalized(&sFinalized);
  if (!sFinalized) printf("Unexpected value: SlepcFinalized() returned False after SlepcFinalize()\n");
  return ierr;
}

/*TEST

   testset:
      output_file: output/test4_1.out
      test:
         suffix: 1
      test:
         suffix: 2
         args: -skip_petsc_finalize
         TODO: shows an mpiexec error message in C++ jobs

TEST*/
