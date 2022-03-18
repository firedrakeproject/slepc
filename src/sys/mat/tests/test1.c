/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test MatCreateTile.\n\n";

#include <slepcsys.h>

int main(int argc,char **argv)
{
  Mat            T,E,A;
  PetscInt       i,Istart,Iend,n=10;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatCreateTile test, n=%" PetscInt_FMT "\n",n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create T=tridiag([-1 2 -1],n,n) and E=eye(n)
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&T));
  CHKERRQ(MatSetSizes(T,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(T));
  CHKERRQ(MatSetUp(T));

  CHKERRQ(MatGetOwnershipRange(T,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) CHKERRQ(MatSetValue(T,i,i-1,-1.0,INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(T,i,i+1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(T,i,i,2.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(T,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(T,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&E));
  CHKERRQ(MatSetSizes(E,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(E));
  CHKERRQ(MatSetUp(E));

  CHKERRQ(MatGetOwnershipRange(E,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    CHKERRQ(MatSetValue(E,i,i,1.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(E,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(E,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create tiled matrix A = [ 2*T -E; 0 3*T ]
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreateTile(2.0,T,-1.0,E,0.0,E,3.0,T,&A));
  CHKERRQ(MatView(A,NULL));

  CHKERRQ(MatDestroy(&T));
  CHKERRQ(MatDestroy(&E));
  CHKERRQ(MatDestroy(&A));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      nsize: 1

   test:
      suffix: 2
      nsize: 2

TEST*/
