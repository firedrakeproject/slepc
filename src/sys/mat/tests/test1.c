/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"MatCreateTile test, n=%" PetscInt_FMT "\n",n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create T=tridiag([-1 2 -1],n,n) and E=eye(n)
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&T));
  PetscCall(MatSetSizes(T,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(T));
  PetscCall(MatSetUp(T));

  PetscCall(MatGetOwnershipRange(T,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) PetscCall(MatSetValue(T,i,i-1,-1.0,INSERT_VALUES));
    if (i<n-1) PetscCall(MatSetValue(T,i,i+1,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(T,i,i,2.0,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(T,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(T,MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&E));
  PetscCall(MatSetSizes(E,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(E));
  PetscCall(MatSetUp(E));

  PetscCall(MatGetOwnershipRange(E,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) PetscCall(MatSetValue(E,i,i,1.0,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(E,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(E,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create tiled matrix A = [ 2*T -E; 0 3*T ]
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreateTile(2.0,T,-1.0,E,0.0,E,3.0,T,&A));
  PetscCall(MatView(A,NULL));

  PetscCall(MatDestroy(&T));
  PetscCall(MatDestroy(&E));
  PetscCall(MatDestroy(&A));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 1

   test:
      suffix: 2
      nsize: 2

TEST*/
