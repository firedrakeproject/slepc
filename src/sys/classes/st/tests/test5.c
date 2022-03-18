/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test various ST interface functions.\n\n";

#include <slepceps.h>

int main (int argc,char **argv)
{
  ST             st;
  KSP            ksp;
  PC             pc;
  Mat            A,mat[1],Op;
  Vec            v,w;
  PetscInt       N,n=4,i,j,II,Istart,Iend;
  PetscScalar    d;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  N = n*n;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create non-symmetric matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    d = 0.0;
    if (i>0) { CHKERRQ(MatSetValue(A,II,II-n,1.0,INSERT_VALUES)); d=d+1.0; }
    if (i<n-1) { CHKERRQ(MatSetValue(A,II,II+n,-1.0,INSERT_VALUES)); d=d+1.0; }
    if (j>0) { CHKERRQ(MatSetValue(A,II,II-1,1.0,INSERT_VALUES)); d=d+1.0; }
    if (j<n-1) { CHKERRQ(MatSetValue(A,II,II+1,-1.0,INSERT_VALUES)); d=d+1.0; }
    CHKERRQ(MatSetValue(A,II,II,d,INSERT_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCreateVecs(A,&v,&w));
  CHKERRQ(VecSetValue(v,0,-.5,INSERT_VALUES));
  CHKERRQ(VecSetValue(v,1,1.5,INSERT_VALUES));
  CHKERRQ(VecSetValue(v,2,2,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(v));
  CHKERRQ(VecAssemblyEnd(v));
  CHKERRQ(VecView(v,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create the spectral transformation object
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(STCreate(PETSC_COMM_WORLD,&st));
  mat[0] = A;
  CHKERRQ(STSetMatrices(st,1,mat));
  CHKERRQ(STSetType(st,STCAYLEY));
  CHKERRQ(STSetShift(st,2.0));
  CHKERRQ(STCayleySetAntishift(st,1.0));
  CHKERRQ(STSetTransform(st,PETSC_TRUE));

  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetType(ksp,KSPPREONLY));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCLU));
  CHKERRQ(KSPSetTolerances(ksp,100*PETSC_MACHINE_EPSILON,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(STSetKSP(st,ksp));
  CHKERRQ(KSPDestroy(&ksp));

  CHKERRQ(STSetFromOptions(st));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Apply the operator
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(STApply(st,v,w));
  CHKERRQ(VecView(w,NULL));

  CHKERRQ(STApplyTranspose(st,v,w));
  CHKERRQ(VecView(w,NULL));

  CHKERRQ(STMatMult(st,1,v,w));
  CHKERRQ(VecView(w,NULL));

  CHKERRQ(STMatMultTranspose(st,1,v,w));
  CHKERRQ(VecView(w,NULL));

  CHKERRQ(STMatSolve(st,v,w));
  CHKERRQ(VecView(w,NULL));

  CHKERRQ(STMatSolveTranspose(st,v,w));
  CHKERRQ(VecView(w,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Get the operator matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(STGetOperator(st,&Op));
  CHKERRQ(MatMult(Op,v,w));
  CHKERRQ(VecView(w,NULL));
  CHKERRQ(MatMultTranspose(Op,v,w));
  CHKERRQ(VecView(w,NULL));
  CHKERRQ(STRestoreOperator(st,&Op));

  CHKERRQ(STDestroy(&st));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&w));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      output_file: output/test5_1.out
      requires: !single
      test:
         args: -st_matmode {{copy inplace}}
      test:
         args: -st_matmode shell -ksp_type bcgs -pc_type jacobi

TEST*/
