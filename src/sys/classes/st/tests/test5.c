/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  N = n*n;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create non-symmetric matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    d = 0.0;
    if (i>0) { PetscCall(MatSetValue(A,II,II-n,1.0,INSERT_VALUES)); d=d+1.0; }
    if (i<n-1) { PetscCall(MatSetValue(A,II,II+n,-1.0,INSERT_VALUES)); d=d+1.0; }
    if (j>0) { PetscCall(MatSetValue(A,II,II-1,1.0,INSERT_VALUES)); d=d+1.0; }
    if (j<n-1) { PetscCall(MatSetValue(A,II,II+1,-1.0,INSERT_VALUES)); d=d+1.0; }
    PetscCall(MatSetValue(A,II,II,d,INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateVecs(A,&v,&w));
  PetscCall(VecSetValue(v,0,-.5,INSERT_VALUES));
  PetscCall(VecSetValue(v,1,1.5,INSERT_VALUES));
  PetscCall(VecSetValue(v,2,2,INSERT_VALUES));
  PetscCall(VecAssemblyBegin(v));
  PetscCall(VecAssemblyEnd(v));
  PetscCall(VecView(v,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create the spectral transformation object
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(STCreate(PETSC_COMM_WORLD,&st));
  mat[0] = A;
  PetscCall(STSetMatrices(st,1,mat));
  PetscCall(STSetType(st,STCAYLEY));
  PetscCall(STSetShift(st,2.0));
  PetscCall(STCayleySetAntishift(st,1.0));
  PetscCall(STSetTransform(st,PETSC_TRUE));

  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetType(ksp,KSPPREONLY));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCLU));
  PetscCall(KSPSetTolerances(ksp,100*PETSC_MACHINE_EPSILON,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(STSetKSP(st,ksp));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(STSetFromOptions(st));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Apply the operator
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(STApply(st,v,w));
  PetscCall(VecView(w,NULL));

  PetscCall(STApplyTranspose(st,v,w));
  PetscCall(VecView(w,NULL));

  PetscCall(STMatMult(st,1,v,w));
  PetscCall(VecView(w,NULL));

  PetscCall(STMatMultTranspose(st,1,v,w));
  PetscCall(VecView(w,NULL));

  PetscCall(STMatSolve(st,v,w));
  PetscCall(VecView(w,NULL));

  PetscCall(STMatSolveTranspose(st,v,w));
  PetscCall(VecView(w,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Get the operator matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(STGetOperator(st,&Op));
  PetscCall(MatMult(Op,v,w));
  PetscCall(VecView(w,NULL));
  PetscCall(MatMultTranspose(Op,v,w));
  PetscCall(VecView(w,NULL));
  PetscCall(STRestoreOperator(st,&Op));

  PetscCall(STDestroy(&st));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&w));
  PetscCall(SlepcFinalize());
  return 0;
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
