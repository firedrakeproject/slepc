/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Tests a user-defined convergence test in NEP.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = matrix dimension.\n";

/*
   Solve T(lambda)x=0 with T(lambda) = -D+sqrt(lambda)*I
      where D is the Laplacian operator in 1 dimension
*/

#include <slepcnep.h>

/*
  MyConvergedRel - Convergence test relative to the norm of D (given in ctx).
*/
PetscErrorCode MyConvergedRel(NEP nep,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscReal norm = *(PetscReal*)ctx;

  PetscFunctionBegin;
  *errest = res/norm;
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  NEP            nep;             /* nonlinear eigensolver context */
  Mat            A[2];
  PetscInt       n=100,Istart,Iend,i;
  PetscErrorCode ierr;
  PetscBool      terse;
  PetscReal      norm;
  FN             f[2];
  PetscScalar    coeffs;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nSquare root eigenproblem, n=%" PetscInt_FMT "\n\n",n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear eigensolver, define problem in split form
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(NEPCreate(PETSC_COMM_WORLD,&nep));

  /* Create matrices */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A[0]));
  CHKERRQ(MatSetSizes(A[0],PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A[0]));
  CHKERRQ(MatSetUp(A[0]));
  CHKERRQ(MatGetOwnershipRange(A[0],&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) CHKERRQ(MatSetValue(A[0],i,i-1,1.0,INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(A[0],i,i+1,1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(A[0],i,i,-2.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A[0],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A[0],MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreateConstantDiagonal(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,1.0,&A[1]));

  /* Define functions */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f[0]));
  CHKERRQ(FNSetType(f[0],FNRATIONAL));
  coeffs = 1.0;
  CHKERRQ(FNRationalSetNumerator(f[0],1,&coeffs));
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f[1]));
  CHKERRQ(FNSetType(f[1],FNSQRT));
  CHKERRQ(NEPSetSplitOperator(nep,2,A,f,SUBSET_NONZERO_PATTERN));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                   Set some options and solve
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(NEPSetTarget(nep,1.1));

  /* setup convergence test relative to the norm of D */
  CHKERRQ(MatNorm(A[0],NORM_1,&norm));
  CHKERRQ(NEPSetConvergenceTestFunction(nep,MyConvergedRel,&norm,NULL));

  CHKERRQ(NEPSetFromOptions(nep));
  CHKERRQ(NEPSolve(nep));

  /* show detailed info unless -terse option is given by user */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) CHKERRQ(NEPErrorView(nep,NEP_ERROR_BACKWARD,NULL));
  else {
    CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    CHKERRQ(NEPConvergedReasonView(nep,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(NEPErrorView(nep,NEP_ERROR_BACKWARD,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(NEPDestroy(&nep));
  CHKERRQ(MatDestroy(&A[0]));
  CHKERRQ(MatDestroy(&A[1]));
  CHKERRQ(FNDestroy(&f[0]));
  CHKERRQ(FNDestroy(&f[1]));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      args: -nep_type slp -nep_nev 2 -terse

TEST*/
