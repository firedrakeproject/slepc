/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Tests multiple calls to NEPSolve() with different matrix size.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions.\n"
  "  -tau <tau>, where <tau> is the delay parameter.\n"
  "  -split <0/1>, to select the split form in the problem definition (enabled by default).\n";

/* Based on ex22.c (delay) */

#include <slepcnep.h>

/*
   User-defined application context
*/
typedef struct {
  PetscScalar tau;
  PetscReal   a;
} ApplicationCtx;

/*
   Create problem matrices in split form
*/
PetscErrorCode BuildSplitMatrices(PetscInt n,PetscReal a,Mat *Id,Mat *A,Mat *B)
{
  PetscInt       i,Istart,Iend;
  PetscReal      h,xi;
  PetscScalar    b;

  PetscFunctionBeginUser;
  h = PETSC_PI/(PetscReal)(n+1);

  /* Identity matrix */
  CHKERRQ(MatCreateConstantDiagonal(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,1.0,Id));
  CHKERRQ(MatSetOption(*Id,MAT_HERMITIAN,PETSC_TRUE));

  /* A = 1/h^2*tridiag(1,-2,1) + a*I */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,A));
  CHKERRQ(MatSetSizes(*A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(*A));
  CHKERRQ(MatSetUp(*A));
  CHKERRQ(MatGetOwnershipRange(*A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) CHKERRQ(MatSetValue(*A,i,i-1,1.0/(h*h),INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(*A,i,i+1,1.0/(h*h),INSERT_VALUES));
    CHKERRQ(MatSetValue(*A,i,i,-2.0/(h*h)+a,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatSetOption(*A,MAT_HERMITIAN,PETSC_TRUE));

  /* B = diag(b(xi)) */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,B));
  CHKERRQ(MatSetSizes(*B,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(*B));
  CHKERRQ(MatSetUp(*B));
  CHKERRQ(MatGetOwnershipRange(*B,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    xi = (i+1)*h;
    b = -4.1+xi*(1.0-PetscExpReal(xi-PETSC_PI));
    CHKERRQ(MatSetValue(*B,i,i,b,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatSetOption(*B,MAT_HERMITIAN,PETSC_TRUE));
  PetscFunctionReturn(0);
}

/*
   Compute Function matrix  T(lambda)
*/
PetscErrorCode FormFunction(NEP nep,PetscScalar lambda,Mat fun,Mat B,void *ctx)
{
  ApplicationCtx *user = (ApplicationCtx*)ctx;
  PetscInt       i,n,Istart,Iend;
  PetscReal      h,xi;
  PetscScalar    b;

  PetscFunctionBeginUser;
  CHKERRQ(MatGetSize(fun,&n,NULL));
  h = PETSC_PI/(PetscReal)(n+1);
  CHKERRQ(MatGetOwnershipRange(fun,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) CHKERRQ(MatSetValue(fun,i,i-1,1.0/(h*h),INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(fun,i,i+1,1.0/(h*h),INSERT_VALUES));
    xi = (i+1)*h;
    b = -4.1+xi*(1.0-PetscExpReal(xi-PETSC_PI));
    CHKERRQ(MatSetValue(fun,i,i,-lambda-2.0/(h*h)+user->a+PetscExpScalar(-user->tau*lambda)*b,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  if (fun != B) {
    CHKERRQ(MatAssemblyBegin(fun,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(fun,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

/*
   Compute Jacobian matrix  T'(lambda)
*/
PetscErrorCode FormJacobian(NEP nep,PetscScalar lambda,Mat jac,void *ctx)
{
  ApplicationCtx *user = (ApplicationCtx*)ctx;
  PetscInt       i,n,Istart,Iend;
  PetscReal      h,xi;
  PetscScalar    b;

  PetscFunctionBeginUser;
  CHKERRQ(MatGetSize(jac,&n,NULL));
  h = PETSC_PI/(PetscReal)(n+1);
  CHKERRQ(MatGetOwnershipRange(jac,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    xi = (i+1)*h;
    b = -4.1+xi*(1.0-PetscExpReal(xi-PETSC_PI));
    CHKERRQ(MatSetValue(jac,i,i,-1.0-user->tau*PetscExpScalar(-user->tau*lambda)*b,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  NEP            nep;             /* nonlinear eigensolver context */
  Mat            Id,A,B,J,F;      /* problem matrices */
  FN             f1,f2,f3;        /* functions to define the nonlinear operator */
  ApplicationCtx ctx;             /* user-defined context */
  Mat            mats[3];
  FN             funs[3];
  PetscScalar    coeffs[2];
  PetscInt       n=128;
  PetscReal      tau=0.001,a=20;
  PetscBool      split=PETSC_TRUE;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-tau",&tau,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-split",&split,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n1-D Delay Eigenproblem, n=%" PetscInt_FMT ", tau=%g\n\n",n,(double)tau));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
              Create nonlinear eigensolver and set options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(NEPCreate(PETSC_COMM_WORLD,&nep));
  CHKERRQ(NEPSetTolerances(nep,1e-9,PETSC_DEFAULT));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      First solve
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  if (split) {
    CHKERRQ(BuildSplitMatrices(n,a,&Id,&A,&B));
    /* f1=-lambda */
    CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f1));
    CHKERRQ(FNSetType(f1,FNRATIONAL));
    coeffs[0] = -1.0; coeffs[1] = 0.0;
    CHKERRQ(FNRationalSetNumerator(f1,2,coeffs));
    /* f2=1.0 */
    CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f2));
    CHKERRQ(FNSetType(f2,FNRATIONAL));
    coeffs[0] = 1.0;
    CHKERRQ(FNRationalSetNumerator(f2,1,coeffs));
    /* f3=exp(-tau*lambda) */
    CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f3));
    CHKERRQ(FNSetType(f3,FNEXP));
    CHKERRQ(FNSetScale(f3,-tau,1.0));
    mats[0] = A;  funs[0] = f2;
    mats[1] = Id; funs[1] = f1;
    mats[2] = B;  funs[2] = f3;
    CHKERRQ(NEPSetSplitOperator(nep,3,mats,funs,SUBSET_NONZERO_PATTERN));
  } else {
    /* callback form  */
    ctx.tau = tau;
    ctx.a   = a;
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&F));
    CHKERRQ(MatSetSizes(F,PETSC_DECIDE,PETSC_DECIDE,n,n));
    CHKERRQ(MatSetFromOptions(F));
    CHKERRQ(MatSeqAIJSetPreallocation(F,3,NULL));
    CHKERRQ(MatMPIAIJSetPreallocation(F,3,NULL,1,NULL));
    CHKERRQ(MatSetUp(F));
    CHKERRQ(NEPSetFunction(nep,F,F,FormFunction,&ctx));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&J));
    CHKERRQ(MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,n,n));
    CHKERRQ(MatSetFromOptions(J));
    CHKERRQ(MatSeqAIJSetPreallocation(J,3,NULL));
    CHKERRQ(MatMPIAIJSetPreallocation(F,3,NULL,1,NULL));
    CHKERRQ(MatSetUp(J));
    CHKERRQ(NEPSetJacobian(nep,J,FormJacobian,&ctx));
  }

  /* Set solver parameters at runtime */
  CHKERRQ(NEPSetFromOptions(nep));

  /* Solve the eigensystem */
  CHKERRQ(NEPSolve(nep));
  CHKERRQ(NEPErrorView(nep,NEP_ERROR_RELATIVE,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
               Second solve, with problem matrices of size 2*n
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  n *= 2;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n1-D Delay Eigenproblem, n=%" PetscInt_FMT ", tau=%g\n\n",n,(double)tau));
  if (split) {
    CHKERRQ(MatDestroy(&Id));
    CHKERRQ(MatDestroy(&A));
    CHKERRQ(MatDestroy(&B));
    CHKERRQ(BuildSplitMatrices(n,a,&Id,&A,&B));
    mats[0] = A;
    mats[1] = Id;
    mats[2] = B;
    CHKERRQ(NEPSetSplitOperator(nep,3,mats,funs,SUBSET_NONZERO_PATTERN));
  } else {
    /* callback form  */
    CHKERRQ(MatDestroy(&F));
    CHKERRQ(MatDestroy(&J));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&F));
    CHKERRQ(MatSetSizes(F,PETSC_DECIDE,PETSC_DECIDE,n,n));
    CHKERRQ(MatSetFromOptions(F));
    CHKERRQ(MatSeqAIJSetPreallocation(F,3,NULL));
    CHKERRQ(MatMPIAIJSetPreallocation(F,3,NULL,1,NULL));
    CHKERRQ(MatSetUp(F));
    CHKERRQ(NEPSetFunction(nep,F,F,FormFunction,&ctx));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&J));
    CHKERRQ(MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,n,n));
    CHKERRQ(MatSetFromOptions(J));
    CHKERRQ(MatSeqAIJSetPreallocation(J,3,NULL));
    CHKERRQ(MatMPIAIJSetPreallocation(F,3,NULL,1,NULL));
    CHKERRQ(MatSetUp(J));
    CHKERRQ(NEPSetJacobian(nep,J,FormJacobian,&ctx));
  }

  /* Solve the eigensystem */
  CHKERRQ(NEPSolve(nep));
  CHKERRQ(NEPErrorView(nep,NEP_ERROR_RELATIVE,NULL));

  CHKERRQ(NEPDestroy(&nep));
  if (split) {
    CHKERRQ(MatDestroy(&Id));
    CHKERRQ(MatDestroy(&A));
    CHKERRQ(MatDestroy(&B));
    CHKERRQ(FNDestroy(&f1));
    CHKERRQ(FNDestroy(&f2));
    CHKERRQ(FNDestroy(&f3));
  } else {
    CHKERRQ(MatDestroy(&F));
    CHKERRQ(MatDestroy(&J));
  }
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   testset:
      nsize: 2
      requires: !single
      output_file: output/test10_1.out
      test:
         suffix: 1
         args: -nep_type narnoldi -nep_target 0.55
      test:
         suffix: 1_rii
         args: -nep_type rii -nep_target 0.55 -nep_rii_hermitian -split {{0 1}}
      test:
         suffix: 1_narnoldi
         args: -nep_type narnoldi -nep_target 0.55 -nep_narnoldi_lag_preconditioner 2
      test:
         suffix: 1_slp
         args: -nep_type slp -nep_slp_st_pc_type redundant -split {{0 1}}
      test:
         suffix: 1_interpol
         args: -nep_type interpol -rg_type interval -rg_interval_endpoints .5,1,-.1,.1 -nep_target .7 -nep_interpol_st_pc_type redundant
      test:
         suffix: 1_narnoldi_sync
         args: -nep_type narnoldi -ds_parallel synchronized

   testset:
      args: -nep_nev 2 -rg_type interval -rg_interval_endpoints .5,15,-.1,.1 -nep_target .7
      requires: !single
      output_file: output/test10_2.out
      filter: sed -e "s/[+-]0\.0*i//g"
      test:
         suffix: 2_interpol
         args: -nep_type interpol -nep_interpol_pep_type jd -nep_interpol_st_pc_type sor
      test:
         suffix: 2_nleigs
         args: -nep_type nleigs -split {{0 1}}
         requires: complex
      test:
         suffix: 2_nleigs_real
         args: -nep_type nleigs -rg_interval_endpoints .5,15 -split {{0 1}}
         requires: !complex

   test:
      suffix: 3
      requires: complex !single
      args: -nep_type ciss -rg_type ellipse -rg_ellipse_center 10 -rg_ellipse_radius 9.5 -rg_ellipse_vscale 0.1 -split {{0 1}}

TEST*/
