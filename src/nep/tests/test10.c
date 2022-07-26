/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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
  PetscCall(MatCreateConstantDiagonal(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,1.0,Id));
  PetscCall(MatSetOption(*Id,MAT_HERMITIAN,PETSC_TRUE));

  /* A = 1/h^2*tridiag(1,-2,1) + a*I */
  PetscCall(MatCreate(PETSC_COMM_WORLD,A));
  PetscCall(MatSetSizes(*A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(*A));
  PetscCall(MatSetUp(*A));
  PetscCall(MatGetOwnershipRange(*A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) PetscCall(MatSetValue(*A,i,i-1,1.0/(h*h),INSERT_VALUES));
    if (i<n-1) PetscCall(MatSetValue(*A,i,i+1,1.0/(h*h),INSERT_VALUES));
    PetscCall(MatSetValue(*A,i,i,-2.0/(h*h)+a,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(*A,MAT_HERMITIAN,PETSC_TRUE));

  /* B = diag(b(xi)) */
  PetscCall(MatCreate(PETSC_COMM_WORLD,B));
  PetscCall(MatSetSizes(*B,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(*B));
  PetscCall(MatSetUp(*B));
  PetscCall(MatGetOwnershipRange(*B,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    xi = (i+1)*h;
    b = -4.1+xi*(1.0-PetscExpReal(xi-PETSC_PI));
    PetscCall(MatSetValue(*B,i,i,b,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(*B,MAT_HERMITIAN,PETSC_TRUE));
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
  PetscCall(MatGetSize(fun,&n,NULL));
  h = PETSC_PI/(PetscReal)(n+1);
  PetscCall(MatGetOwnershipRange(fun,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) PetscCall(MatSetValue(fun,i,i-1,1.0/(h*h),INSERT_VALUES));
    if (i<n-1) PetscCall(MatSetValue(fun,i,i+1,1.0/(h*h),INSERT_VALUES));
    xi = (i+1)*h;
    b = -4.1+xi*(1.0-PetscExpReal(xi-PETSC_PI));
    PetscCall(MatSetValue(fun,i,i,-lambda-2.0/(h*h)+user->a+PetscExpScalar(-user->tau*lambda)*b,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  if (fun != B) {
    PetscCall(MatAssemblyBegin(fun,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(fun,MAT_FINAL_ASSEMBLY));
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
  PetscCall(MatGetSize(jac,&n,NULL));
  h = PETSC_PI/(PetscReal)(n+1);
  PetscCall(MatGetOwnershipRange(jac,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    xi = (i+1)*h;
    b = -4.1+xi*(1.0-PetscExpReal(xi-PETSC_PI));
    PetscCall(MatSetValue(jac,i,i,-1.0-user->tau*PetscExpScalar(-user->tau*lambda)*b,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
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

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-tau",&tau,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-split",&split,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n1-D Delay Eigenproblem, n=%" PetscInt_FMT ", tau=%g\n\n",n,(double)tau));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
              Create nonlinear eigensolver and set options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(NEPCreate(PETSC_COMM_WORLD,&nep));
  PetscCall(NEPSetTolerances(nep,1e-9,PETSC_DEFAULT));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      First solve
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  if (split) {
    PetscCall(BuildSplitMatrices(n,a,&Id,&A,&B));
    /* f1=-lambda */
    PetscCall(FNCreate(PETSC_COMM_WORLD,&f1));
    PetscCall(FNSetType(f1,FNRATIONAL));
    coeffs[0] = -1.0; coeffs[1] = 0.0;
    PetscCall(FNRationalSetNumerator(f1,2,coeffs));
    /* f2=1.0 */
    PetscCall(FNCreate(PETSC_COMM_WORLD,&f2));
    PetscCall(FNSetType(f2,FNRATIONAL));
    coeffs[0] = 1.0;
    PetscCall(FNRationalSetNumerator(f2,1,coeffs));
    /* f3=exp(-tau*lambda) */
    PetscCall(FNCreate(PETSC_COMM_WORLD,&f3));
    PetscCall(FNSetType(f3,FNEXP));
    PetscCall(FNSetScale(f3,-tau,1.0));
    mats[0] = A;  funs[0] = f2;
    mats[1] = Id; funs[1] = f1;
    mats[2] = B;  funs[2] = f3;
    PetscCall(NEPSetSplitOperator(nep,3,mats,funs,SUBSET_NONZERO_PATTERN));
  } else {
    /* callback form  */
    ctx.tau = tau;
    ctx.a   = a;
    PetscCall(MatCreate(PETSC_COMM_WORLD,&F));
    PetscCall(MatSetSizes(F,PETSC_DECIDE,PETSC_DECIDE,n,n));
    PetscCall(MatSetFromOptions(F));
    PetscCall(MatSeqAIJSetPreallocation(F,3,NULL));
    PetscCall(MatMPIAIJSetPreallocation(F,3,NULL,1,NULL));
    PetscCall(MatSetUp(F));
    PetscCall(NEPSetFunction(nep,F,F,FormFunction,&ctx));
    PetscCall(MatCreate(PETSC_COMM_WORLD,&J));
    PetscCall(MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,n,n));
    PetscCall(MatSetFromOptions(J));
    PetscCall(MatSeqAIJSetPreallocation(J,3,NULL));
    PetscCall(MatMPIAIJSetPreallocation(F,3,NULL,1,NULL));
    PetscCall(MatSetUp(J));
    PetscCall(NEPSetJacobian(nep,J,FormJacobian,&ctx));
  }

  /* Set solver parameters at runtime */
  PetscCall(NEPSetFromOptions(nep));

  /* Solve the eigensystem */
  PetscCall(NEPSolve(nep));
  PetscCall(NEPErrorView(nep,NEP_ERROR_RELATIVE,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
               Second solve, with problem matrices of size 2*n
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  n *= 2;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n1-D Delay Eigenproblem, n=%" PetscInt_FMT ", tau=%g\n\n",n,(double)tau));
  if (split) {
    PetscCall(MatDestroy(&Id));
    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&B));
    PetscCall(BuildSplitMatrices(n,a,&Id,&A,&B));
    mats[0] = A;
    mats[1] = Id;
    mats[2] = B;
    PetscCall(NEPSetSplitOperator(nep,3,mats,funs,SUBSET_NONZERO_PATTERN));
  } else {
    /* callback form  */
    PetscCall(MatDestroy(&F));
    PetscCall(MatDestroy(&J));
    PetscCall(MatCreate(PETSC_COMM_WORLD,&F));
    PetscCall(MatSetSizes(F,PETSC_DECIDE,PETSC_DECIDE,n,n));
    PetscCall(MatSetFromOptions(F));
    PetscCall(MatSeqAIJSetPreallocation(F,3,NULL));
    PetscCall(MatMPIAIJSetPreallocation(F,3,NULL,1,NULL));
    PetscCall(MatSetUp(F));
    PetscCall(NEPSetFunction(nep,F,F,FormFunction,&ctx));
    PetscCall(MatCreate(PETSC_COMM_WORLD,&J));
    PetscCall(MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,n,n));
    PetscCall(MatSetFromOptions(J));
    PetscCall(MatSeqAIJSetPreallocation(J,3,NULL));
    PetscCall(MatMPIAIJSetPreallocation(F,3,NULL,1,NULL));
    PetscCall(MatSetUp(J));
    PetscCall(NEPSetJacobian(nep,J,FormJacobian,&ctx));
  }

  /* Solve the eigensystem */
  PetscCall(NEPSolve(nep));
  PetscCall(NEPErrorView(nep,NEP_ERROR_RELATIVE,NULL));

  PetscCall(NEPDestroy(&nep));
  if (split) {
    PetscCall(MatDestroy(&Id));
    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&B));
    PetscCall(FNDestroy(&f1));
    PetscCall(FNDestroy(&f2));
    PetscCall(FNDestroy(&f3));
  } else {
    PetscCall(MatDestroy(&F));
    PetscCall(MatDestroy(&J));
  }
  PetscCall(SlepcFinalize());
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
      timeoutfactor: 2

TEST*/
