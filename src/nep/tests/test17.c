/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Tests a user-provided preconditioner.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions.\n"
  "  -tau <tau>, where <tau> is the delay parameter.\n"
  "  -a <a>, where <a> is the coefficient that multiplies u in the equation.\n"
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
  PetscErrorCode ierr;
  PetscInt       i,Istart,Iend;
  PetscReal      h,xi;
  PetscScalar    b;

  PetscFunctionBeginUser;
  h = PETSC_PI/(PetscReal)(n+1);

  /* Identity matrix */
  ierr = MatCreateConstantDiagonal(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,1.0,Id);CHKERRQ(ierr);
  ierr = MatSetOption(*Id,MAT_HERMITIAN,PETSC_TRUE);CHKERRQ(ierr);

  /* A = 1/h^2*tridiag(1,-2,1) + a*I */
  ierr = MatCreate(PETSC_COMM_WORLD,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*A);CHKERRQ(ierr);
  ierr = MatSetUp(*A);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(*A,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    if (i>0) { ierr = MatSetValue(*A,i,i-1,1.0/(h*h),INSERT_VALUES);CHKERRQ(ierr); }
    if (i<n-1) { ierr = MatSetValue(*A,i,i+1,1.0/(h*h),INSERT_VALUES);CHKERRQ(ierr); }
    ierr = MatSetValue(*A,i,i,-2.0/(h*h)+a,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(*A,MAT_HERMITIAN,PETSC_TRUE);CHKERRQ(ierr);

  /* B = diag(b(xi)) */
  ierr = MatCreate(PETSC_COMM_WORLD,B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*B);CHKERRQ(ierr);
  ierr = MatSetUp(*B);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(*B,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    xi = (i+1)*h;
    b = -4.1+xi*(1.0-PetscExpReal(xi-PETSC_PI));
    ierr = MatSetValue(*B,i,i,b,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(*B,MAT_HERMITIAN,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Create preconditioner matrices (only Ap=diag(A))
*/
PetscErrorCode BuildSplitPreconditioner(PetscInt n,PetscReal a,Mat *Ap)
{
  PetscErrorCode ierr;
  PetscInt       i,Istart,Iend;
  PetscReal      h;

  PetscFunctionBeginUser;
  h = PETSC_PI/(PetscReal)(n+1);

  /* Ap = diag(A) */
  ierr = MatCreate(PETSC_COMM_WORLD,Ap);CHKERRQ(ierr);
  ierr = MatSetSizes(*Ap,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*Ap);CHKERRQ(ierr);
  ierr = MatSetUp(*Ap);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(*Ap,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    ierr = MatSetValue(*Ap,i,i,-2.0/(h*h)+a,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*Ap,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*Ap,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(*Ap,MAT_HERMITIAN,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Compute Function matrix  T(lambda)
*/
PetscErrorCode FormFunction(NEP nep,PetscScalar lambda,Mat fun,Mat B,void *ctx)
{
  PetscErrorCode ierr;
  ApplicationCtx *user = (ApplicationCtx*)ctx;
  PetscInt       i,n,Istart,Iend;
  PetscReal      h,xi;
  PetscScalar    b;

  PetscFunctionBeginUser;
  ierr = MatGetSize(fun,&n,NULL);CHKERRQ(ierr);
  h = PETSC_PI/(PetscReal)(n+1);
  ierr = MatGetOwnershipRange(fun,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    if (i>0) { ierr = MatSetValue(fun,i,i-1,1.0/(h*h),INSERT_VALUES);CHKERRQ(ierr); }
    if (i<n-1) { ierr = MatSetValue(fun,i,i+1,1.0/(h*h),INSERT_VALUES);CHKERRQ(ierr); }
    xi = (i+1)*h;
    b = -4.1+xi*(1.0-PetscExpReal(xi-PETSC_PI));
    ierr = MatSetValue(fun,i,i,-lambda-2.0/(h*h)+user->a+PetscExpScalar(-user->tau*lambda)*b,INSERT_VALUES);CHKERRQ(ierr);
    if (B!=fun) { ierr = MatSetValue(B,i,i,-lambda-2.0/(h*h)+user->a+PetscExpScalar(-user->tau*lambda)*b,INSERT_VALUES);CHKERRQ(ierr); }
  }
  ierr = MatAssemblyBegin(fun,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(fun,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (fun != B) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   Compute Jacobian matrix  T'(lambda)
*/
PetscErrorCode FormJacobian(NEP nep,PetscScalar lambda,Mat jac,void *ctx)
{
  PetscErrorCode ierr;
  ApplicationCtx *user = (ApplicationCtx*)ctx;
  PetscInt       i,n,Istart,Iend;
  PetscReal      h,xi;
  PetscScalar    b;

  PetscFunctionBeginUser;
  ierr = MatGetSize(jac,&n,NULL);CHKERRQ(ierr);
  h = PETSC_PI/(PetscReal)(n+1);
  ierr = MatGetOwnershipRange(jac,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    xi = (i+1)*h;
    b = -4.1+xi*(1.0-PetscExpReal(xi-PETSC_PI));
    ierr = MatSetValue(jac,i,i,-1.0-user->tau*PetscExpScalar(-user->tau*lambda)*b,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  NEP            nep;             /* nonlinear eigensolver context */
  Mat            Id,A,B,Ap,J,F,P; /* problem matrices */
  FN             f1,f2,f3;        /* functions to define the nonlinear operator */
  ApplicationCtx ctx;             /* user-defined context */
  Mat            mats[3];
  FN             funs[3];
  PetscScalar    coeffs[2];
  PetscInt       n=128;
  PetscReal      tau=0.001,a=20;
  PetscBool      split=PETSC_TRUE;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-tau",&tau,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-a",&a,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-split",&split,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n1-D Delay Eigenproblem, n=%" PetscInt_FMT ", tau=%g, a=%g\n\n",n,(double)tau,(double)a);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
              Create nonlinear eigensolver and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = NEPCreate(PETSC_COMM_WORLD,&nep);CHKERRQ(ierr);
  if (split) {
    ierr = BuildSplitMatrices(n,a,&Id,&A,&B);CHKERRQ(ierr);
    /* f1=-lambda */
    ierr = FNCreate(PETSC_COMM_WORLD,&f1);CHKERRQ(ierr);
    ierr = FNSetType(f1,FNRATIONAL);CHKERRQ(ierr);
    coeffs[0] = -1.0; coeffs[1] = 0.0;
    ierr = FNRationalSetNumerator(f1,2,coeffs);CHKERRQ(ierr);
    /* f2=1.0 */
    ierr = FNCreate(PETSC_COMM_WORLD,&f2);CHKERRQ(ierr);
    ierr = FNSetType(f2,FNRATIONAL);CHKERRQ(ierr);
    coeffs[0] = 1.0;
    ierr = FNRationalSetNumerator(f2,1,coeffs);CHKERRQ(ierr);
    /* f3=exp(-tau*lambda) */
    ierr = FNCreate(PETSC_COMM_WORLD,&f3);CHKERRQ(ierr);
    ierr = FNSetType(f3,FNEXP);CHKERRQ(ierr);
    ierr = FNSetScale(f3,-tau,1.0);CHKERRQ(ierr);
    mats[0] = A;  funs[0] = f2;
    mats[1] = Id; funs[1] = f1;
    mats[2] = B;  funs[2] = f3;
    ierr = NEPSetSplitOperator(nep,3,mats,funs,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = BuildSplitPreconditioner(n,a,&Ap);CHKERRQ(ierr);
    mats[0] = Ap;
    mats[1] = Id;
    mats[2] = B;
    ierr = NEPSetSplitPreconditioner(nep,3,mats,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  } else {
    /* callback form  */
    ctx.tau = tau;
    ctx.a   = a;
    ierr = MatCreate(PETSC_COMM_WORLD,&F);CHKERRQ(ierr);
    ierr = MatSetSizes(F,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
    ierr = MatSetFromOptions(F);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(F,3,NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(F,3,NULL,1,NULL);CHKERRQ(ierr);
    ierr = MatSetUp(F);CHKERRQ(ierr);
    ierr = MatDuplicate(F,MAT_DO_NOT_COPY_VALUES,&P);CHKERRQ(ierr);
    ierr = NEPSetFunction(nep,F,P,FormFunction,&ctx);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
    ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
    ierr = MatSetFromOptions(J);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(J,3,NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(F,3,NULL,1,NULL);CHKERRQ(ierr);
    ierr = MatSetUp(J);CHKERRQ(ierr);
    ierr = NEPSetJacobian(nep,J,FormJacobian,&ctx);CHKERRQ(ierr);
  }

  /* Set solver parameters at runtime */
  ierr = NEPSetFromOptions(nep);CHKERRQ(ierr);

  /* Solve the eigensystem */
  ierr = NEPSolve(nep);CHKERRQ(ierr);
  ierr = NEPErrorView(nep,NEP_ERROR_RELATIVE,NULL);CHKERRQ(ierr);

  ierr = NEPDestroy(&nep);CHKERRQ(ierr);
  if (split) {
    ierr = MatDestroy(&Id);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    ierr = MatDestroy(&Ap);CHKERRQ(ierr);
    ierr = FNDestroy(&f1);CHKERRQ(ierr);
    ierr = FNDestroy(&f2);CHKERRQ(ierr);
    ierr = FNDestroy(&f3);CHKERRQ(ierr);
  } else {
    ierr = MatDestroy(&F);CHKERRQ(ierr);
    ierr = MatDestroy(&P);CHKERRQ(ierr);
    ierr = MatDestroy(&J);CHKERRQ(ierr);
  }
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      args: -a 90000 -nep_nev 2
      requires: double
      output_file: output/test17_1.out
      test:
         suffix: 1
         args: -nep_type slp -nep_two_sided {{0 1}} -split {{0 1}}

   testset:
      args: -nep_nev 2 -rg_type interval -rg_interval_endpoints .5,15,-.1,.1 -nep_target .7
      requires: !single
      output_file: output/test17_2.out
      filter: sed -e "s/[+-]0\.0*i//g"
      test:
         suffix: 2_interpol
         args: -nep_type interpol -nep_interpol_st_ksp_type bcgs -nep_interpol_st_pc_type sor -nep_tol 1e-6 -nep_interpol_st_ksp_rtol 1e-7
      test:
         suffix: 2_nleigs
         args: -nep_type nleigs -split {{0 1}}
         requires: complex
      test:
         suffix: 2_nleigs_real
         args: -nep_type nleigs -rg_interval_endpoints .5,15 -split {{0 1}} -nep_nleigs_ksp_type tfqmr
         requires: !complex

   testset:
      args: -nep_type ciss -rg_type ellipse -rg_ellipse_center 10 -rg_ellipse_radius 9.5 -rg_ellipse_vscale 0.1 -nep_ciss_ksp_type bcgs -nep_ciss_pc_type sor
      output_file: output/test17_3.out
      requires: complex !single
      test:
         suffix: 3
         args: -split {{0 1}}
      test:
         suffix: 3_par
         nsize: 2
         args: -nep_ciss_partitions 2

TEST*/
