/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test the NArnoldi solver with a user-provided KSP.\n\n"
  "This is based on ex22.\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions.\n"
  "  -tau <tau>, where <tau> is the delay parameter.\n"
  "  -initv ... set an initial vector.\n\n";

/*
   Solve parabolic partial differential equation with time delay tau

            u_t = u_xx + a*u(t) + b*u(t-tau)
            u(0,t) = u(pi,t) = 0

   with a = 20 and b(x) = -4.1+x*(1-exp(x-pi)).

   Discretization leads to a DDE of dimension n

            -u' = A*u(t) + B*u(t-tau)

   which results in the nonlinear eigenproblem

            (-lambda*I + A + exp(-tau*lambda)*B)*u = 0
*/

#include <slepcnep.h>

int main(int argc,char **argv)
{
  NEP            nep;
  KSP            ksp;
  PC             pc;
  Mat            Id,A,B,mats[3];
  FN             f1,f2,f3,funs[3];
  Vec            v0;
  PetscScalar    coeffs[2],b,*pv;
  PetscInt       n=128,nev,Istart,Iend,i,lag;
  PetscReal      tau=0.001,h,a=20,xi;
  PetscBool      terse,initv=PETSC_FALSE;
  const char     *prefix;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-tau",&tau,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-initv",&initv,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n1-D Delay Eigenproblem, n=%" PetscInt_FMT ", tau=%g\n\n",n,(double)tau));
  h = PETSC_PI/(PetscReal)(n+1);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Create a standalone KSP with appropriate settings
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetType(ksp,KSPBCGS));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCBJACOBI));
  CHKERRQ(KSPSetFromOptions(ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear eigensolver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(NEPCreate(PETSC_COMM_WORLD,&nep));

  /* Identity matrix */
  CHKERRQ(MatCreateConstantDiagonal(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,1.0,&Id));
  CHKERRQ(MatSetOption(Id,MAT_HERMITIAN,PETSC_TRUE));

  /* A = 1/h^2*tridiag(1,-2,1) + a*I */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) CHKERRQ(MatSetValue(A,i,i-1,1.0/(h*h),INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(A,i,i+1,1.0/(h*h),INSERT_VALUES));
    CHKERRQ(MatSetValue(A,i,i,-2.0/(h*h)+a,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatSetOption(A,MAT_HERMITIAN,PETSC_TRUE));

  /* B = diag(b(xi)) */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(B));
  CHKERRQ(MatGetOwnershipRange(B,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    xi = (i+1)*h;
    b = -4.1+xi*(1.0-PetscExpReal(xi-PETSC_PI));
    CHKERRQ(MatSetValues(B,1,&i,1,&i,&b,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatSetOption(B,MAT_HERMITIAN,PETSC_TRUE));

  /* Functions: f1=-lambda, f2=1.0, f3=exp(-tau*lambda) */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f1));
  CHKERRQ(FNSetType(f1,FNRATIONAL));
  coeffs[0] = -1.0; coeffs[1] = 0.0;
  CHKERRQ(FNRationalSetNumerator(f1,2,coeffs));

  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f2));
  CHKERRQ(FNSetType(f2,FNRATIONAL));
  coeffs[0] = 1.0;
  CHKERRQ(FNRationalSetNumerator(f2,1,coeffs));

  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f3));
  CHKERRQ(FNSetType(f3,FNEXP));
  CHKERRQ(FNSetScale(f3,-tau,1.0));

  /* Set the split operator */
  mats[0] = A;  funs[0] = f2;
  mats[1] = Id; funs[1] = f1;
  mats[2] = B;  funs[2] = f3;
  CHKERRQ(NEPSetSplitOperator(nep,3,mats,funs,SUBSET_NONZERO_PATTERN));

  /* Customize nonlinear solver; set runtime options */
  CHKERRQ(NEPSetOptionsPrefix(nep,"check_"));
  CHKERRQ(NEPAppendOptionsPrefix(nep,"myprefix_"));
  CHKERRQ(NEPGetOptionsPrefix(nep,&prefix));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"NEP prefix is currently: %s\n\n",prefix));
  CHKERRQ(NEPSetType(nep,NEPNARNOLDI));
  CHKERRQ(NEPNArnoldiSetKSP(nep,ksp));
  if (initv) { /* initial vector */
    CHKERRQ(MatCreateVecs(A,&v0,NULL));
    CHKERRQ(VecGetArray(v0,&pv));
    for (i=Istart;i<Iend;i++) pv[i-Istart] = PetscSinReal((4.0*PETSC_PI*i)/n);
    CHKERRQ(VecRestoreArray(v0,&pv));
    CHKERRQ(NEPSetInitialSpace(nep,1,&v0));
    CHKERRQ(VecDestroy(&v0));
  }
  CHKERRQ(NEPSetFromOptions(nep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(NEPSolve(nep));
  CHKERRQ(NEPGetDimensions(nep,&nev,NULL,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %" PetscInt_FMT "\n",nev));
  CHKERRQ(NEPNArnoldiGetLagPreconditioner(nep,&lag));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," N-Arnoldi lag parameter: %" PetscInt_FMT "\n",lag));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) CHKERRQ(NEPErrorView(nep,NEP_ERROR_RELATIVE,NULL));
  else {
    CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    CHKERRQ(NEPConvergedReasonView(nep,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(NEPErrorView(nep,NEP_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(NEPDestroy(&nep));
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(MatDestroy(&Id));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(FNDestroy(&f1));
  CHKERRQ(FNDestroy(&f2));
  CHKERRQ(FNDestroy(&f3));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -check_myprefix_nep_view -check_myprefix_nep_monitor_conv -initv -terse
      filter: grep -v "tolerance" | sed -e "s/[0-9]\.[0-9]*e[+-]\([0-9]*\)/removed/g" -e "s/+0i//g"
      requires: double

TEST*/
