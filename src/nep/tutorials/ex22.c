/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Delay differential equation.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions.\n"
  "  -tau <tau>, where <tau> is the delay parameter.\n\n";

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
  NEP            nep;             /* nonlinear eigensolver context */
  Mat            Id,A,B;          /* problem matrices */
  FN             f1,f2,f3;        /* functions to define the nonlinear operator */
  Mat            mats[3];
  FN             funs[3];
  PetscScalar    coeffs[2],b;
  PetscInt       n=128,nev,Istart,Iend,i;
  PetscReal      tau=0.001,h,a=20,xi;
  PetscBool      terse;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-tau",&tau,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n1-D Delay Eigenproblem, n=%" PetscInt_FMT ", tau=%g\n\n",n,(double)tau));
  h = PETSC_PI/(PetscReal)(n+1);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear eigensolver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(NEPCreate(PETSC_COMM_WORLD,&nep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create problem matrices and coefficient functions. Pass them to NEP
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Identity matrix
  */
  PetscCall(MatCreateConstantDiagonal(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,1.0,&Id));
  PetscCall(MatSetOption(Id,MAT_HERMITIAN,PETSC_TRUE));

  /*
     A = 1/h^2*tridiag(1,-2,1) + a*I
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) PetscCall(MatSetValue(A,i,i-1,1.0/(h*h),INSERT_VALUES));
    if (i<n-1) PetscCall(MatSetValue(A,i,i+1,1.0/(h*h),INSERT_VALUES));
    PetscCall(MatSetValue(A,i,i,-2.0/(h*h)+a,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(A,MAT_HERMITIAN,PETSC_TRUE));

  /*
     B = diag(b(xi))
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
  PetscCall(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetUp(B));
  PetscCall(MatGetOwnershipRange(B,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    xi = (i+1)*h;
    b = -4.1+xi*(1.0-PetscExpReal(xi-PETSC_PI));
    PetscCall(MatSetValue(B,i,i,b,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(B,MAT_HERMITIAN,PETSC_TRUE));

  /*
     Functions: f1=-lambda, f2=1.0, f3=exp(-tau*lambda)
  */
  PetscCall(FNCreate(PETSC_COMM_WORLD,&f1));
  PetscCall(FNSetType(f1,FNRATIONAL));
  coeffs[0] = -1.0; coeffs[1] = 0.0;
  PetscCall(FNRationalSetNumerator(f1,2,coeffs));

  PetscCall(FNCreate(PETSC_COMM_WORLD,&f2));
  PetscCall(FNSetType(f2,FNRATIONAL));
  coeffs[0] = 1.0;
  PetscCall(FNRationalSetNumerator(f2,1,coeffs));

  PetscCall(FNCreate(PETSC_COMM_WORLD,&f3));
  PetscCall(FNSetType(f3,FNEXP));
  PetscCall(FNSetScale(f3,-tau,1.0));

  /*
     Set the split operator. Note that A is passed first so that
     SUBSET_NONZERO_PATTERN can be used
  */
  mats[0] = A;  funs[0] = f2;
  mats[1] = Id; funs[1] = f1;
  mats[2] = B;  funs[2] = f3;
  PetscCall(NEPSetSplitOperator(nep,3,mats,funs,SUBSET_NONZERO_PATTERN));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Customize nonlinear solver; set runtime options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(NEPSetTolerances(nep,1e-9,PETSC_DEFAULT));
  PetscCall(NEPSetDimensions(nep,1,PETSC_DEFAULT,PETSC_DEFAULT));
  PetscCall(NEPRIISetLagPreconditioner(nep,0));

  /*
     Set solver parameters at runtime
  */
  PetscCall(NEPSetFromOptions(nep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(NEPSolve(nep));
  PetscCall(NEPGetDimensions(nep,&nev,NULL,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %" PetscInt_FMT "\n",nev));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  PetscCall(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) PetscCall(NEPErrorView(nep,NEP_ERROR_RELATIVE,NULL));
  else {
    PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    PetscCall(NEPConvergedReasonView(nep,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(NEPErrorView(nep,NEP_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(NEPDestroy(&nep));
  PetscCall(MatDestroy(&Id));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(FNDestroy(&f1));
  PetscCall(FNDestroy(&f2));
  PetscCall(FNDestroy(&f3));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   testset:
      suffix: 1
      args: -nep_type {{rii slp narnoldi}} -terse
      filter: sed -e "s/[+-]0\.0*i//g"
      requires: !single

   test:
      suffix: 1_ciss
      args: -nep_type ciss -nep_ciss_extraction {{ritz hankel caa}} -rg_type ellipse -rg_ellipse_center 10 -rg_ellipse_radius 9.5 -nep_ncv 24 -terse
      requires: complex !single

   test:
      suffix: 2
      args: -nep_type interpol -nep_interpol_pep_extract {{none norm residual}} -rg_type interval -rg_interval_endpoints 5,20,-.1,.1 -nep_nev 3 -nep_target 5 -terse
      filter: sed -e "s/[+-]0\.0*i//g"
      requires: !single

   testset:
      args: -n 512 -nep_target 10 -nep_nev 3 -terse
      filter: sed -e "s/[+-]0\.0*i//g"
      requires: !single
      output_file: output/ex22_3.out
      test:
         suffix: 3
         args: -nep_type {{rii slp narnoldi}}
      test:
         suffix: 3_simpleu
         args: -nep_type {{rii slp narnoldi}} -nep_deflation_simpleu
      test:
         suffix: 3_slp_thres
         args: -nep_type slp -nep_slp_deflation_threshold 1e-8
      test:
         suffix: 3_rii_thres
         args: -nep_type rii -nep_rii_deflation_threshold 1e-8

   test:
      suffix: 4
      args: -nep_type interpol -rg_type interval -rg_interval_endpoints 5,20,-.1,.1 -nep_nev 3 -nep_target 5 -terse -nep_monitor draw::draw_lg
      filter: sed -e "s/[+-]0\.0*i//g"
      requires: x !single
      output_file: output/ex22_2.out

TEST*/
