/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test the CISS solver with the problem of ex22.\n\n"
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
  NEP               nep;
  Mat               Id,A,B,mats[3];
  FN                f1,f2,f3,funs[3];
  RG                rg;
  KSP               *ksp;
  PC                pc;
  NEPCISSExtraction ext;
  PetscScalar       coeffs[2],b;
  PetscInt          n=128,Istart,Iend,i,nsolve;
  PetscReal         tau=0.001,h,a=20,xi;
  PetscBool         flg,terse;

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

  /* Identity matrix */
  PetscCall(MatCreateConstantDiagonal(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,1.0,&Id));
  PetscCall(MatSetOption(Id,MAT_HERMITIAN,PETSC_TRUE));

  /* A = 1/h^2*tridiag(1,-2,1) + a*I */
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

  /* B = diag(b(xi)) */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
  PetscCall(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetUp(B));
  PetscCall(MatGetOwnershipRange(B,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    xi = (i+1)*h;
    b = -4.1+xi*(1.0-PetscExpReal(xi-PETSC_PI));
    PetscCall(MatSetValues(B,1,&i,1,&i,&b,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(B,MAT_HERMITIAN,PETSC_TRUE));

  /* Functions: f1=-lambda, f2=1.0, f3=exp(-tau*lambda) */
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

  /* Set the split operator */
  mats[0] = A;  funs[0] = f2;
  mats[1] = Id; funs[1] = f1;
  mats[2] = B;  funs[2] = f3;
  PetscCall(NEPSetSplitOperator(nep,3,mats,funs,SUBSET_NONZERO_PATTERN));

  /* Customize nonlinear solver; set runtime options */
  PetscCall(NEPSetType(nep,NEPCISS));
  PetscCall(NEPSetDimensions(nep,1,24,PETSC_DEFAULT));
  PetscCall(NEPSetTolerances(nep,1e-9,PETSC_DEFAULT));
  PetscCall(NEPGetRG(nep,&rg));
  PetscCall(RGSetType(rg,RGELLIPSE));
  PetscCall(RGEllipseSetParameters(rg,10.0,9.5,0.1));
  PetscCall(NEPCISSSetSizes(nep,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,1,PETSC_DEFAULT,PETSC_TRUE));
  PetscCall(NEPCISSGetKSPs(nep,&nsolve,&ksp));
  for (i=0;i<nsolve;i++) {
    PetscCall(KSPSetTolerances(ksp[i],1e-12,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
    PetscCall(KSPSetType(ksp[i],KSPBCGS));
    PetscCall(KSPGetPC(ksp[i],&pc));
    PetscCall(PCSetType(pc,PCSOR));
  }
  PetscCall(NEPSetFromOptions(nep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PetscObjectTypeCompare((PetscObject)nep,NEPCISS,&flg));
  if (flg) {
    PetscCall(NEPCISSGetExtraction(nep,&ext));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," Running CISS with %" PetscInt_FMT " KSP solvers (%s extraction)\n",nsolve,NEPCISSExtractions[ext]));
  }
  PetscCall(NEPSolve(nep));

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

   build:
      requires: complex

   testset:
      args: -nep_ciss_extraction {{ritz hankel caa}} -terse
      requires: complex !single
      output_file: output/test11_1.out
      filter: sed -e "s/([A-Z]* extraction)//"
      test:
         suffix: 1
         args: -nep_ciss_delta 1e-10
      test:
         suffix: 2
         nsize: 2
         args: -nep_ciss_partitions 2
      test:
         suffix: 3
         args: -nep_ciss_moments 6 -nep_ciss_blocksize 4 -nep_ciss_refine_inner 1 -nep_ciss_refine_blocksize 2

   test:
      suffix: 4
      args: -terse -nep_view
      requires: complex !single
      filter: grep -v tolerance | grep -v threshold

TEST*/
