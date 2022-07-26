/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Tests user interface for TRLANCZOS with GSVD.\n\n"
  "The command line options are:\n"
  "  -m <m>, where <m> = number of rows of A.\n"
  "  -n <n>, where <n> = number of columns of A.\n"
  "  -p <p>, where <p> = number of rows of B.\n"
  "  -s <s>, where <s> = scale parameter.\n\n";

#include <slepcsvd.h>

int main(int argc,char **argv)
{
  Mat                 A,B;
  SVD                 svd;
  KSP                 ksp;
  PC                  pc;
  PetscInt            m=15,n=20,p=21,i,j,d,Istart,Iend;
  PetscReal           keep,scale=1.0;
  PetscBool           flg,lock,oneside;
  SVDTRLanczosGBidiag bidiag;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-p",&p,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-s",&scale,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nGeneralized singular value decomposition, (%" PetscInt_FMT "+%" PetscInt_FMT ")x%" PetscInt_FMT "\n\n",m,p,n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                     Generate the matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0 && i-1<n) PetscCall(MatSetValue(A,i,i-1,-1.0,INSERT_VALUES));
    if (i+1<n) PetscCall(MatSetValue(A,i,i+1,-1.0,INSERT_VALUES));
    if (i<n) PetscCall(MatSetValue(A,i,i,2.0,INSERT_VALUES));
    if (i>n) PetscCall(MatSetValue(A,i,n-1,1.0,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
  PetscCall(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,p,n));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetUp(B));

  PetscCall(MatGetOwnershipRange(B,&Istart,&Iend));
  d = PetscMax(0,n-p);
  for (i=Istart;i<Iend;i++) {
    for (j=0;j<=PetscMin(i,n-1);j++) PetscCall(MatSetValue(B,i,j+d,1.0,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create the singular value solver, set options and solve
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(SVDCreate(PETSC_COMM_WORLD,&svd));
  PetscCall(SVDSetOperators(svd,A,B));
  PetscCall(SVDSetProblemType(svd,SVD_GENERALIZED));
  PetscCall(SVDSetDimensions(svd,4,PETSC_DEFAULT,PETSC_DEFAULT));
  PetscCall(SVDSetConvergenceTest(svd,SVD_CONV_NORM));

  PetscCall(SVDSetType(svd,SVDTRLANCZOS));
  PetscCall(SVDTRLanczosSetGBidiag(svd,SVD_TRLANCZOS_GBIDIAG_UPPER));
  PetscCall(SVDTRLanczosSetScale(svd,scale));

  /* create a standalone KSP with appropriate settings */
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetType(ksp,KSPLSQR));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCNONE));
  PetscCall(KSPSetTolerances(ksp,PETSC_SMALL,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(SVDTRLanczosSetKSP(svd,ksp));
  PetscCall(SVDTRLanczosSetRestart(svd,0.4));
  PetscCall(SVDTRLanczosSetLocking(svd,PETSC_TRUE));

  PetscCall(SVDSetFromOptions(svd));

  PetscCall(PetscObjectTypeCompare((PetscObject)svd,SVDTRLANCZOS,&flg));
  if (flg) {
    PetscCall(SVDTRLanczosGetGBidiag(svd,&bidiag));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"TRLANCZOS: using %s bidiagonalization\n",SVDTRLanczosGBidiags[bidiag]));
    PetscCall(SVDTRLanczosGetRestart(svd,&keep));
    PetscCall(SVDTRLanczosGetLocking(svd,&lock));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"TRLANCZOS: restarting parameter %.2f %s\n",(double)keep,lock?"(locking)":""));
    PetscCall(SVDTRLanczosGetScale(svd,&scale));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"TRLANCZOS: scale parameter %g\n",(double)scale));
    PetscCall(SVDTRLanczosGetOneSide(svd,&oneside));
    if (oneside) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"TRLANCZOS: using one-sided orthogonalization\n"));
  }

  PetscCall(SVDSolve(svd));
  PetscCall(SVDErrorView(svd,SVD_ERROR_NORM,NULL));

  /* Free work space */
  PetscCall(SVDDestroy(&svd));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -svd_trlanczos_gbidiag {{single upper lower}}
      filter: grep -v "TRLANCZOS: using"
      requires: !single

   test:
      suffix: 2
      args: -m 6 -n 12 -p 12 -svd_trlanczos_restart .7
      requires: !single

   test:
      suffix: 3
      args: -s 8 -svd_trlanczos_gbidiag lower
      requires: !single

   test:
      suffix: 4
      args: -s -5 -svd_trlanczos_gbidiag lower
      requires: !single

TEST*/
