/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Tests user interface for TRLANCZOS with GSVD.\n\n"
  "The command line options are:\n"
  "  -m <m>, where <m> = number of rows of A.\n"
  "  -n <n>, where <n> = number of columns of A.\n"
  "  -p <p>, where <p> = number of rows of B.\n\n";

#include <slepcsvd.h>

int main(int argc,char **argv)
{
  Mat                 A,B;
  SVD                 svd;
  KSP                 ksp;
  PC                  pc;
  PetscInt            m=15,n=20,p=21,i,j,d,Istart,Iend;
  PetscReal           keep;
  PetscBool           flg,lock;
  SVDTRLanczosGBidiag bidiag;
  PetscErrorCode      ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-p",&p,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nGeneralized singular value decomposition, (%" PetscInt_FMT "+%" PetscInt_FMT ")x%" PetscInt_FMT "\n\n",m,p,n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                     Generate the matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0 && i-1<n) CHKERRQ(MatSetValue(A,i,i-1,-1.0,INSERT_VALUES));
    if (i+1<n) CHKERRQ(MatSetValue(A,i,i+1,-1.0,INSERT_VALUES));
    if (i<n) CHKERRQ(MatSetValue(A,i,i,2.0,INSERT_VALUES));
    if (i>n) CHKERRQ(MatSetValue(A,i,n-1,1.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,p,n));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(B));

  CHKERRQ(MatGetOwnershipRange(B,&Istart,&Iend));
  d = PetscMax(0,n-p);
  for (i=Istart;i<Iend;i++) {
    for (j=0;j<=PetscMin(i,n-1);j++) {
      CHKERRQ(MatSetValue(B,i,j+d,1.0,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create the singular value solver, set options and solve
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(SVDCreate(PETSC_COMM_WORLD,&svd));
  CHKERRQ(SVDSetOperators(svd,A,B));
  CHKERRQ(SVDSetDimensions(svd,4,PETSC_DEFAULT,PETSC_DEFAULT));
  CHKERRQ(SVDSetConvergenceTest(svd,SVD_CONV_NORM));

  CHKERRQ(SVDSetType(svd,SVDTRLANCZOS));
  CHKERRQ(SVDTRLanczosSetGBidiag(svd,SVD_TRLANCZOS_GBIDIAG_UPPER));

  /* create a standalone KSP with appropriate settings */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetType(ksp,KSPLSQR));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCNONE));
  CHKERRQ(KSPSetTolerances(ksp,PETSC_SMALL,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(SVDTRLanczosSetKSP(svd,ksp));
  CHKERRQ(SVDTRLanczosSetRestart(svd,0.4));
  CHKERRQ(SVDTRLanczosSetLocking(svd,PETSC_TRUE));

  CHKERRQ(SVDSetFromOptions(svd));

  CHKERRQ(PetscObjectTypeCompare((PetscObject)svd,SVDTRLANCZOS,&flg));
  if (flg) {
    CHKERRQ(SVDTRLanczosGetGBidiag(svd,&bidiag));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"TRLANCZOS: using %s bidiagonalization\n",SVDTRLanczosGBidiags[bidiag]));
    CHKERRQ(SVDTRLanczosGetRestart(svd,&keep));
    CHKERRQ(SVDTRLanczosGetLocking(svd,&lock));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"TRLANCZOS: restarting parameter %.2f %s\n",(double)keep,lock?"(locking)":""));
  }

  CHKERRQ(SVDSolve(svd));
  CHKERRQ(SVDErrorView(svd,SVD_ERROR_NORM,NULL));

  /* Free work space */
  CHKERRQ(SVDDestroy(&svd));
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  ierr = SlepcFinalize();
  return ierr;
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

TEST*/
