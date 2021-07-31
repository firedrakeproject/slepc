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
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-p",&p,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nGeneralized singular value decomposition, (%D+%D)x%D\n\n",m,p,n);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                     Generate the matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    if (i>0 && i-1<n) { ierr = MatSetValue(A,i,i-1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (i+1<n) { ierr = MatSetValue(A,i,i+1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (i<n) { ierr = MatSetValue(A,i,i,2.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (i>n) { ierr = MatSetValue(A,i,n-1,1.0,INSERT_VALUES);CHKERRQ(ierr); }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,p,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(B,&Istart,&Iend);CHKERRQ(ierr);
  d = PetscMax(0,n-p);
  for (i=Istart;i<Iend;i++) {
    for (j=0;j<=PetscMin(i,n-1);j++) {
      ierr = MatSetValue(B,i,j+d,1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create the singular value solver, set options and solve
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SVDCreate(PETSC_COMM_WORLD,&svd);CHKERRQ(ierr);
  ierr = SVDSetOperators(svd,A,B);CHKERRQ(ierr);
  ierr = SVDSetDimensions(svd,4,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = SVDSetConvergenceTest(svd,SVD_CONV_NORM);CHKERRQ(ierr);

  ierr = SVDSetType(svd,SVDTRLANCZOS);CHKERRQ(ierr);
  ierr = SVDTRLanczosSetGBidiag(svd,SVD_TRLANCZOS_GBIDIAG_UPPER);CHKERRQ(ierr);

  /* create a standalone KSP with appropriate settings */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPLSQR);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,PETSC_SMALL,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = SVDTRLanczosSetKSP(svd,ksp);CHKERRQ(ierr);
  ierr = SVDTRLanczosSetRestart(svd,0.4);CHKERRQ(ierr);
  ierr = SVDTRLanczosSetLocking(svd,PETSC_TRUE);CHKERRQ(ierr);

  ierr = SVDSetFromOptions(svd);CHKERRQ(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)svd,SVDTRLANCZOS,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = SVDTRLanczosGetGBidiag(svd,&bidiag);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"TRLANCZOS: using %s bidiagonalization\n",SVDTRLanczosGBidiags[bidiag]);CHKERRQ(ierr);
    ierr = SVDTRLanczosGetRestart(svd,&keep);CHKERRQ(ierr);
    ierr = SVDTRLanczosGetLocking(svd,&lock);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"TRLANCZOS: restarting parameter %.2f %s\n",(double)keep,lock?"(locking)":"");CHKERRQ(ierr);
  }

  ierr = SVDSolve(svd);CHKERRQ(ierr);
  ierr = SVDErrorView(svd,SVD_ERROR_RELATIVE,NULL);CHKERRQ(ierr);

  /* Free work space */
  ierr = SVDDestroy(&svd);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      requires: !single

   test:
      suffix: 2
      args: -m 6 -n 12 -p 12 -svd_trlanczos_gbidiag {{single upper lower}}
      filter: grep -v "TRLANCZOS: using"
      requires: !single

TEST*/
