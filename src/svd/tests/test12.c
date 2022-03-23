/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "SVD problem with user-defined stopping test.\n\n"
  "The command line options are:\n"
  "  -m <m>, where <m> = matrix rows.\n"
  "  -n <n>, where <n> = matrix columns (defaults to m+2).\n\n";

#include <slepcsvd.h>
#include <petsctime.h>

/*
   This example computes the singular values of a rectangular bidiagonal matrix

              |  1  2                     |
              |     1  2                  |
              |        1  2               |
          A = |          .  .             |
              |             .  .          |
              |                1  2       |
              |                   1  2    |
 */

/*
    Function for user-defined stopping test.

    Checks that the computing time has not exceeded the deadline.
*/
PetscErrorCode MyStoppingTest(SVD svd,PetscInt its,PetscInt max_it,PetscInt nconv,PetscInt nev,SVDConvergedReason *reason,void *ctx)
{
  PetscLogDouble now,deadline = *(PetscLogDouble*)ctx;

  PetscFunctionBeginUser;
  /* check if usual termination conditions are met */
  CHKERRQ(SVDStoppingBasic(svd,its,max_it,nconv,nev,reason,NULL));
  if (*reason==SVD_CONVERGED_ITERATING) {
    /* check if deadline has expired */
    CHKERRQ(PetscTime(&now));
    if (now>deadline) *reason = SVD_CONVERGED_USER;
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Mat                A;
  SVD                svd;
  SVDConvergedReason reason;
  PetscInt           m=20,n,Istart,Iend,i,col[2],nconv;
  PetscReal          seconds=2.5;     /* maximum time allowed for computation */
  PetscLogDouble     deadline;        /* time to abort computation */
  PetscScalar        value[] = { 1, 2 };
  PetscBool          terse,flg;
  PetscViewer        viewer;
  PetscErrorCode     ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,&flg));
  if (!flg) n=m+2;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nRectangular bidiagonal matrix, m=%" PetscInt_FMT " n=%" PetscInt_FMT "\n\n",m,n));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-seconds",&seconds,NULL));
  deadline = seconds;
  CHKERRQ(PetscTimeAdd(&deadline));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    col[0]=i; col[1]=i+1;
    if (i<n-1) CHKERRQ(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
    else if (i==n-1) CHKERRQ(MatSetValue(A,i,col[0],value[0],INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Compute singular values
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(SVDCreate(PETSC_COMM_WORLD,&svd));
  CHKERRQ(SVDSetOperators(svd,A,NULL));
  CHKERRQ(SVDSetWhichSingularTriplets(svd,SVD_SMALLEST));
  CHKERRQ(SVDSetTolerances(svd,PETSC_DEFAULT,1000));
  CHKERRQ(SVDSetType(svd,SVDTRLANCZOS));
  CHKERRQ(SVDSetStoppingTestFunction(svd,MyStoppingTest,&deadline,NULL));
  CHKERRQ(SVDSetFromOptions(svd));

  /* call the solver */
  CHKERRQ(SVDSolve(svd));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) CHKERRQ(SVDErrorView(svd,SVD_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
  else {
    CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
    CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
    CHKERRQ(SVDGetConvergedReason(svd,&reason));
    if (reason!=SVD_CONVERGED_USER) {
      CHKERRQ(SVDConvergedReasonView(svd,viewer));
      CHKERRQ(SVDErrorView(svd,SVD_ERROR_RELATIVE,viewer));
    } else {
      CHKERRQ(SVDGetConverged(svd,&nconv));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"SVD solve finished with %" PetscInt_FMT " converged eigenpairs; reason=%s\n",nconv,SVDConvergedReasons[reason]));
    }
    CHKERRQ(PetscViewerPopFormat(viewer));
  }
  CHKERRQ(SVDDestroy(&svd));
  CHKERRQ(MatDestroy(&A));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      args: -m 750 -seconds 0.1 -svd_max_it 10000

TEST*/
