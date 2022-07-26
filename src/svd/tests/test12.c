/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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
  PetscCall(SVDStoppingBasic(svd,its,max_it,nconv,nev,reason,NULL));
  if (*reason==SVD_CONVERGED_ITERATING) {
    /* check if deadline has expired */
    PetscCall(PetscTime(&now));
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

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,&flg));
  if (!flg) n=m+2;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nRectangular bidiagonal matrix, m=%" PetscInt_FMT " n=%" PetscInt_FMT "\n\n",m,n));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-seconds",&seconds,NULL));
  deadline = seconds;
  PetscCall(PetscTimeAdd(&deadline));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    col[0]=i; col[1]=i+1;
    if (i<n-1) PetscCall(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
    else if (i==n-1) PetscCall(MatSetValue(A,i,col[0],value[0],INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Compute singular values
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(SVDCreate(PETSC_COMM_WORLD,&svd));
  PetscCall(SVDSetOperators(svd,A,NULL));
  PetscCall(SVDSetWhichSingularTriplets(svd,SVD_SMALLEST));
  PetscCall(SVDSetTolerances(svd,PETSC_DEFAULT,1000));
  PetscCall(SVDSetType(svd,SVDTRLANCZOS));
  PetscCall(SVDSetStoppingTestFunction(svd,MyStoppingTest,&deadline,NULL));
  PetscCall(SVDSetFromOptions(svd));

  /* call the solver */
  PetscCall(SVDSolve(svd));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  PetscCall(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) PetscCall(SVDErrorView(svd,SVD_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
  else {
    PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
    PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
    PetscCall(SVDGetConvergedReason(svd,&reason));
    if (reason!=SVD_CONVERGED_USER) {
      PetscCall(SVDConvergedReasonView(svd,viewer));
      PetscCall(SVDErrorView(svd,SVD_ERROR_RELATIVE,viewer));
    } else {
      PetscCall(SVDGetConverged(svd,&nconv));
      PetscCall(PetscViewerASCIIPrintf(viewer,"SVD solve finished with %" PetscInt_FMT " converged eigenpairs; reason=%s\n",nconv,SVDConvergedReasons[reason]));
    }
    PetscCall(PetscViewerPopFormat(viewer));
  }
  PetscCall(SVDDestroy(&svd));
  PetscCall(MatDestroy(&A));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -m 750 -seconds 0.1 -svd_max_it 10000

TEST*/
