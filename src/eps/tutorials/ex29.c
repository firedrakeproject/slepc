/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Solves the same problem as in ex5, with a user-defined stopping test."
  "It is a standard nonsymmetric eigenproblem with real eigenvalues and the rightmost eigenvalue is known to be 1.\n"
  "This example illustrates how the user can set a custom stopping test function.\n\n"
  "The command line options are:\n"
  "  -m <m>, where <m> = number of grid subdivisions in each dimension.\n"
  "  -seconds <s>, where <s> = maximum time in seconds allowed for computation.\n\n";

#include <slepceps.h>
#include <petsctime.h>

/*
   User-defined routines
*/

PetscErrorCode MyStoppingTest(EPS,PetscInt,PetscInt,PetscInt,PetscInt,EPSConvergedReason*,void*);
PetscErrorCode MatMarkovModel(PetscInt,Mat);

int main(int argc,char **argv)
{
  Mat                A;               /* operator matrix */
  EPS                eps;             /* eigenproblem solver context */
  PetscReal          seconds=2.5;     /* maximum time allowed for computation */
  PetscLogDouble     deadline;        /* time to abort computation */
  PetscInt           N,m=15,nconv;
  PetscBool          terse;
  PetscViewer        viewer;
  EPSConvergedReason reason;
  PetscErrorCode     ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  N = m*(m+1)/2;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nMarkov Model, N=%" PetscInt_FMT " (m=%" PetscInt_FMT ")\n",N,m));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-seconds",&seconds,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Maximum time for computation is set to %g seconds.\n\n",(double)seconds));
  deadline = seconds;
  CHKERRQ(PetscTimeAdd(&deadline));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the operator matrix that defines the eigensystem, Ax=kx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatMarkovModel(m,A));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetOperators(eps,A,NULL));
  CHKERRQ(EPSSetProblemType(eps,EPS_NHEP));
  CHKERRQ(EPSSetStoppingTestFunction(eps,MyStoppingTest,&deadline,NULL));
  CHKERRQ(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSSolve(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  else {
    CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
    CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
    CHKERRQ(EPSGetConvergedReason(eps,&reason));
    if (reason!=EPS_CONVERGED_USER) {
      CHKERRQ(EPSConvergedReasonView(eps,viewer));
      CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,viewer));
    } else {
      CHKERRQ(EPSGetConverged(eps,&nconv));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"Eigensolve finished with %" PetscInt_FMT " converged eigenpairs; reason=%s\n",nconv,EPSConvergedReasons[reason]));
    }
    CHKERRQ(PetscViewerPopFormat(viewer));
  }
  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  ierr = SlepcFinalize();
  return ierr;
}

/*
    Matrix generator for a Markov model of a random walk on a triangular grid.

    This subroutine generates a test matrix that models a random walk on a
    triangular grid. This test example was used by G. W. Stewart ["{SRRIT} - a
    FORTRAN subroutine to calculate the dominant invariant subspaces of a real
    matrix", Tech. report. TR-514, University of Maryland (1978).] and in a few
    papers on eigenvalue problems by Y. Saad [see e.g. LAA, vol. 34, pp. 269-295
    (1980) ]. These matrices provide reasonably easy test problems for eigenvalue
    algorithms. The transpose of the matrix  is stochastic and so it is known
    that one is an exact eigenvalue. One seeks the eigenvector of the transpose
    associated with the eigenvalue unity. The problem is to calculate the steady
    state probability distribution of the system, which is the eigevector
    associated with the eigenvalue one and scaled in such a way that the sum all
    the components is equal to one.

    Note: the code will actually compute the transpose of the stochastic matrix
    that contains the transition probabilities.
*/
PetscErrorCode MatMarkovModel(PetscInt m,Mat A)
{
  const PetscReal cst = 0.5/(PetscReal)(m-1);
  PetscReal       pd,pu;
  PetscInt        Istart,Iend,i,j,jmax,ix=0;

  PetscFunctionBeginUser;
  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=1;i<=m;i++) {
    jmax = m-i+1;
    for (j=1;j<=jmax;j++) {
      ix = ix + 1;
      if (ix-1<Istart || ix>Iend) continue;  /* compute only owned rows */
      if (j!=jmax) {
        pd = cst*(PetscReal)(i+j-1);
        /* north */
        if (i==1) CHKERRQ(MatSetValue(A,ix-1,ix,2*pd,INSERT_VALUES));
        else CHKERRQ(MatSetValue(A,ix-1,ix,pd,INSERT_VALUES));
        /* east */
        if (j==1) CHKERRQ(MatSetValue(A,ix-1,ix+jmax-1,2*pd,INSERT_VALUES));
        else CHKERRQ(MatSetValue(A,ix-1,ix+jmax-1,pd,INSERT_VALUES));
      }
      /* south */
      pu = 0.5 - cst*(PetscReal)(i+j-3);
      if (j>1) CHKERRQ(MatSetValue(A,ix-1,ix-2,pu,INSERT_VALUES));
      /* west */
      if (i>1) CHKERRQ(MatSetValue(A,ix-1,ix-jmax-2,pu,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*
    Function for user-defined stopping test.

    Checks that the computing time has not exceeded the deadline.
*/
PetscErrorCode MyStoppingTest(EPS eps,PetscInt its,PetscInt max_it,PetscInt nconv,PetscInt nev,EPSConvergedReason *reason,void *ctx)
{
  PetscLogDouble now,deadline = *(PetscLogDouble*)ctx;

  PetscFunctionBeginUser;
  /* check if usual termination conditions are met */
  CHKERRQ(EPSStoppingBasic(eps,its,max_it,nconv,nev,reason,NULL));
  if (*reason==EPS_CONVERGED_ITERATING) {
    /* check if deadline has expired */
    CHKERRQ(PetscTime(&now));
    if (now>deadline) *reason = EPS_CONVERGED_USER;
  }
  PetscFunctionReturn(0);
}

/*TEST

   test:
      suffix: 1
      args: -m 350 -seconds 0.6

TEST*/
