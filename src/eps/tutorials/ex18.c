/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Solves the same problem as in ex5, but with a user-defined sorting criterion."
  "It is a standard nonsymmetric eigenproblem with real eigenvalues and the rightmost eigenvalue is known to be 1.\n"
  "This example illustrates how the user can set a custom spectrum selection.\n\n"
  "The command line options are:\n"
  "  -m <m>, where <m> = number of grid subdivisions in each dimension.\n\n";

#include <slepceps.h>

/*
   User-defined routines
*/

PetscErrorCode MyEigenSort(PetscScalar ar,PetscScalar ai,PetscScalar br,PetscScalar bi,PetscInt *r,void *ctx);
PetscErrorCode MatMarkovModel(PetscInt m,Mat A);

int main(int argc,char **argv)
{
  Mat            A;               /* operator matrix */
  EPS            eps;             /* eigenproblem solver context */
  EPSType        type;
  PetscScalar    target=0.5;
  PetscInt       N,m=15,nev;
  PetscBool      terse;
  PetscViewer    viewer;
  PetscErrorCode ierr;
  char           str[50];

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  N = m*(m+1)/2;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nMarkov Model, N=%" PetscInt_FMT " (m=%" PetscInt_FMT ")\n",N,m));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-target",&target,NULL));
  CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),target,PETSC_FALSE));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Searching closest eigenvalues to the right of %s.\n\n",str));

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

  /*
     Create eigensolver context
  */
  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));

  /*
     Set operators. In this case, it is a standard eigenvalue problem
  */
  CHKERRQ(EPSSetOperators(eps,A,NULL));
  CHKERRQ(EPSSetProblemType(eps,EPS_NHEP));

  /*
     Set the custom comparing routine in order to obtain the eigenvalues
     closest to the target on the right only
  */
  CHKERRQ(EPSSetEigenvalueComparison(eps,MyEigenSort,&target));

  /*
     Set solver parameters at runtime
  */
  CHKERRQ(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSSolve(eps));

  /*
     Optional: Get some information from the solver and display it
  */
  CHKERRQ(EPSGetType(eps,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type));
  CHKERRQ(EPSGetDimensions(eps,&nev,NULL,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %" PetscInt_FMT "\n",nev));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) {
    CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  } else {
    CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
    CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
    CHKERRQ(EPSConvergedReasonView(eps,viewer));
    CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,viewer));
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
        if (i==1) {
          CHKERRQ(MatSetValue(A,ix-1,ix,2*pd,INSERT_VALUES));
        } else {
          CHKERRQ(MatSetValue(A,ix-1,ix,pd,INSERT_VALUES));
        }
        /* east */
        if (j==1) {
          CHKERRQ(MatSetValue(A,ix-1,ix+jmax-1,2*pd,INSERT_VALUES));
        } else {
          CHKERRQ(MatSetValue(A,ix-1,ix+jmax-1,pd,INSERT_VALUES));
        }
      }
      /* south */
      pu = 0.5 - cst*(PetscReal)(i+j-3);
      if (j>1) {
        CHKERRQ(MatSetValue(A,ix-1,ix-2,pu,INSERT_VALUES));
      }
      /* west */
      if (i>1) {
        CHKERRQ(MatSetValue(A,ix-1,ix-jmax-2,pu,INSERT_VALUES));
      }
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*
    Function for user-defined eigenvalue ordering criterion.

    Given two eigenvalues ar+i*ai and br+i*bi, the subroutine must choose
    one of them as the preferred one according to the criterion.
    In this example, the preferred value is the one closest to the target,
    but on the right side.
*/
PetscErrorCode MyEigenSort(PetscScalar ar,PetscScalar ai,PetscScalar br,PetscScalar bi,PetscInt *r,void *ctx)
{
  PetscScalar target = *(PetscScalar*)ctx;
  PetscReal   da,db;
  PetscBool   aisright,bisright;

  PetscFunctionBeginUser;
  if (PetscRealPart(target) < PetscRealPart(ar)) aisright = PETSC_TRUE;
  else aisright = PETSC_FALSE;
  if (PetscRealPart(target) < PetscRealPart(br)) bisright = PETSC_TRUE;
  else bisright = PETSC_FALSE;
  if (aisright == bisright) {
    /* both are on the same side of the target */
    da = SlepcAbsEigenvalue(ar-target,ai);
    db = SlepcAbsEigenvalue(br-target,bi);
    if (da < db) *r = -1;
    else if (da > db) *r = 1;
    else *r = 0;
  } else if (aisright && !bisright) *r = -1; /* 'a' is on the right */
  else *r = 1;  /* 'b' is on the right */
  PetscFunctionReturn(0);
}

/*TEST

   test:
      suffix: 1
      args: -eps_nev 4 -terse
      requires: !single
      filter: sed -e "s/[+-]0\.0*i//g"

TEST*/
