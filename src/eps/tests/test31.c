/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test STFILTER interface functions.\n\n"
  "Based on ex2.\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n\n";

#include <slepceps.h>

int main(int argc,char **argv)
{
  Mat            A;
  EPS            eps;
  ST             st;
  PetscInt       N,n=10,m,Istart,Iend,II,i,j,degree;
  PetscBool      flag,modify=PETSC_FALSE,terse;
  PetscReal      inta,intb,rleft,rright;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  N = n*m;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n2-D Laplacian Eigenproblem, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-modify",&modify,&flag));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Create the 2-D Laplacian
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) CHKERRQ(MatSetValue(A,II,II-n,-1.0,INSERT_VALUES));
    if (i<m-1) CHKERRQ(MatSetValue(A,II,II+n,-1.0,INSERT_VALUES));
    if (j>0) CHKERRQ(MatSetValue(A,II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) CHKERRQ(MatSetValue(A,II,II+1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(A,II,II,4.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetOperators(eps,A,NULL));
  CHKERRQ(EPSSetProblemType(eps,EPS_HEP));
  CHKERRQ(EPSSetType(eps,EPSKRYLOVSCHUR));
  CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_ALL));
  CHKERRQ(EPSSetInterval(eps,0.5,1.3));
  CHKERRQ(EPSGetST(eps,&st));
  CHKERRQ(STSetType(st,STFILTER));
  CHKERRQ(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Solve the problem and display the solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSSolve(eps));

  /* print filter information */
  CHKERRQ(PetscObjectTypeCompare((PetscObject)st,STFILTER,&flag));
  if (flag) {
    CHKERRQ(STFilterGetDegree(st,&degree));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Filter degree: %" PetscInt_FMT "\n",degree));
    CHKERRQ(STFilterGetInterval(st,&inta,&intb));
    CHKERRQ(STFilterGetRange(st,&rleft,&rright));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Requested interval: [%g,%g],  range: [%g,%g]\n\n",(double)inta,(double)intb,(double)rleft,(double)rright));
  }

  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  else {
    CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    CHKERRQ(EPSConvergedReasonView(eps,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
              Solve the problem again after changing the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (modify) {
    CHKERRQ(MatSetValue(A,0,0,0.3,INSERT_VALUES));
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(EPSSetOperators(eps,A,NULL));
    CHKERRQ(EPSSolve(eps));
    CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
    if (terse) CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
    else {
      CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
      CHKERRQ(EPSConvergedReasonView(eps,PETSC_VIEWER_STDOUT_WORLD));
      CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
      CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
    }
  }

  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      args: -terse
      filter: sed -e "s/0.161982,7.83797/0.162007,7.83897/"
      requires: !single

   test:
      suffix: 2
      args: -modify -st_filter_range -0.5,8 -terse
      requires: !single

   test:
      suffix: 3
      args: -modify -terse
      filter: sed -e "s/0.161982,7.83797/0.162007,7.83897/"
      requires: !single

TEST*/
