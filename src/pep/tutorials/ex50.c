/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "User-defined split preconditioner when solving a quadratic eigenproblem.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n\n";

#include <slepcpep.h>

int main(int argc,char **argv)
{
  Mat            A[3],P[3];      /* problem matrices and split preconditioner matrices */
  PEP            pep;            /* polynomial eigenproblem solver context */
  ST             st;
  PetscInt       N,n=10,m,Istart,Iend,II,i,j;
  PetscBool      flag,terse;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  N = n*m;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nQuadratic Eigenproblem, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the matrices for (k^2*A_2+k*A_1+A_0)x=0, and their approximations
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* A[0] is the 2-D Laplacian */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A[0]));
  CHKERRQ(MatSetSizes(A[0],PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(A[0]));
  CHKERRQ(MatSetUp(A[0]));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&P[0]));
  CHKERRQ(MatSetSizes(P[0],PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(P[0]));
  CHKERRQ(MatSetUp(P[0]));

  CHKERRQ(MatGetOwnershipRange(A[0],&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) CHKERRQ(MatSetValue(A[0],II,II-n,-1.0,INSERT_VALUES));
    if (i<m-1) CHKERRQ(MatSetValue(A[0],II,II+n,-1.0,INSERT_VALUES));
    if (j>0) CHKERRQ(MatSetValue(A[0],II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) CHKERRQ(MatSetValue(A[0],II,II+1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(A[0],II,II,4.0,INSERT_VALUES));
    if (j>0) CHKERRQ(MatSetValue(P[0],II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) CHKERRQ(MatSetValue(P[0],II,II+1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(P[0],II,II,4.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A[0],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A[0],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(P[0],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(P[0],MAT_FINAL_ASSEMBLY));

  /* A[1] is the 1-D Laplacian on horizontal lines */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A[1]));
  CHKERRQ(MatSetSizes(A[1],PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(A[1]));
  CHKERRQ(MatSetUp(A[1]));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&P[1]));
  CHKERRQ(MatSetSizes(P[1],PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(P[1]));
  CHKERRQ(MatSetUp(P[1]));

  CHKERRQ(MatGetOwnershipRange(A[1],&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (j>0) CHKERRQ(MatSetValue(A[1],II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) CHKERRQ(MatSetValue(A[1],II,II+1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(A[1],II,II,2.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(P[1],II,II,2.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A[1],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A[1],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(P[1],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(P[1],MAT_FINAL_ASSEMBLY));

  /* A[2] is a diagonal matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A[2]));
  CHKERRQ(MatSetSizes(A[2],PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(A[2]));
  CHKERRQ(MatSetUp(A[2]));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&P[2]));
  CHKERRQ(MatSetSizes(P[2],PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(P[2]));
  CHKERRQ(MatSetUp(P[2]));

  CHKERRQ(MatGetOwnershipRange(A[2],&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    CHKERRQ(MatSetValue(A[2],II,II,(PetscReal)(II+1),INSERT_VALUES));
    CHKERRQ(MatSetValue(P[2],II,II,(PetscReal)(II+1),INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A[2],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A[2],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(P[2],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(P[2],MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PEPCreate(PETSC_COMM_WORLD,&pep));
  CHKERRQ(PEPSetOperators(pep,3,A));
  CHKERRQ(PEPSetProblemType(pep,PEP_HERMITIAN));

  CHKERRQ(PEPGetST(pep,&st));
  CHKERRQ(STSetType(st,STSINVERT));
  CHKERRQ(STSetSplitPreconditioner(st,3,P,SUBSET_NONZERO_PATTERN));

  CHKERRQ(PEPSetTarget(pep,-2.0));
  CHKERRQ(PEPSetWhichEigenpairs(pep,PEP_TARGET_MAGNITUDE));

  /*
     Set solver parameters at runtime
  */
  CHKERRQ(PEPSetFromOptions(pep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Solve the eigensystem, display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PEPSolve(pep));
  /* show detailed info unless -terse option is given by user */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) CHKERRQ(PEPErrorView(pep,PEP_ERROR_BACKWARD,NULL));
  else {
    CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    CHKERRQ(PEPConvergedReasonView(pep,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PEPErrorView(pep,PEP_ERROR_BACKWARD,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(PEPDestroy(&pep));
  CHKERRQ(MatDestroy(&A[0]));
  CHKERRQ(MatDestroy(&A[1]));
  CHKERRQ(MatDestroy(&A[2]));
  CHKERRQ(MatDestroy(&P[0]));
  CHKERRQ(MatDestroy(&P[1]));
  CHKERRQ(MatDestroy(&P[2]));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      args: -pep_nev 4 -pep_ncv 28 -n 12 -terse
      output_file: output/ex50_1.out
      requires: double
      test:
         suffix: 1
         args: -pep_type {{toar qarnoldi}}
      test:
         suffix: 1_linear
         args: -pep_type linear -pep_general

   testset:
      args: -pep_all -n 12 -pep_type ciss -rg_type ellipse -rg_ellipse_center -1+1.5i -rg_ellipse_radius .3 -terse
      output_file: output/ex50_2.out
      requires: complex double
      test:
         suffix: 2
      test:
         suffix: 2_par
         nsize: 2
         args: -pep_ciss_partitions 2

TEST*/
