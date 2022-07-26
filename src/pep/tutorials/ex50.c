/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  N = n*m;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nQuadratic Eigenproblem, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the matrices for (k^2*A_2+k*A_1+A_0)x=0, and their approximations
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* A[0] is the 2-D Laplacian */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A[0]));
  PetscCall(MatSetSizes(A[0],PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(A[0]));
  PetscCall(MatSetUp(A[0]));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&P[0]));
  PetscCall(MatSetSizes(P[0],PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(P[0]));
  PetscCall(MatSetUp(P[0]));

  PetscCall(MatGetOwnershipRange(A[0],&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) PetscCall(MatSetValue(A[0],II,II-n,-1.0,INSERT_VALUES));
    if (i<m-1) PetscCall(MatSetValue(A[0],II,II+n,-1.0,INSERT_VALUES));
    if (j>0) PetscCall(MatSetValue(A[0],II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) PetscCall(MatSetValue(A[0],II,II+1,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(A[0],II,II,4.0,INSERT_VALUES));
    if (j>0) PetscCall(MatSetValue(P[0],II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) PetscCall(MatSetValue(P[0],II,II+1,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(P[0],II,II,4.0,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A[0],MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A[0],MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(P[0],MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P[0],MAT_FINAL_ASSEMBLY));

  /* A[1] is the 1-D Laplacian on horizontal lines */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A[1]));
  PetscCall(MatSetSizes(A[1],PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(A[1]));
  PetscCall(MatSetUp(A[1]));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&P[1]));
  PetscCall(MatSetSizes(P[1],PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(P[1]));
  PetscCall(MatSetUp(P[1]));

  PetscCall(MatGetOwnershipRange(A[1],&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (j>0) PetscCall(MatSetValue(A[1],II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) PetscCall(MatSetValue(A[1],II,II+1,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(A[1],II,II,2.0,INSERT_VALUES));
    PetscCall(MatSetValue(P[1],II,II,2.0,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A[1],MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A[1],MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(P[1],MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P[1],MAT_FINAL_ASSEMBLY));

  /* A[2] is a diagonal matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A[2]));
  PetscCall(MatSetSizes(A[2],PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(A[2]));
  PetscCall(MatSetUp(A[2]));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&P[2]));
  PetscCall(MatSetSizes(P[2],PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(P[2]));
  PetscCall(MatSetUp(P[2]));

  PetscCall(MatGetOwnershipRange(A[2],&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    PetscCall(MatSetValue(A[2],II,II,(PetscReal)(II+1),INSERT_VALUES));
    PetscCall(MatSetValue(P[2],II,II,(PetscReal)(II+1),INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A[2],MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A[2],MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(P[2],MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P[2],MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PEPCreate(PETSC_COMM_WORLD,&pep));
  PetscCall(PEPSetOperators(pep,3,A));
  PetscCall(PEPSetProblemType(pep,PEP_HERMITIAN));

  PetscCall(PEPGetST(pep,&st));
  PetscCall(STSetType(st,STSINVERT));
  PetscCall(STSetSplitPreconditioner(st,3,P,SUBSET_NONZERO_PATTERN));

  PetscCall(PEPSetTarget(pep,-2.0));
  PetscCall(PEPSetWhichEigenpairs(pep,PEP_TARGET_MAGNITUDE));

  /*
     Set solver parameters at runtime
  */
  PetscCall(PEPSetFromOptions(pep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Solve the eigensystem, display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PEPSolve(pep));
  /* show detailed info unless -terse option is given by user */
  PetscCall(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) PetscCall(PEPErrorView(pep,PEP_ERROR_BACKWARD,NULL));
  else {
    PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    PetscCall(PEPConvergedReasonView(pep,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PEPErrorView(pep,PEP_ERROR_BACKWARD,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(PEPDestroy(&pep));
  PetscCall(MatDestroy(&A[0]));
  PetscCall(MatDestroy(&A[1]));
  PetscCall(MatDestroy(&A[2]));
  PetscCall(MatDestroy(&P[0]));
  PetscCall(MatDestroy(&P[1]));
  PetscCall(MatDestroy(&P[2]));
  PetscCall(SlepcFinalize());
  return 0;
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
      timeoutfactor: 2
      test:
         suffix: 2
      test:
         suffix: 2_par
         nsize: 2
         args: -pep_ciss_partitions 2

TEST*/
