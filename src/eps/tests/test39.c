/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Tests multiple calls to EPSSolve with matrices of different local size.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n\n";

#include <slepceps.h>

/*
   Create 2-D Laplacian matrix
*/
PetscErrorCode Laplacian(MPI_Comm comm,PetscInt n,PetscInt m,PetscInt shift,Mat *A)
{
  PetscInt       N = n*m,i,j,II,Istart,Iend,nloc;
  PetscMPIInt    rank;

  PetscFunctionBeginUser;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  nloc = PETSC_DECIDE;
  CHKERRQ(PetscSplitOwnership(comm,&nloc,&N));
  if (rank==0) nloc += shift;
  else if (rank==1) nloc -= shift;

  CHKERRQ(MatCreate(comm,A));
  CHKERRQ(MatSetSizes(*A,nloc,nloc,N,N));
  CHKERRQ(MatSetFromOptions(*A));
  CHKERRQ(MatSetUp(*A));
  CHKERRQ(MatGetOwnershipRange(*A,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) CHKERRQ(MatSetValue(*A,II,II-n,-1.0,INSERT_VALUES));
    if (i<m-1) CHKERRQ(MatSetValue(*A,II,II+n,-1.0,INSERT_VALUES));
    if (j>0) CHKERRQ(MatSetValue(*A,II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) CHKERRQ(MatSetValue(*A,II,II+1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(*A,II,II,4.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Mat            A,B;
  EPS            eps;
  PetscInt       N,n=10,m=11,nev=3;
  PetscMPIInt    size;
  PetscBool      flag,terse;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size>1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This example requires at least two processes");
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  N = n*m;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n2-D Laplacian Eigenproblem, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create 2-D Laplacian matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(Laplacian(PETSC_COMM_WORLD,n,m,1,&A));
  CHKERRQ(Laplacian(PETSC_COMM_WORLD,n,m,-1,&B));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Create the eigensolver, set options and solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"First solve:\n\n"));
  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetOperators(eps,A,NULL));
  CHKERRQ(EPSSetProblemType(eps,EPS_HEP));
  CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL));
  CHKERRQ(EPSSetDimensions(eps,nev,PETSC_DEFAULT,PETSC_DEFAULT));
  CHKERRQ(EPSSetFromOptions(eps));

  CHKERRQ(EPSSolve(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution of first solve
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) {
    CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  } else {
    CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    CHKERRQ(EPSConvergedReasonView(eps,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                       Solve with second matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nSecond solve:\n\n"));
  /*CHKERRQ(EPSReset(eps));*/  /* not required, will be called in EPSSetOperators() */
  CHKERRQ(EPSSetOperators(eps,B,NULL));
  CHKERRQ(EPSSolve(eps));

  if (terse) {
    CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  } else {
    CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    CHKERRQ(EPSConvergedReasonView(eps,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }

  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      nsize: 2
      requires: !single
      output_file: output/test39_1.out
      test:
         suffix: 1
         args: -eps_type {{krylovschur arnoldi lobpcg lapack}} -terse
      test:
         suffix: 1_lanczos
         args: -eps_type lanczos -eps_lanczos_reorthog local -terse

TEST*/
