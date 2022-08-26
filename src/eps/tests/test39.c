/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  nloc = PETSC_DECIDE;
  PetscCall(PetscSplitOwnership(comm,&nloc,&N));
  if (rank==0) nloc += shift;
  else if (rank==1) nloc -= shift;

  PetscCall(MatCreate(comm,A));
  PetscCall(MatSetSizes(*A,nloc,nloc,N,N));
  PetscCall(MatSetFromOptions(*A));
  PetscCall(MatSetUp(*A));
  PetscCall(MatGetOwnershipRange(*A,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) PetscCall(MatSetValue(*A,II,II-n,-1.0,INSERT_VALUES));
    if (i<m-1) PetscCall(MatSetValue(*A,II,II+n,-1.0,INSERT_VALUES));
    if (j>0) PetscCall(MatSetValue(*A,II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) PetscCall(MatSetValue(*A,II,II+1,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(*A,II,II,4.0,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Mat            A,B;
  EPS            eps;
  PetscInt       N,n=10,m=11,nev=3;
  PetscMPIInt    size;
  PetscBool      flag,terse;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size>1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This example requires at least two processes");
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  N = n*m;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n2-D Laplacian Eigenproblem, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create 2-D Laplacian matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(Laplacian(PETSC_COMM_WORLD,n,m,1,&A));
  PetscCall(Laplacian(PETSC_COMM_WORLD,n,m,-1,&B));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Create the eigensolver, set options and solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"First solve:\n\n"));
  PetscCall(EPSCreate(PETSC_COMM_WORLD,&eps));
  PetscCall(EPSSetOperators(eps,A,NULL));
  PetscCall(EPSSetProblemType(eps,EPS_HEP));
  PetscCall(EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL));
  PetscCall(EPSSetDimensions(eps,nev,PETSC_DEFAULT,PETSC_DEFAULT));
  PetscCall(EPSSetFromOptions(eps));

  PetscCall(EPSSolve(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution of first solve
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) PetscCall(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  else {
    PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    PetscCall(EPSConvergedReasonView(eps,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                       Solve with second matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nSecond solve:\n\n"));
  /* PetscCall(EPSReset(eps)); */  /* not required, will be called in EPSSetOperators() */
  PetscCall(EPSSetOperators(eps,B,NULL));
  PetscCall(EPSSolve(eps));

  if (terse) PetscCall(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  else {
    PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    PetscCall(EPSConvergedReasonView(eps,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }

  PetscCall(EPSDestroy(&eps));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(SlepcFinalize());
  return 0;
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
