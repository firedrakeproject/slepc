/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2020, Universitat Politecnica de Valencia, Spain

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
  PetscErrorCode ierr;
  PetscInt       N = n*m,i,j,II,Istart,Iend,nloc;
  PetscMPIInt    rank;

  PetscFunctionBeginUser;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  nloc = PETSC_DECIDE;
  ierr = PetscSplitOwnership(comm,&nloc,&N);CHKERRQ(ierr);
  if (rank==0) nloc += shift;
  else if (rank==1) nloc -= shift;

  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,nloc,nloc,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*A);CHKERRQ(ierr);
  ierr = MatSetUp(*A);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(*A,&Istart,&Iend);CHKERRQ(ierr);
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) { ierr = MatSetValue(*A,II,II-n,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (i<m-1) { ierr = MatSetValue(*A,II,II+n,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (j>0) { ierr = MatSetValue(*A,II,II-1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (j<n-1) { ierr = MatSetValue(*A,II,II+1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    ierr = MatSetValue(*A,II,II,4.0,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size==1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This example requires at least two processes");
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag);CHKERRQ(ierr);
  N = n*m;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n2-D Laplacian Eigenproblem, N=%D (%Dx%D grid)\n\n",N,n,m);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create 2-D Laplacian matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = Laplacian(PETSC_COMM_WORLD,n,m,1,&A);CHKERRQ(ierr);
  ierr = Laplacian(PETSC_COMM_WORLD,n,m,-1,&B);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Create the eigensolver, set options and solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscPrintf(PETSC_COMM_WORLD,"First solve:\n\n");CHKERRQ(ierr);
  ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);
  ierr = EPSSetOperators(eps,A,NULL);CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps,EPS_HEP);CHKERRQ(ierr);
  ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL);CHKERRQ(ierr);
  ierr = EPSSetDimensions(eps,nev,PETSC_DEFAULT,PETSC_DEFAULT);
  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

  ierr = EPSSolve(eps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution of first solve
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscOptionsHasName(NULL,NULL,"-terse",&terse);CHKERRQ(ierr);
  if (terse) {
    ierr = EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
    ierr = EPSConvergedReasonView(eps,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                       Solve with second matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSecond solve:\n\n");CHKERRQ(ierr);
  /*ierr = EPSReset(eps);CHKERRQ(ierr);*/  /* not required, will be called in EPSSetOperators() */
  ierr = EPSSetOperators(eps,B,NULL);CHKERRQ(ierr);
  ierr = EPSSolve(eps);CHKERRQ(ierr);

  if (terse) {
    ierr = EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
    ierr = EPSConvergedReasonView(eps,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  ierr = EPSDestroy(&eps);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      nsize: 2
      args: -eps_type {{krylovschur arnoldi lanczos lobpcg lapack}} -terse
      requires: !single

TEST*/
