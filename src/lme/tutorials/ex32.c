/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Solves a Lypunov equation with the shifted 2-D Laplacian.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n\n";

#include <slepclme.h>

int main(int argc,char **argv)
{
  Mat                A;           /* problem matrix */
  Mat                C,C1;        /* right-hand side */
  Mat                X,X1;        /* solution */
  LME                lme;
  PetscReal          tol,errest,error;
  PetscScalar        *u,sigma=0.0;
  PetscInt           N,n=10,m,Istart,Iend,II,maxit,its,ncv,i,j,rank=0;
  PetscErrorCode     ierr;
  PetscBool          flag;
  LMEConvergedReason reason;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  N = n*m;
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-sigma",&sigma,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-rank",&rank,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nLyapunov equation, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                       Create the 2-D Laplacian, A
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) CHKERRQ(MatSetValue(A,II,II-n,1.0,INSERT_VALUES));
    if (i<m-1) CHKERRQ(MatSetValue(A,II,II+n,1.0,INSERT_VALUES));
    if (j>0) CHKERRQ(MatSetValue(A,II,II-1,1.0,INSERT_VALUES));
    if (j<n-1) CHKERRQ(MatSetValue(A,II,II+1,1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(A,II,II,-4.0-sigma,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create a low-rank Mat to store the right-hand side C = C1*C1'
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C1));
  CHKERRQ(MatSetSizes(C1,PETSC_DECIDE,PETSC_DECIDE,N,2));
  CHKERRQ(MatSetType(C1,MATDENSE));
  CHKERRQ(MatSetUp(C1));
  CHKERRQ(MatGetOwnershipRange(C1,&Istart,&Iend));
  CHKERRQ(MatDenseGetArray(C1,&u));
  for (i=Istart;i<Iend;i++) {
    if (i<N/2) u[i-Istart] = 1.0;
    if (i==0) u[i+Iend-2*Istart] = -2.0;
    if (i==1) u[i+Iend-2*Istart] = -1.0;
    if (i==2) u[i+Iend-2*Istart] = -1.0;
  }
  CHKERRQ(MatDenseRestoreArray(C1,&u));
  CHKERRQ(MatAssemblyBegin(C1,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C1,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCreateLRC(NULL,C1,NULL,NULL,&C));
  CHKERRQ(MatDestroy(&C1));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create the matrix equation solver context
  */
  CHKERRQ(LMECreate(PETSC_COMM_WORLD,&lme));

  /*
     Set the type of equation
  */
  CHKERRQ(LMESetProblemType(lme,LME_LYAPUNOV));

  /*
     Set the matrix coefficients, the right-hand side, and the solution.
     In this case, it is a Lyapunov equation A*X+X*A'=-C where both
     C and X are symmetric and low-rank, C=C1*C1', X=X1*X1'
  */
  CHKERRQ(LMESetCoefficients(lme,A,NULL,NULL,NULL));
  CHKERRQ(LMESetRHS(lme,C));

  if (rank) {  /* Create X only if the user has specified a nonzero value of rank */
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Computing a solution with prescribed rank=%" PetscInt_FMT "\n",rank));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&X1));
    CHKERRQ(MatSetSizes(X1,PETSC_DECIDE,PETSC_DECIDE,N,rank));
    CHKERRQ(MatSetType(X1,MATDENSE));
    CHKERRQ(MatSetUp(X1));
    CHKERRQ(MatAssemblyBegin(X1,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(X1,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatCreateLRC(NULL,X1,NULL,NULL,&X));
    CHKERRQ(MatDestroy(&X1));
    CHKERRQ(LMESetSolution(lme,X));
    CHKERRQ(MatDestroy(&X));
  }

  /*
     (Optional) Set other solver options
  */
  CHKERRQ(LMESetTolerances(lme,1e-07,PETSC_DEFAULT));

  /*
     Set solver parameters at runtime
  */
  CHKERRQ(LMESetFromOptions(lme));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                   Solve the matrix equation, A*X+X*A'=-C
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(LMESolve(lme));
  CHKERRQ(LMEGetConvergedReason(lme,&reason));
  PetscCheck(reason>=0,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Solver did not converge");

  if (!rank) {  /* X1 was created by the solver, so extract it and see how many columns it has */
    CHKERRQ(LMEGetSolution(lme,&X));
    CHKERRQ(MatLRCGetMats(X,NULL,&X1,NULL,NULL));
    CHKERRQ(MatGetSize(X1,NULL,&rank));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," The solver has computed a solution with rank=%" PetscInt_FMT "\n",rank));
  }

  /*
     Optional: Get some information from the solver and display it
  */
  CHKERRQ(LMEGetIterationNumber(lme,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %" PetscInt_FMT "\n",its));
  CHKERRQ(LMEGetDimensions(lme,&ncv));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Subspace dimension: %" PetscInt_FMT "\n",ncv));
  CHKERRQ(LMEGetTolerances(lme,&tol,&maxit));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%" PetscInt_FMT "\n",(double)tol,maxit));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Compute residual error
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(LMEGetErrorEstimate(lme,&errest));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Error estimate reported by the solver: %.4g\n",(double)errest));
  if (n<=150) {
    CHKERRQ(LMEComputeError(lme,&error));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Computed residual norm: %.4g\n\n",(double)error));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Matrix too large to compute residual norm\n\n"));
  }

  /*
     Free work space
  */
  CHKERRQ(LMEDestroy(&lme));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&C));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      requires: !single

   test:
      suffix: 2
      args: -rank 40
      requires: !single

TEST*/
