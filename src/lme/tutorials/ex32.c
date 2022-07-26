/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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
  PetscBool          flag;
  LMEConvergedReason reason;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  N = n*m;
  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-sigma",&sigma,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-rank",&rank,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nLyapunov equation, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                       Create the 2-D Laplacian, A
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) PetscCall(MatSetValue(A,II,II-n,1.0,INSERT_VALUES));
    if (i<m-1) PetscCall(MatSetValue(A,II,II+n,1.0,INSERT_VALUES));
    if (j>0) PetscCall(MatSetValue(A,II,II-1,1.0,INSERT_VALUES));
    if (j<n-1) PetscCall(MatSetValue(A,II,II+1,1.0,INSERT_VALUES));
    PetscCall(MatSetValue(A,II,II,-4.0-sigma,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create a low-rank Mat to store the right-hand side C = C1*C1'
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&C1));
  PetscCall(MatSetSizes(C1,PETSC_DECIDE,PETSC_DECIDE,N,2));
  PetscCall(MatSetType(C1,MATDENSE));
  PetscCall(MatSetUp(C1));
  PetscCall(MatGetOwnershipRange(C1,&Istart,&Iend));
  PetscCall(MatDenseGetArray(C1,&u));
  for (i=Istart;i<Iend;i++) {
    if (i<N/2) u[i-Istart] = 1.0;
    if (i==0) u[i+Iend-2*Istart] = -2.0;
    if (i==1) u[i+Iend-2*Istart] = -1.0;
    if (i==2) u[i+Iend-2*Istart] = -1.0;
  }
  PetscCall(MatDenseRestoreArray(C1,&u));
  PetscCall(MatAssemblyBegin(C1,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C1,MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateLRC(NULL,C1,NULL,NULL,&C));
  PetscCall(MatDestroy(&C1));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create the matrix equation solver context
  */
  PetscCall(LMECreate(PETSC_COMM_WORLD,&lme));

  /*
     Set the type of equation
  */
  PetscCall(LMESetProblemType(lme,LME_LYAPUNOV));

  /*
     Set the matrix coefficients, the right-hand side, and the solution.
     In this case, it is a Lyapunov equation A*X+X*A'=-C where both
     C and X are symmetric and low-rank, C=C1*C1', X=X1*X1'
  */
  PetscCall(LMESetCoefficients(lme,A,NULL,NULL,NULL));
  PetscCall(LMESetRHS(lme,C));

  if (rank) {  /* Create X only if the user has specified a nonzero value of rank */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," Computing a solution with prescribed rank=%" PetscInt_FMT "\n",rank));
    PetscCall(MatCreate(PETSC_COMM_WORLD,&X1));
    PetscCall(MatSetSizes(X1,PETSC_DECIDE,PETSC_DECIDE,N,rank));
    PetscCall(MatSetType(X1,MATDENSE));
    PetscCall(MatSetUp(X1));
    PetscCall(MatAssemblyBegin(X1,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(X1,MAT_FINAL_ASSEMBLY));
    PetscCall(MatCreateLRC(NULL,X1,NULL,NULL,&X));
    PetscCall(MatDestroy(&X1));
    PetscCall(LMESetSolution(lme,X));
    PetscCall(MatDestroy(&X));
  }

  /*
     (Optional) Set other solver options
  */
  PetscCall(LMESetTolerances(lme,1e-07,PETSC_DEFAULT));

  /*
     Set solver parameters at runtime
  */
  PetscCall(LMESetFromOptions(lme));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                   Solve the matrix equation, A*X+X*A'=-C
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(LMESolve(lme));
  PetscCall(LMEGetConvergedReason(lme,&reason));
  PetscCheck(reason>=0,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Solver did not converge");

  if (!rank) {  /* X1 was created by the solver, so extract it and see how many columns it has */
    PetscCall(LMEGetSolution(lme,&X));
    PetscCall(MatLRCGetMats(X,NULL,&X1,NULL,NULL));
    PetscCall(MatGetSize(X1,NULL,&rank));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," The solver has computed a solution with rank=%" PetscInt_FMT "\n",rank));
  }

  /*
     Optional: Get some information from the solver and display it
  */
  PetscCall(LMEGetIterationNumber(lme,&its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %" PetscInt_FMT "\n",its));
  PetscCall(LMEGetDimensions(lme,&ncv));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Subspace dimension: %" PetscInt_FMT "\n",ncv));
  PetscCall(LMEGetTolerances(lme,&tol,&maxit));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%" PetscInt_FMT "\n",(double)tol,maxit));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Compute residual error
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(LMEGetErrorEstimate(lme,&errest));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Error estimate reported by the solver: %.4g\n",(double)errest));
  if (n<=150) {
    PetscCall(LMEComputeError(lme,&error));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," Computed residual norm: %.4g\n\n",(double)error));
  } else PetscCall(PetscPrintf(PETSC_COMM_WORLD," Matrix too large to compute residual norm\n\n"));

  /*
     Free work space
  */
  PetscCall(LMEDestroy(&lme));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&C));
  PetscCall(SlepcFinalize());
  return 0;
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
