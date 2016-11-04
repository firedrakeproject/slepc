/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.

   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Solves a Lypunov equation with the 2-D Laplacian.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n\n";

#include <slepclme.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat                A;           /* problem matrix */
  BV                 C1,X1;
  Vec                t,v;
  LME                lme;
  PetscReal          tol,errest,error;
  PetscInt           N,n=10,m,Istart,Iend,II,maxit,its,ncv,i,j,rank=5;
  PetscErrorCode     ierr;
  PetscBool          flag;
  LMEConvergedReason reason;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag);CHKERRQ(ierr);
  if (!flag) m=n;
  N = n*m;
  ierr = PetscOptionsGetInt(NULL,NULL,"-rank",&rank,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nLyapunov equation, N=%D (%Dx%D grid), rank=%d\n\n",N,n,m,rank);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                       Create the 2-D Laplacian, A
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) { ierr = MatSetValue(A,II,II-n,1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (i<m-1) { ierr = MatSetValue(A,II,II+n,1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (j>0) { ierr = MatSetValue(A,II,II-1,1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (j<n-1) { ierr = MatSetValue(A,II,II+1,1.0,INSERT_VALUES);CHKERRQ(ierr); }
    ierr = MatSetValue(A,II,II,-4.0,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&t,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create two BV objects to store the solution and right-hand side
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = BVCreate(PETSC_COMM_WORLD,&X1);CHKERRQ(ierr);
  ierr = BVSetSizesFromVec(X1,t,rank);CHKERRQ(ierr);
  ierr = BVSetFromOptions(X1);CHKERRQ(ierr);

  ierr = BVCreate(PETSC_COMM_WORLD,&C1);CHKERRQ(ierr);
  ierr = BVSetSizesFromVec(C1,t,1);CHKERRQ(ierr);
  ierr = BVSetFromOptions(C1);CHKERRQ(ierr);

  /* fill the rhs factor */
  ierr = BVGetColumn(C1,0,&v);CHKERRQ(ierr);
  ierr = VecSet(v,1.0);CHKERRQ(ierr);
  ierr = BVRestoreColumn(C1,0,&v);CHKERRQ(ierr);
  /*ierr = BVGetColumn(C1,1,&v);CHKERRQ(ierr);
  ierr = VecSet(v,0.0);CHKERRQ(ierr);
  ierr = VecSetValue(v,0,-2.0,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValue(v,1,-1.0,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValue(v,2,-1.0,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(v);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(v);CHKERRQ(ierr);
  ierr = BVRestoreColumn(C1,1,&v);CHKERRQ(ierr);*/

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create the matrix equation solver context
  */
  ierr = LMECreate(PETSC_COMM_WORLD,&lme);CHKERRQ(ierr);

  /*
     Set the type of equation
  */
  ierr = LMESetProblemType(lme,LME_LYAPUNOV);CHKERRQ(ierr);

  /*
     Set the matrix coefficients, the right-hand side, and the solution.
     In this case, it is a Lyapunov equation A*X+X*A'=-C where both
     C and X are symmetric, so they are given as C=C1*C1', X=X1*X1'
  */
  ierr = LMESetCoefficients(lme,A,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = LMESetRHS(lme,C1,NULL);CHKERRQ(ierr);
  ierr = LMESetSolution(lme,X1,NULL);CHKERRQ(ierr);

  /*
     (Optoinal) Set other solver options
  */
  ierr = LMESetTolerances(lme,1e-07,PETSC_DEFAULT);CHKERRQ(ierr);

  /*
     Set solver parameters at runtime
  */
  ierr = LMESetFromOptions(lme);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                   Solve the matrix equation, A*X+X*A'=-C
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = LMESolve(lme);CHKERRQ(ierr);
  ierr = LMEGetConvergedReason(lme,&reason);CHKERRQ(ierr);
  if (reason<0) SETERRQ(PETSC_COMM_WORLD,1,"Solver did not converge");

  /*
     Optional: Get some information from the solver and display it
  */
  ierr = LMEGetIterationNumber(lme,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);CHKERRQ(ierr);
  ierr = LMEGetDimensions(lme,&ncv);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Subspace dimension: %D\n",ncv);CHKERRQ(ierr);
  ierr = LMEGetTolerances(lme,&tol,&maxit);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Compute residual error
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = LMEGetErrorEstimate(lme,&errest);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Error estimate reported by the solver: %g\n",(double)errest);CHKERRQ(ierr);
  ierr = LMEComputeError(lme,&error);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Computed residual norm: %g\n\n",(double)error);CHKERRQ(ierr);

  /*
     Free work space
  */
  ierr = LMEDestroy(&lme);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = BVDestroy(&C1);CHKERRQ(ierr);
  ierr = BVDestroy(&X1);CHKERRQ(ierr);
  ierr = VecDestroy(&t);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}

