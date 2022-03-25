/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test dense LME functions.\n\n";

#include <slepclme.h>

int main(int argc,char **argv)
{
  LME            lme;
  Mat            A,B,C,X;
  PetscInt       i,j,n=10,k=2;
  PetscScalar    *As,*Bs,*Cs,*Xs;
  PetscViewer    viewer;
  PetscBool      verbose;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Dense matrix equations, n=%" PetscInt_FMT ".\n",n));

  /* Create LME object */
  CHKERRQ(LMECreate(PETSC_COMM_WORLD,&lme));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  if (verbose) CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));

  /* Create matrices */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&A));
  CHKERRQ(PetscObjectSetName((PetscObject)A,"A"));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&B));
  CHKERRQ(PetscObjectSetName((PetscObject)B,"B"));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n,k,NULL,&C));
  CHKERRQ(PetscObjectSetName((PetscObject)C,"C"));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&X));
  CHKERRQ(PetscObjectSetName((PetscObject)X,"X"));

  /* Fill A with an upper Hessenberg Toeplitz matrix */
  CHKERRQ(MatDenseGetArray(A,&As));
  for (i=0;i<n;i++) As[i+i*n]=3.0-(PetscReal)n/2;
  for (i=0;i<n-1;i++) As[i+1+i*n]=0.5;
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) As[i+(i+j)*n]=1.0;
  }
  CHKERRQ(MatDenseRestoreArray(A,&As));

  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Matrix A - - - - - - - -\n"));
    CHKERRQ(MatView(A,viewer));
  }

  /* Fill B with the 1-D Laplacian matrix */
  CHKERRQ(MatDenseGetArray(B,&Bs));
  for (i=0;i<n;i++) Bs[i+i*n]=2.0;
  for (i=0;i<n-1;i++) { Bs[i+1+i*n]=-1; Bs[i+(i+1)*n]=-1; }
  CHKERRQ(MatDenseRestoreArray(B,&Bs));

  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Matrix B - - - - - - - -\n"));
    CHKERRQ(MatView(B,viewer));
  }

  /* Solve Lyapunov equation A*X+X*A'= -B */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Solving Lyapunov equation for B\n"));
  CHKERRQ(MatDenseGetArray(A,&As));
  CHKERRQ(MatDenseGetArray(B,&Bs));
  CHKERRQ(MatDenseGetArray(X,&Xs));
  CHKERRQ(LMEDenseLyapunov(lme,n,As,n,Bs,n,Xs,n));
  CHKERRQ(MatDenseRestoreArray(A,&As));
  CHKERRQ(MatDenseRestoreArray(B,&Bs));
  CHKERRQ(MatDenseRestoreArray(X,&Xs));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Solution X - - - - - - - -\n"));
    CHKERRQ(MatView(X,viewer));
  }

  /* Fill C with a full-rank nx2 matrix */
  CHKERRQ(MatDenseGetArray(C,&Cs));
  for (i=0;i<k;i++) Cs[i+i*n] = (i%2)? -1.0: 1.0;
  CHKERRQ(MatDenseRestoreArray(C,&Cs));

  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Matrix C - - - - - - - -\n"));
    CHKERRQ(MatView(C,viewer));
  }

  /* Solve Lyapunov equation A*X+X*A'= -C*C' */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Solving Lyapunov equation for C (Cholesky)\n"));
  CHKERRQ(MatDenseGetArray(A,&As));
  CHKERRQ(MatDenseGetArray(C,&Cs));
  CHKERRQ(MatDenseGetArray(X,&Xs));
  CHKERRQ(LMEDenseHessLyapunovChol(lme,n,As,n,2,Cs,n,Xs,n,NULL));
  CHKERRQ(MatDenseRestoreArray(A,&As));
  CHKERRQ(MatDenseRestoreArray(C,&Cs));
  CHKERRQ(MatDenseRestoreArray(X,&Xs));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Solution X - - - - - - - -\n"));
    CHKERRQ(MatView(X,viewer));
  }

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&X));
  CHKERRQ(LMEDestroy(&lme));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      args: -info :lme
      requires: double
      filter: sed -e "s/[0-9]\.[0-9]*e[+-]\([0-9]*\)/1e-\\1/g"

TEST*/
