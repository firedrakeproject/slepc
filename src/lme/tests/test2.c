/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Dense matrix equations, n=%" PetscInt_FMT ".\n",n));

  /* Create LME object */
  PetscCall(LMECreate(PETSC_COMM_WORLD,&lme));

  /* Set up viewer */
  PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  if (verbose) PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));

  /* Create matrices */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&A));
  PetscCall(PetscObjectSetName((PetscObject)A,"A"));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&B));
  PetscCall(PetscObjectSetName((PetscObject)B,"B"));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,n,k,NULL,&C));
  PetscCall(PetscObjectSetName((PetscObject)C,"C"));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&X));
  PetscCall(PetscObjectSetName((PetscObject)X,"X"));

  /* Fill A with an upper Hessenberg Toeplitz matrix */
  PetscCall(MatDenseGetArray(A,&As));
  for (i=0;i<n;i++) As[i+i*n]=3.0-(PetscReal)n/2;
  for (i=0;i<n-1;i++) As[i+1+i*n]=0.5;
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) As[i+(i+j)*n]=1.0;
  }
  PetscCall(MatDenseRestoreArray(A,&As));

  if (verbose) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Matrix A - - - - - - - -\n"));
    PetscCall(MatView(A,viewer));
  }

  /* Fill B with the 1-D Laplacian matrix */
  PetscCall(MatDenseGetArray(B,&Bs));
  for (i=0;i<n;i++) Bs[i+i*n]=2.0;
  for (i=0;i<n-1;i++) { Bs[i+1+i*n]=-1; Bs[i+(i+1)*n]=-1; }
  PetscCall(MatDenseRestoreArray(B,&Bs));

  if (verbose) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Matrix B - - - - - - - -\n"));
    PetscCall(MatView(B,viewer));
  }

  /* Solve Lyapunov equation A*X+X*A'= -B */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Solving Lyapunov equation for B\n"));
  PetscCall(MatDenseGetArray(A,&As));
  PetscCall(MatDenseGetArray(B,&Bs));
  PetscCall(MatDenseGetArray(X,&Xs));
  PetscCall(LMEDenseLyapunov(lme,n,As,n,Bs,n,Xs,n));
  PetscCall(MatDenseRestoreArray(A,&As));
  PetscCall(MatDenseRestoreArray(B,&Bs));
  PetscCall(MatDenseRestoreArray(X,&Xs));
  if (verbose) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Solution X - - - - - - - -\n"));
    PetscCall(MatView(X,viewer));
  }

  /* Fill C with a full-rank nx2 matrix */
  PetscCall(MatDenseGetArray(C,&Cs));
  for (i=0;i<k;i++) Cs[i+i*n] = (i%2)? -1.0: 1.0;
  PetscCall(MatDenseRestoreArray(C,&Cs));

  if (verbose) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Matrix C - - - - - - - -\n"));
    PetscCall(MatView(C,viewer));
  }

  /* Solve Lyapunov equation A*X+X*A'= -C*C' */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Solving Lyapunov equation for C (Cholesky)\n"));
  PetscCall(MatDenseGetArray(A,&As));
  PetscCall(MatDenseGetArray(C,&Cs));
  PetscCall(MatDenseGetArray(X,&Xs));
  PetscCall(LMEDenseHessLyapunovChol(lme,n,As,n,2,Cs,n,Xs,n,NULL));
  PetscCall(MatDenseRestoreArray(A,&As));
  PetscCall(MatDenseRestoreArray(C,&Cs));
  PetscCall(MatDenseRestoreArray(X,&Xs));
  if (verbose) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Solution X - - - - - - - -\n"));
    PetscCall(MatView(X,viewer));
  }

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&X));
  PetscCall(LMEDestroy(&lme));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      args: -info :lme
      requires: double
      filter: sed -e "s/[0-9]\.[0-9]*e[+-]\([0-9]*\)/1e-\\1/g"

TEST*/
