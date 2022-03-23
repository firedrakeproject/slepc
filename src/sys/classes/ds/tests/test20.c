/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test DSGNHEP with upper quasi-triangular matrix pair.\n\n";

#include <slepcds.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DS             ds;
  PetscScalar    *A,*B,*X;
  PetscReal      rnorm,aux;
  PetscInt       i,j,n=10,ld;
  PetscViewer    viewer;
  PetscBool      verbose;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Solve a Dense System of type GNHEP - dimension %" PetscInt_FMT ".\n",n));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));

  /* Create DS object */
  CHKERRQ(DSCreate(PETSC_COMM_WORLD,&ds));
  CHKERRQ(DSSetType(ds,DSGNHEP));
  CHKERRQ(DSSetFromOptions(ds));
  ld = n+2;  /* test leading dimension larger than n */
  CHKERRQ(DSAllocate(ds,ld));
  CHKERRQ(DSSetDimensions(ds,n,0,0));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
  CHKERRQ(DSView(ds,viewer));
  CHKERRQ(PetscViewerPopFormat(viewer));
  if (verbose) CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));

  /* Fill A,B with upper quasi-triangular matrices */
  CHKERRQ(DSGetArray(ds,DS_MAT_A,&A));
  CHKERRQ(DSGetArray(ds,DS_MAT_B,&B));
  CHKERRQ(PetscArrayzero(A,ld*n));
  for (i=0;i<n;i++) A[i+i*ld]=2.0;
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) A[i+(i+j)*ld]=0.001;
  }
  CHKERRQ(PetscArrayzero(B,ld*n));
  for (i=0;i<n;i++) B[i+i*ld]=1.0;
  B[1+0*ld]=B[0+1*ld]=PETSC_MACHINE_EPSILON;
  for (i=1;i<n;i+=3) {
    A[i+(i-1)*ld]=-A[(i-1)+i*ld];
  }
  CHKERRQ(DSRestoreArray(ds,DS_MAT_A,&A));
  CHKERRQ(DSRestoreArray(ds,DS_MAT_B,&B));
  CHKERRQ(DSSetState(ds,DS_STATE_INTERMEDIATE));

  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }

  /* Eigenvectors */
  j = 0;
  CHKERRQ(DSVectors(ds,DS_MAT_X,&j,&rnorm));  /* first eigenvector */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Value of rnorm for 2nd vector = %.3f\n",(double)rnorm));
  CHKERRQ(DSVectors(ds,DS_MAT_X,NULL,NULL));  /* all eigenvectors */
  j = 0;
  rnorm = 0.0;
  CHKERRQ(DSGetArray(ds,DS_MAT_X,&X));
  for (i=0;i<n;i++) {
#if defined(PETSC_USE_COMPLEX)
    aux = PetscAbsScalar(X[i+j*ld]);
#else
    aux = SlepcAbsEigenvalue(X[i+j*ld],X[i+(j+1)*ld]);
#endif
    rnorm += aux*aux;
  }
  CHKERRQ(DSRestoreArray(ds,DS_MAT_X,&X));
  rnorm = PetscSqrtReal(rnorm);
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of 1st columns = %.3f\n",(double)rnorm));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After vectors - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }

  CHKERRQ(DSDestroy(&ds));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1

TEST*/
