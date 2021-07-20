/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test DSSVD.\n\n";

#include <slepcds.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DS             ds;
  SlepcSC        sc;
  PetscReal      sigma,rnorm,aux;
  PetscScalar    *A,*U,*w,d;
  PetscInt       i,j,k,n=15,m=10,m1,ld;
  PetscViewer    viewer;
  PetscBool      verbose,extrarow;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  k = PetscMin(n,m);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Solve a Dense System of type SVD - dimension %Dx%D.\n",n,m);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-verbose",&verbose);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-extrarow",&extrarow);CHKERRQ(ierr);

  /* Create DS object */
  ierr = DSCreate(PETSC_COMM_WORLD,&ds);CHKERRQ(ierr);
  ierr = DSSetType(ds,DSSVD);CHKERRQ(ierr);
  ierr = DSSetFromOptions(ds);CHKERRQ(ierr);
  ld = n+2;  /* test leading dimension larger than n */
  ierr = DSAllocate(ds,ld);CHKERRQ(ierr);
  ierr = DSSetDimensions(ds,n,0,0);CHKERRQ(ierr);
  ierr = DSSVDSetDimensions(ds,m);CHKERRQ(ierr);
  ierr = DSSVDGetDimensions(ds,&m1);CHKERRQ(ierr);
  if (m1!=m) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Inconsistent dimension value");
  ierr = DSSetExtraRow(ds,extrarow);CHKERRQ(ierr);

  /* Set up viewer */
  ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
  ierr = DSView(ds,viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);

  /* Fill with a rectangular Toeplitz matrix */
  ierr = DSGetArray(ds,DS_MAT_A,&A);CHKERRQ(ierr);
  for (i=0;i<k;i++) A[i+i*ld]=1.0;
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) { if ((i+j)<m) A[i+(i+j)*ld]=(PetscScalar)(j+1); }
  }
  for (j=1;j<n/2;j++) {
    for (i=0;i<n-j;i++) { if ((i+j)<n && i<m) A[(i+j)+i*ld]=-1.0; }
  }
  if (extrarow) { A[n-2+m*ld]=1.0; A[n-1+m*ld]=1.0; }  /* really an extra column */
  ierr = DSRestoreArray(ds,DS_MAT_A,&A);CHKERRQ(ierr);
  ierr = DSSetState(ds,DS_STATE_RAW);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n");CHKERRQ(ierr);
  }
  ierr = DSView(ds,viewer);CHKERRQ(ierr);

  /* Solve */
  ierr = PetscMalloc1(k,&w);CHKERRQ(ierr);
  ierr = DSGetSlepcSC(ds,&sc);CHKERRQ(ierr);
  sc->comparison    = SlepcCompareLargestReal;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  ierr = DSSolve(ds,w,NULL);CHKERRQ(ierr);
  ierr = DSSort(ds,w,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  if (extrarow) { ierr = DSUpdateExtraRow(ds);CHKERRQ(ierr); }
  if (verbose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n");CHKERRQ(ierr);
    ierr = DSView(ds,viewer);CHKERRQ(ierr);
  }

  /* Print singular values */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Computed singular values =\n");CHKERRQ(ierr);
  for (i=0;i<k;i++) {
    sigma = PetscRealPart(w[i]);
    ierr = PetscViewerASCIIPrintf(viewer,"  %.5f\n",(double)sigma);CHKERRQ(ierr);
  }

  if (extrarow) {
    /* Check that extra column is correct */
    ierr = DSGetArray(ds,DS_MAT_A,&A);CHKERRQ(ierr);
    ierr = DSGetArray(ds,DS_MAT_U,&U);CHKERRQ(ierr);
    d = 0.0;
    for (i=0;i<n;i++) d += A[i+m*ld]-U[n-2+i*ld]-U[n-1+i*ld];
    if (PetscAbsScalar(d)>10*PETSC_MACHINE_EPSILON) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: there is a mismatch in the extra row of %g\n",(double)PetscAbsScalar(d));CHKERRQ(ierr);
    }
    ierr = DSRestoreArray(ds,DS_MAT_A,&A);CHKERRQ(ierr);
    ierr = DSRestoreArray(ds,DS_MAT_U,&U);CHKERRQ(ierr);
  }

  /* Singular vectors */
  ierr = DSVectors(ds,DS_MAT_U,NULL,NULL);CHKERRQ(ierr);  /* all singular vectors */
  j = 0;
  rnorm = 0.0;
  ierr = DSGetArray(ds,DS_MAT_U,&U);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    aux = PetscAbsScalar(U[i+j*ld]);
    rnorm += aux*aux;
  }
  ierr = DSRestoreArray(ds,DS_MAT_U,&U);CHKERRQ(ierr);
  rnorm = PetscSqrtReal(rnorm);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of 1st U vector = %.3f\n",(double)rnorm);CHKERRQ(ierr);

  ierr = PetscFree(w);CHKERRQ(ierr);
  ierr = DSDestroy(&ds);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      requires: !single

   test:
      suffix: 2
      args: -extrarow
      requires: !single

TEST*/
