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
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  k = PetscMin(n,m);
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Solve a Dense System of type SVD - dimension %" PetscInt_FMT "x%" PetscInt_FMT ".\n",n,m));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-extrarow",&extrarow));

  /* Create DS object */
  CHKERRQ(DSCreate(PETSC_COMM_WORLD,&ds));
  CHKERRQ(DSSetType(ds,DSSVD));
  CHKERRQ(DSSetFromOptions(ds));
  ld = n+2;  /* test leading dimension larger than n */
  CHKERRQ(DSAllocate(ds,ld));
  CHKERRQ(DSSetDimensions(ds,n,0,0));
  CHKERRQ(DSSVDSetDimensions(ds,m));
  CHKERRQ(DSSVDGetDimensions(ds,&m1));
  PetscCheck(m1==m,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Inconsistent dimension value");
  CHKERRQ(DSSetExtraRow(ds,extrarow));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
  CHKERRQ(DSView(ds,viewer));
  CHKERRQ(PetscViewerPopFormat(viewer));

  /* Fill with a rectangular Toeplitz matrix */
  CHKERRQ(DSGetArray(ds,DS_MAT_A,&A));
  for (i=0;i<k;i++) A[i+i*ld]=1.0;
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) { if ((i+j)<m) A[i+(i+j)*ld]=(PetscScalar)(j+1); }
  }
  for (j=1;j<n/2;j++) {
    for (i=0;i<n-j;i++) { if ((i+j)<n && i<m) A[(i+j)+i*ld]=-1.0; }
  }
  if (extrarow) { A[n-2+m*ld]=1.0; A[n-1+m*ld]=1.0; }  /* really an extra column */
  CHKERRQ(DSRestoreArray(ds,DS_MAT_A,&A));
  CHKERRQ(DSSetState(ds,DS_STATE_RAW));
  if (verbose) {
    CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n"));
  }
  CHKERRQ(DSView(ds,viewer));

  /* Solve */
  CHKERRQ(PetscMalloc1(k,&w));
  CHKERRQ(DSGetSlepcSC(ds,&sc));
  sc->comparison    = SlepcCompareLargestReal;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  CHKERRQ(DSSolve(ds,w,NULL));
  CHKERRQ(DSSort(ds,w,NULL,NULL,NULL,NULL));
  if (extrarow) CHKERRQ(DSUpdateExtraRow(ds));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }

  /* Print singular values */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Computed singular values =\n"));
  for (i=0;i<k;i++) {
    sigma = PetscRealPart(w[i]);
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %.5f\n",(double)sigma));
  }

  if (extrarow) {
    /* Check that extra column is correct */
    CHKERRQ(DSGetArray(ds,DS_MAT_A,&A));
    CHKERRQ(DSGetArray(ds,DS_MAT_U,&U));
    d = 0.0;
    for (i=0;i<n;i++) d += A[i+m*ld]-U[n-2+i*ld]-U[n-1+i*ld];
    if (PetscAbsScalar(d)>10*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: there is a mismatch in the extra row of %g\n",(double)PetscAbsScalar(d)));
    CHKERRQ(DSRestoreArray(ds,DS_MAT_A,&A));
    CHKERRQ(DSRestoreArray(ds,DS_MAT_U,&U));
  }

  /* Singular vectors */
  CHKERRQ(DSVectors(ds,DS_MAT_U,NULL,NULL));  /* all singular vectors */
  j = 0;
  rnorm = 0.0;
  CHKERRQ(DSGetArray(ds,DS_MAT_U,&U));
  for (i=0;i<n;i++) {
    aux = PetscAbsScalar(U[i+j*ld]);
    rnorm += aux*aux;
  }
  CHKERRQ(DSRestoreArray(ds,DS_MAT_U,&U));
  rnorm = PetscSqrtReal(rnorm);
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of 1st U vector = %.3f\n",(double)rnorm));

  CHKERRQ(PetscFree(w));
  CHKERRQ(DSDestroy(&ds));
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
