/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test DSHEP.\n\n";

#include <slepcds.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DS             ds;
  SlepcSC        sc;
  PetscScalar    *A,*X,*Q,*eig,d;
  PetscReal      rnorm,aux;
  PetscInt       i,j,n=10,ld;
  PetscViewer    viewer;
  PetscBool      verbose,extrarow;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Solve a Dense System of type HEP - dimension %" PetscInt_FMT ".\n",n));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-extrarow",&extrarow));

  /* Create DS object */
  CHKERRQ(DSCreate(PETSC_COMM_WORLD,&ds));
  CHKERRQ(DSSetType(ds,DSHEP));
  CHKERRQ(DSSetFromOptions(ds));
  ld = n+2;  /* test leading dimension larger than n */
  CHKERRQ(DSAllocate(ds,ld));
  CHKERRQ(DSSetDimensions(ds,n,0,0));
  CHKERRQ(DSSetExtraRow(ds,extrarow));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
  CHKERRQ(DSView(ds,viewer));
  CHKERRQ(PetscViewerPopFormat(viewer));
  if (verbose) CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));

  /* Fill with a symmetric Toeplitz matrix */
  CHKERRQ(DSGetArray(ds,DS_MAT_A,&A));
  for (i=0;i<n;i++) A[i+i*ld]=2.0;
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) { A[i+(i+j)*ld]=1.0; A[(i+j)+i*ld]=1.0; }
  }
  if (extrarow) { A[n+(n-2)*ld]=1.0; A[n+(n-1)*ld]=1.0; }
  CHKERRQ(DSRestoreArray(ds,DS_MAT_A,&A));
  CHKERRQ(DSSetState(ds,DS_STATE_RAW));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }

  /* Solve */
  CHKERRQ(PetscMalloc1(n,&eig));
  CHKERRQ(DSGetSlepcSC(ds,&sc));
  sc->comparison    = SlepcCompareLargestMagnitude;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  CHKERRQ(DSSolve(ds,eig,NULL));
  CHKERRQ(DSSort(ds,eig,NULL,NULL,NULL,NULL));
  if (extrarow) CHKERRQ(DSUpdateExtraRow(ds));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }

  /* Print eigenvalues */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Computed eigenvalues =\n"));
  for (i=0;i<n;i++) CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %.5f\n",(double)PetscRealPart(eig[i])));

  if (extrarow) {
    /* Check that extra row is correct */
    CHKERRQ(DSGetArray(ds,DS_MAT_A,&A));
    CHKERRQ(DSGetArray(ds,DS_MAT_Q,&Q));
    d = 0.0;
    for (i=0;i<n;i++) d += A[n+i*ld]-Q[n-2+i*ld]-Q[n-1+i*ld];
    if (PetscAbsScalar(d)>10*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: there is a mismatch in the extra row of %g\n",(double)PetscAbsScalar(d)));
    CHKERRQ(DSRestoreArray(ds,DS_MAT_A,&A));
    CHKERRQ(DSRestoreArray(ds,DS_MAT_Q,&Q));
  }

  /* Eigenvectors */
  j = 2;
  CHKERRQ(DSVectors(ds,DS_MAT_X,&j,&rnorm));  /* third eigenvector */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Value of rnorm for 3rd vector = %.3f\n",(double)rnorm));
  CHKERRQ(DSVectors(ds,DS_MAT_X,NULL,NULL));  /* all eigenvectors */
  j = 0;
  rnorm = 0.0;
  CHKERRQ(DSGetArray(ds,DS_MAT_X,&X));
  for (i=0;i<n;i++) {
    aux = PetscAbsScalar(X[i+j*ld]);
    rnorm += aux*aux;
  }
  CHKERRQ(DSRestoreArray(ds,DS_MAT_X,&X));
  rnorm = PetscSqrtReal(rnorm);
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of 1st vector = %.3f\n",(double)rnorm));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After vectors - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }

  CHKERRQ(PetscFree(eig));
  CHKERRQ(DSDestroy(&ds));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      args: -n 12 -ds_method {{0 1 2}}
      filter: grep -v "solving the problem" | sed -e "s/extrarow//"
      output_file: output/test2_1.out
      requires: !single
      test:
         suffix: 1
      test:
         suffix: 2
         args: -extrarow

TEST*/
