/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test DSSVD with compact storage.\n\n";

#include <slepcds.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DS             ds;
  SlepcSC        sc;
  PetscReal      *T,sigma;
  PetscScalar    *U,*w,d;
  PetscInt       i,n=10,m,l=2,k=5,ld;
  PetscViewer    viewer;
  PetscBool      verbose,extrarow;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  m = n;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Solve a Dense System of type SVD with compact storage - dimension %" PetscInt_FMT "x%" PetscInt_FMT ".\n",n,m));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-l",&l,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  PetscCheck(l<=n && k<=n && l<=k,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Wrong value of dimensions");
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-extrarow",&extrarow));

  /* Create DS object */
  CHKERRQ(DSCreate(PETSC_COMM_WORLD,&ds));
  CHKERRQ(DSSetType(ds,DSSVD));
  CHKERRQ(DSSetFromOptions(ds));
  ld = n+2;  /* test leading dimension larger than n */
  CHKERRQ(DSAllocate(ds,ld));
  CHKERRQ(DSSetDimensions(ds,n,l,k));
  CHKERRQ(DSSVDSetDimensions(ds,m));
  CHKERRQ(DSSetCompact(ds,PETSC_TRUE));
  CHKERRQ(DSSetExtraRow(ds,extrarow));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
  CHKERRQ(DSView(ds,viewer));
  CHKERRQ(PetscViewerPopFormat(viewer));
  CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));

  /* Fill upper arrow-bidiagonal matrix */
  CHKERRQ(DSGetArrayReal(ds,DS_MAT_T,&T));
  for (i=0;i<n;i++) T[i] = (PetscReal)(i+1);
  for (i=l;i<n-1;i++) T[i+ld] = 1.0;
  if (extrarow) T[n-1+ld] = 1.0;
  CHKERRQ(DSRestoreArrayReal(ds,DS_MAT_T,&T));
  if (l==0 && k==0) CHKERRQ(DSSetState(ds,DS_STATE_INTERMEDIATE));
  else CHKERRQ(DSSetState(ds,DS_STATE_RAW));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n"));
  CHKERRQ(DSView(ds,viewer));

  /* Solve */
  CHKERRQ(PetscMalloc1(n,&w));
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
  for (i=0;i<n;i++) {
    sigma = PetscRealPart(w[i]);
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %.5f\n",(double)sigma));
  }

  if (extrarow) {
    /* Check that extra row is correct */
    CHKERRQ(DSGetArrayReal(ds,DS_MAT_T,&T));
    CHKERRQ(DSGetArray(ds,DS_MAT_U,&U));
    d = 0.0;
    for (i=l;i<n;i++) d += T[i+ld]-U[n-1+i*ld];
    if (PetscAbsScalar(d)>10*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: there is a mismatch in the extra row of %g\n",(double)PetscAbsScalar(d)));
    CHKERRQ(DSRestoreArrayReal(ds,DS_MAT_T,&T));
    CHKERRQ(DSRestoreArray(ds,DS_MAT_U,&U));
  }
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
      args: -l 0 -k 0
      suffix: 2
      requires: !single

   test:
      args: -extrarow
      suffix: 3
      requires: !single

TEST*/
