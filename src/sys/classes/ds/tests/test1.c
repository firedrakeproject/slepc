/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test DSNHEP.\n\n";

#include <slepcds.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DS             ds;
  SlepcSC        sc;
  DSType         type;
  DSStateType    state;
  PetscScalar    *A,*X,*Q,*wr,*wi,d;
  PetscReal      re,im,rnorm,aux;
  PetscInt       i,j,n=10,ld,method;
  PetscViewer    viewer;
  PetscBool      verbose,extrarow;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Solve a Dense System of type NHEP - dimension %" PetscInt_FMT ".\n",n));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-extrarow",&extrarow));

  /* Create DS object */
  CHKERRQ(DSCreate(PETSC_COMM_WORLD,&ds));
  CHKERRQ(DSSetType(ds,DSNHEP));
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
  if (verbose) {
    CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));
  }

  /* Fill with Grcar matrix */
  CHKERRQ(DSGetArray(ds,DS_MAT_A,&A));
  for (i=1;i<n;i++) A[i+(i-1)*ld]=-1.0;
  for (j=0;j<4;j++) {
    for (i=0;i<n-j;i++) A[i+(i+j)*ld]=1.0;
  }
  if (extrarow) A[n+(n-1)*ld]=-1.0;
  CHKERRQ(DSRestoreArray(ds,DS_MAT_A,&A));
  CHKERRQ(DSSetState(ds,DS_STATE_INTERMEDIATE));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }

  /* Solve */
  CHKERRQ(PetscMalloc2(n,&wr,n,&wi));
  CHKERRQ(DSGetSlepcSC(ds,&sc));
  sc->comparison    = SlepcCompareLargestMagnitude;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  CHKERRQ(DSSolve(ds,wr,wi));
  CHKERRQ(DSSort(ds,wr,wi,NULL,NULL,NULL));
  if (extrarow) CHKERRQ(DSUpdateExtraRow(ds));

  CHKERRQ(DSGetType(ds,&type));
  CHKERRQ(DSGetMethod(ds,&method));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"DS of type %s, method used=%" PetscInt_FMT "\n",type,method));
  CHKERRQ(DSGetState(ds,&state));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"State after solve: %s\n",DSStateTypes[state]));

  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }

  /* Print eigenvalues */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Computed eigenvalues =\n"));
  for (i=0;i<n;i++) {
#if defined(PETSC_USE_COMPLEX)
    re = PetscRealPart(wr[i]);
    im = PetscImaginaryPart(wr[i]);
#else
    re = wr[i];
    im = wi[i];
#endif
    if (PetscAbs(im)<1e-10) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %.5f\n",(double)re));
    } else {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %.5f%+.5fi\n",(double)re,(double)im));
    }
  }

  if (extrarow) {
    /* Check that extra row is correct */
    CHKERRQ(DSGetArray(ds,DS_MAT_A,&A));
    CHKERRQ(DSGetArray(ds,DS_MAT_Q,&Q));
    d = 0.0;
    for (i=0;i<n;i++) d += A[n+i*ld]+Q[n-1+i*ld];
    if (PetscAbsScalar(d)>10*PETSC_MACHINE_EPSILON) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: there is a mismatch in the extra row of %g\n",(double)PetscAbsScalar(d)));
    }
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
#if defined(PETSC_USE_COMPLEX)
    aux = PetscAbsScalar(X[i+j*ld]);
#else
    if (PetscAbs(wi[j])==0.0) aux = PetscAbsScalar(X[i+j*ld]);
    else aux = SlepcAbsEigenvalue(X[i+j*ld],X[i+(j+1)*ld]);
#endif
    rnorm += aux*aux;
  }
  CHKERRQ(DSRestoreArray(ds,DS_MAT_X,&X));
  rnorm = PetscSqrtReal(rnorm);
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of 1st vector = %.3f\n",(double)rnorm));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After vectors - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }

  CHKERRQ(PetscFree2(wr,wi));
  CHKERRQ(DSDestroy(&ds));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      filter: sed -e "s/[+-]\([0-9]\.[0-9]*i\)/+-\\1/" | sed -e "s/extrarow//"
      output_file: output/test1_1.out
      requires: !single
      test:
         suffix: 1
      test:
         suffix: 2
         args: -extrarow

TEST*/
