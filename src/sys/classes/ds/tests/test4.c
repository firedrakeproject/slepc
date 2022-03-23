/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test DSGNHEP.\n\n";

#include <slepcds.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DS             ds;
  SlepcSC        sc;
  PetscScalar    *A,*B,*X,*wr,*wi;
  PetscReal      re,im,rnorm,aux;
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

  /* Fill A with Grcar matrix */
  CHKERRQ(DSGetArray(ds,DS_MAT_A,&A));
  CHKERRQ(PetscArrayzero(A,ld*n));
  for (i=1;i<n;i++) A[i+(i-1)*ld]=-1.0;
  for (j=0;j<4;j++) {
    for (i=0;i<n-j;i++) A[i+(i+j)*ld]=1.0;
  }
  CHKERRQ(DSRestoreArray(ds,DS_MAT_A,&A));
  /* Fill B with an upper triangular matrix */
  CHKERRQ(DSGetArray(ds,DS_MAT_B,&B));
  CHKERRQ(PetscArrayzero(B,ld*n));
  B[0+0*ld]=-1.0;
  B[0+1*ld]=2.0;
  for (i=1;i<n;i++) B[i+i*ld]=1.0;
  CHKERRQ(DSRestoreArray(ds,DS_MAT_B,&B));

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
    if (PetscAbs(im)<1e-10) CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %.5f\n",(double)re));
    else CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %.5f%+.5fi\n",(double)re,(double)im));
  }

  /* Eigenvectors */
  j = 1;
  CHKERRQ(DSVectors(ds,DS_MAT_X,&j,&rnorm));  /* second eigenvector */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Value of rnorm for 2nd vector = %.3f\n",(double)rnorm));
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

   test:
      suffix: 1
      filter: sed -e "s/[+-]\([0-9]\.[0-9]*i\)/+-\\1/" | sed -e "s/+-0\.0*i//"

TEST*/
