/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test DSHEP.\n\n";

#include <slepcds.h>

int main(int argc,char **argv)
{
  DS             ds;
  SlepcSC        sc;
  PetscScalar    *A,*X,*Q,*eig,d;
  PetscReal      rnorm,aux;
  PetscInt       i,j,n=10,ld;
  PetscViewer    viewer;
  PetscBool      verbose,extrarow;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Solve a Dense System of type HEP - dimension %" PetscInt_FMT ".\n",n));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-extrarow",&extrarow));

  /* Create DS object */
  PetscCall(DSCreate(PETSC_COMM_WORLD,&ds));
  PetscCall(DSSetType(ds,DSHEP));
  PetscCall(DSSetFromOptions(ds));
  ld = n+2;  /* test leading dimension larger than n */
  PetscCall(DSAllocate(ds,ld));
  PetscCall(DSSetDimensions(ds,n,0,0));
  PetscCall(DSSetExtraRow(ds,extrarow));

  /* Set up viewer */
  PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
  PetscCall(DSView(ds,viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  if (verbose) PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));

  /* Fill with a symmetric Toeplitz matrix */
  PetscCall(DSGetArray(ds,DS_MAT_A,&A));
  for (i=0;i<n;i++) A[i+i*ld]=2.0;
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) { A[i+(i+j)*ld]=1.0; A[(i+j)+i*ld]=1.0; }
  }
  if (extrarow) { A[n+(n-2)*ld]=1.0; A[n+(n-1)*ld]=1.0; }
  PetscCall(DSRestoreArray(ds,DS_MAT_A,&A));
  PetscCall(DSSetState(ds,DS_STATE_RAW));
  if (verbose) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n"));
    PetscCall(DSView(ds,viewer));
  }

  /* Solve */
  PetscCall(PetscMalloc1(n,&eig));
  PetscCall(DSGetSlepcSC(ds,&sc));
  sc->comparison    = SlepcCompareLargestMagnitude;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  PetscCall(DSSolve(ds,eig,NULL));
  PetscCall(DSSort(ds,eig,NULL,NULL,NULL,NULL));
  if (extrarow) PetscCall(DSUpdateExtraRow(ds));
  if (verbose) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n"));
    PetscCall(DSView(ds,viewer));
  }

  /* Print eigenvalues */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Computed eigenvalues =\n"));
  for (i=0;i<n;i++) PetscCall(PetscViewerASCIIPrintf(viewer,"  %.5f\n",(double)PetscRealPart(eig[i])));

  if (extrarow) {
    /* Check that extra row is correct */
    PetscCall(DSGetArray(ds,DS_MAT_A,&A));
    PetscCall(DSGetArray(ds,DS_MAT_Q,&Q));
    d = 0.0;
    for (i=0;i<n;i++) d += A[n+i*ld]-Q[n-2+i*ld]-Q[n-1+i*ld];
    if (PetscAbsScalar(d)>10*PETSC_MACHINE_EPSILON) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Warning: there is a mismatch in the extra row of %g\n",(double)PetscAbsScalar(d)));
    PetscCall(DSRestoreArray(ds,DS_MAT_A,&A));
    PetscCall(DSRestoreArray(ds,DS_MAT_Q,&Q));
  }

  /* Eigenvectors */
  j = 2;
  PetscCall(DSVectors(ds,DS_MAT_X,&j,&rnorm));  /* third eigenvector */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Value of rnorm for 3rd vector = %.3f\n",(double)rnorm));
  PetscCall(DSVectors(ds,DS_MAT_X,NULL,NULL));  /* all eigenvectors */
  j = 0;
  rnorm = 0.0;
  PetscCall(DSGetArray(ds,DS_MAT_X,&X));
  for (i=0;i<n;i++) {
    aux = PetscAbsScalar(X[i+j*ld]);
    rnorm += aux*aux;
  }
  PetscCall(DSRestoreArray(ds,DS_MAT_X,&X));
  rnorm = PetscSqrtReal(rnorm);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm of 1st vector = %.3f\n",(double)rnorm));
  if (verbose) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"After vectors - - - - - - - - -\n"));
    PetscCall(DSView(ds,viewer));
  }

  PetscCall(PetscFree(eig));
  PetscCall(DSDestroy(&ds));
  PetscCall(SlepcFinalize());
  return 0;
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
