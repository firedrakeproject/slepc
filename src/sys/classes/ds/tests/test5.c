/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test DSGHIEP.\n\n";

#include <slepcds.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DS             ds;
  SlepcSC        sc;
  PetscReal      re,im;
  PetscScalar    *A,*B,*Q,*eigr,*eigi,d;
  PetscInt       i,j,n=10,ld;
  PetscViewer    viewer;
  PetscBool      verbose,extrarow;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Solve a Dense System of type GHIEP - dimension %" PetscInt_FMT ".\n",n));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-extrarow",&extrarow));

  /* Create DS object */
  CHKERRQ(DSCreate(PETSC_COMM_WORLD,&ds));
  CHKERRQ(DSSetType(ds,DSGHIEP));
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
  CHKERRQ(DSGetArray(ds,DS_MAT_B,&B));
  for (i=0;i<n;i++) A[i+i*ld]=2.0;
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) { A[i+(i+j)*ld]=1.0; A[(i+j)+i*ld]=1.0; }
  }
  for (j=1;j<3;j++) { A[0+j*ld]=-1.0*(j+2); A[j+0*ld]=-1.0*(j+2); }
  if (extrarow) A[n+(n-1)*ld]=-1.0;
  /* Signature matrix */
  for (i=0;i<n;i++) B[i+i*ld]=1.0;
  B[0] = -1.0;
  B[n-1+(n-1)*ld] = -1.0;
  CHKERRQ(DSRestoreArray(ds,DS_MAT_A,&A));
  CHKERRQ(DSRestoreArray(ds,DS_MAT_B,&B));
  CHKERRQ(DSSetState(ds,DS_STATE_RAW));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }

  /* Solve */
  CHKERRQ(PetscCalloc2(n,&eigr,n,&eigi));
  CHKERRQ(DSGetSlepcSC(ds,&sc));
  sc->comparison    = SlepcCompareLargestMagnitude;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  CHKERRQ(DSSolve(ds,eigr,eigi));
  CHKERRQ(DSSort(ds,eigr,eigi,NULL,NULL,NULL));
  if (extrarow) CHKERRQ(DSUpdateExtraRow(ds));

  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }

  /* Print eigenvalues */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Computed eigenvalues =\n"));
  for (i=0;i<n;i++) {
#if defined(PETSC_USE_COMPLEX)
    re = PetscRealPart(eigr[i]);
    im = PetscImaginaryPart(eigr[i]);
#else
    re = eigr[i];
    im = eigi[i];
#endif
    if (PetscAbs(im)<1e-10) CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %.5f\n",(double)re));
    else CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %.5f%+.5fi\n",(double)re,(double)im));
  }

  if (extrarow) {
    /* Check that extra row is correct */
    CHKERRQ(DSGetArray(ds,DS_MAT_A,&A));
    CHKERRQ(DSGetArray(ds,DS_MAT_Q,&Q));
    d = 0.0;
    for (i=0;i<n;i++) d += A[n+i*ld]+Q[n-1+i*ld];
    if (PetscAbsScalar(d)>10*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: there is a mismatch in the extra row of %g\n",(double)PetscAbsScalar(d)));
    CHKERRQ(DSRestoreArray(ds,DS_MAT_A,&A));
    CHKERRQ(DSRestoreArray(ds,DS_MAT_Q,&Q));
  }
  CHKERRQ(PetscFree2(eigr,eigi));
  CHKERRQ(DSDestroy(&ds));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      args: -ds_method {{0 1 2}}
      filter: grep -v "solving the problem" | sed -e "s/extrarow//"
      output_file: output/test5_1.out
      requires: !single
      test:
         suffix: 1
      test:
         suffix: 2
         args: -extrarow

TEST*/
