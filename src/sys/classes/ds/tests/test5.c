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
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Solve a Dense System of type GHIEP - dimension %D.\n",n);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-verbose",&verbose);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-extrarow",&extrarow);CHKERRQ(ierr);

  /* Create DS object */
  ierr = DSCreate(PETSC_COMM_WORLD,&ds);CHKERRQ(ierr);
  ierr = DSSetType(ds,DSGHIEP);CHKERRQ(ierr);
  ierr = DSSetFromOptions(ds);CHKERRQ(ierr);
  ld = n+2;  /* test leading dimension larger than n */
  ierr = DSAllocate(ds,ld);CHKERRQ(ierr);
  ierr = DSSetDimensions(ds,n,0,0);CHKERRQ(ierr);
  ierr = DSSetExtraRow(ds,extrarow);CHKERRQ(ierr);

  /* Set up viewer */
  ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
  ierr = DSView(ds,viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  }

  /* Fill with a symmetric Toeplitz matrix */
  ierr = DSGetArray(ds,DS_MAT_A,&A);CHKERRQ(ierr);
  ierr = DSGetArray(ds,DS_MAT_B,&B);CHKERRQ(ierr);
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
  ierr = DSRestoreArray(ds,DS_MAT_A,&A);CHKERRQ(ierr);
  ierr = DSRestoreArray(ds,DS_MAT_B,&B);CHKERRQ(ierr);
  ierr = DSSetState(ds,DS_STATE_RAW);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n");CHKERRQ(ierr);
    ierr = DSView(ds,viewer);CHKERRQ(ierr);
  }

  /* Solve */
  ierr = PetscCalloc2(n,&eigr,n,&eigi);CHKERRQ(ierr);
  ierr = DSGetSlepcSC(ds,&sc);CHKERRQ(ierr);
  sc->comparison    = SlepcCompareLargestMagnitude;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  ierr = DSSolve(ds,eigr,eigi);CHKERRQ(ierr);
  ierr = DSSort(ds,eigr,eigi,NULL,NULL,NULL);CHKERRQ(ierr);
  if (extrarow) { ierr = DSUpdateExtraRow(ds);CHKERRQ(ierr); }

  if (verbose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n");CHKERRQ(ierr);
    ierr = DSView(ds,viewer);CHKERRQ(ierr);
  }

  /* Print eigenvalues */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Computed eigenvalues =\n");CHKERRQ(ierr);
  for (i=0;i<n;i++) {
#if defined(PETSC_USE_COMPLEX)
    re = PetscRealPart(eigr[i]);
    im = PetscImaginaryPart(eigr[i]);
#else
    re = eigr[i];
    im = eigi[i];
#endif
    if (PetscAbs(im)<1e-10) {
      ierr = PetscViewerASCIIPrintf(viewer,"  %.5f\n",(double)re);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  %.5f%+.5fi\n",(double)re,(double)im);CHKERRQ(ierr);
    }
  }

  if (extrarow) {
    /* Check that extra row is correct */
    ierr = DSGetArray(ds,DS_MAT_A,&A);CHKERRQ(ierr);
    ierr = DSGetArray(ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
    d = 0.0;
    for (i=0;i<n;i++) d += A[n+i*ld]+Q[n-1+i*ld];
    if (PetscAbsScalar(d)>10*PETSC_MACHINE_EPSILON) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: there is a mismatch in the extra row of %g\n",(double)PetscAbsScalar(d));CHKERRQ(ierr);
    }
    ierr = DSRestoreArray(ds,DS_MAT_A,&A);CHKERRQ(ierr);
    ierr = DSRestoreArray(ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
  }
  ierr = PetscFree2(eigr,eigi);CHKERRQ(ierr);
  ierr = DSDestroy(&ds);CHKERRQ(ierr);
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
