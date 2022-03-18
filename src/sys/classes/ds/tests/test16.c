/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test pseudo-orthogonalization.\n\n";

#include <slepcds.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DS             ds;
  PetscReal      *s,*ns;
  PetscScalar    *A;
  PetscInt       i,j,n=10;
  PetscViewer    viewer;
  PetscBool      verbose;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test pseudo-orthogonalization for GHIEP - dimension %" PetscInt_FMT ".\n",n));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));

  /* Create DS object */
  CHKERRQ(DSCreate(PETSC_COMM_WORLD,&ds));
  CHKERRQ(DSSetType(ds,DSGHIEP));
  CHKERRQ(DSSetFromOptions(ds));
  CHKERRQ(DSAllocate(ds,n));
  CHKERRQ(DSSetDimensions(ds,n,0,0));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  if (verbose) {
    CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));
  }

  /* Fill with a symmetric Toeplitz matrix */
  CHKERRQ(DSGetArray(ds,DS_MAT_A,&A));
  for (i=0;i<n;i++) A[i+i*n]=2.0;
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) { A[i+(i+j)*n]=1.0; A[(i+j)+i*n]=1.0; }
  }
  for (j=1;j<3;j++) { A[0+j*n]=-1.0*(j+2); A[j+0*n]=-1.0*(j+2); }
  CHKERRQ(DSRestoreArray(ds,DS_MAT_A,&A));
  CHKERRQ(DSSetState(ds,DS_STATE_RAW));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }

  /* Signature matrix */
  CHKERRQ(PetscMalloc2(n,&s,n,&ns));
  s[0] = -1.0;
  for (i=1;i<n-1;i++) s[i]=1.0;
  s[n-1] = -1.0;

  /* Orthogonalize and show signature */
  CHKERRQ(DSPseudoOrthogonalize(ds,DS_MAT_A,n,s,NULL,ns));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After pseudo-orthogonalize - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Resulting signature:\n"));
  for (i=0;i<n;i++) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%g\n",(double)ns[i]));
  }
  CHKERRQ(PetscFree2(s,ns));

  CHKERRQ(DSDestroy(&ds));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1

TEST*/
