/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test pseudo-orthogonalization.\n\n";

#include <slepcds.h>

int main(int argc,char **argv)
{
  DS             ds;
  PetscReal      *s,*ns;
  PetscScalar    *A;
  PetscInt       i,j,n=10;
  PetscViewer    viewer;
  PetscBool      verbose;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test pseudo-orthogonalization for GHIEP - dimension %" PetscInt_FMT ".\n",n));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));

  /* Create DS object */
  PetscCall(DSCreate(PETSC_COMM_WORLD,&ds));
  PetscCall(DSSetType(ds,DSGHIEP));
  PetscCall(DSSetFromOptions(ds));
  PetscCall(DSAllocate(ds,n));
  PetscCall(DSSetDimensions(ds,n,0,0));

  /* Set up viewer */
  PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  if (verbose) PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));

  /* Fill with a symmetric Toeplitz matrix */
  PetscCall(DSGetArray(ds,DS_MAT_A,&A));
  for (i=0;i<n;i++) A[i+i*n]=2.0;
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) { A[i+(i+j)*n]=1.0; A[(i+j)+i*n]=1.0; }
  }
  for (j=1;j<3;j++) { A[0+j*n]=-1.0*(j+2); A[j+0*n]=-1.0*(j+2); }
  PetscCall(DSRestoreArray(ds,DS_MAT_A,&A));
  PetscCall(DSSetState(ds,DS_STATE_RAW));
  if (verbose) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n"));
    PetscCall(DSView(ds,viewer));
  }

  /* Signature matrix */
  PetscCall(PetscMalloc2(n,&s,n,&ns));
  s[0] = -1.0;
  for (i=1;i<n-1;i++) s[i]=1.0;
  s[n-1] = -1.0;

  /* Orthogonalize and show signature */
  PetscCall(DSPseudoOrthogonalize(ds,DS_MAT_A,n,s,NULL,ns));
  if (verbose) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"After pseudo-orthogonalize - - - - - - - - -\n"));
    PetscCall(DSView(ds,viewer));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Resulting signature:\n"));
  for (i=0;i<n;i++) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%g\n",(double)ns[i]));
  PetscCall(PetscFree2(s,ns));

  PetscCall(DSDestroy(&ds));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1

TEST*/
