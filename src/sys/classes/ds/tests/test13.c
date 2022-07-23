/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test DSHEP with block size larger than one.\n\n";

#include <slepcds.h>

int main(int argc,char **argv)
{
  DS             ds;
  SlepcSC        sc;
  PetscScalar    *A,*eig;
  PetscInt       i,j,n,ld,bs,maxbw=3,nblks=8;
  PetscViewer    viewer;
  PetscBool      verbose;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-maxbw",&maxbw,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nblks",&nblks,NULL));
  n = maxbw*nblks;
  bs = maxbw;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Solve a block HEP Dense System - dimension %" PetscInt_FMT " (bandwidth=%" PetscInt_FMT ", blocks=%" PetscInt_FMT ").\n",n,maxbw,nblks));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));

  /* Create DS object */
  PetscCall(DSCreate(PETSC_COMM_WORLD,&ds));
  PetscCall(DSSetType(ds,DSHEP));
  PetscCall(DSSetMethod(ds,3));   /* Select block divide-and-conquer */
  PetscCall(DSSetBlockSize(ds,bs));
  PetscCall(DSSetFromOptions(ds));
  ld = n;
  PetscCall(DSAllocate(ds,ld));
  PetscCall(DSSetDimensions(ds,n,0,0));

  /* Set up viewer */
  PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
  PetscCall(DSView(ds,viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  if (verbose) PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));

  /* Fill with a symmetric band Toeplitz matrix */
  PetscCall(DSGetArray(ds,DS_MAT_A,&A));
  for (i=0;i<n;i++) A[i+i*ld]=2.0;
  for (j=1;j<=bs;j++) {
    for (i=0;i<n-j;i++) { A[i+(i+j)*ld]=1.0; A[(i+j)+i*ld]=1.0; }
  }
  PetscCall(DSRestoreArray(ds,DS_MAT_A,&A));
  PetscCall(DSSetState(ds,DS_STATE_RAW));
  if (verbose) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n"));
    PetscCall(DSView(ds,viewer));
  }

  /* Solve */
  PetscCall(PetscMalloc1(n,&eig));
  PetscCall(DSGetSlepcSC(ds,&sc));
  sc->comparison    = SlepcCompareSmallestReal;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  PetscCall(DSSolve(ds,eig,NULL));
  PetscCall(DSSort(ds,eig,NULL,NULL,NULL,NULL));
  if (verbose) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n"));
    PetscCall(DSView(ds,viewer));
  }

  /* Print eigenvalues */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Computed eigenvalues =\n"));
  for (i=0;i<n;i++) PetscCall(PetscViewerASCIIPrintf(viewer,"  %.5f\n",(double)PetscRealPart(eig[i])));

  PetscCall(PetscFree(eig));
  PetscCall(DSDestroy(&ds));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      requires: !complex !single

   test:
      suffix: 2
      args: -nblks 1
      requires: !complex

TEST*/
