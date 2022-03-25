/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

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

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-maxbw",&maxbw,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nblks",&nblks,NULL));
  n = maxbw*nblks;
  bs = maxbw;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Solve a block HEP Dense System - dimension %" PetscInt_FMT " (bandwidth=%" PetscInt_FMT ", blocks=%" PetscInt_FMT ").\n",n,maxbw,nblks));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));

  /* Create DS object */
  CHKERRQ(DSCreate(PETSC_COMM_WORLD,&ds));
  CHKERRQ(DSSetType(ds,DSHEP));
  CHKERRQ(DSSetMethod(ds,3));   /* Select block divide-and-conquer */
  CHKERRQ(DSSetBlockSize(ds,bs));
  CHKERRQ(DSSetFromOptions(ds));
  ld = n;
  CHKERRQ(DSAllocate(ds,ld));
  CHKERRQ(DSSetDimensions(ds,n,0,0));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
  CHKERRQ(DSView(ds,viewer));
  CHKERRQ(PetscViewerPopFormat(viewer));
  if (verbose) CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));

  /* Fill with a symmetric band Toeplitz matrix */
  CHKERRQ(DSGetArray(ds,DS_MAT_A,&A));
  for (i=0;i<n;i++) A[i+i*ld]=2.0;
  for (j=1;j<=bs;j++) {
    for (i=0;i<n-j;i++) { A[i+(i+j)*ld]=1.0; A[(i+j)+i*ld]=1.0; }
  }
  CHKERRQ(DSRestoreArray(ds,DS_MAT_A,&A));
  CHKERRQ(DSSetState(ds,DS_STATE_RAW));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }

  /* Solve */
  CHKERRQ(PetscMalloc1(n,&eig));
  CHKERRQ(DSGetSlepcSC(ds,&sc));
  sc->comparison    = SlepcCompareSmallestReal;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  CHKERRQ(DSSolve(ds,eig,NULL));
  CHKERRQ(DSSort(ds,eig,NULL,NULL,NULL,NULL));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }

  /* Print eigenvalues */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Computed eigenvalues =\n"));
  for (i=0;i<n;i++) CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %.5f\n",(double)PetscRealPart(eig[i])));

  CHKERRQ(PetscFree(eig));
  CHKERRQ(DSDestroy(&ds));
  CHKERRQ(SlepcFinalize());
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
