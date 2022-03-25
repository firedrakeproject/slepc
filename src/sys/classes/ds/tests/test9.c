/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test DSGHEP.\n\n";

#include <slepcds.h>

/*
   Compute the norm of the j-th column of matrix mat in ds
 */
PetscErrorCode ComputeNorm(DS ds,DSMatType mat,PetscInt j,PetscReal *onrm)
{
  PetscScalar    *X;
  PetscReal      aux,nrm=0.0;
  PetscInt       i,n,ld;

  PetscFunctionBeginUser;
  CHKERRQ(DSGetLeadingDimension(ds,&ld));
  CHKERRQ(DSGetDimensions(ds,&n,NULL,NULL,NULL));
  CHKERRQ(DSGetArray(ds,mat,&X));
  for (i=0;i<n;i++) {
    aux = PetscAbsScalar(X[i+j*ld]);
    nrm += aux*aux;
  }
  CHKERRQ(DSRestoreArray(ds,mat,&X));
  *onrm = PetscSqrtReal(nrm);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  DS             ds;
  SlepcSC        sc;
  PetscReal      re;
  PetscScalar    *A,*B,*eig;
  PetscReal      nrm;
  PetscInt       i,j,n=10,ld;
  PetscViewer    viewer;
  PetscBool      verbose;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Solve a System of type GHEP - dimension %" PetscInt_FMT ".\n",n));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));

  /* Create DS object */
  CHKERRQ(DSCreate(PETSC_COMM_WORLD,&ds));
  CHKERRQ(DSSetType(ds,DSGHEP));
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

  /* Fill with a symmetric Toeplitz matrix */
  CHKERRQ(DSGetArray(ds,DS_MAT_A,&A));
  CHKERRQ(DSGetArray(ds,DS_MAT_B,&B));
  for (i=0;i<n;i++) A[i+i*ld]=2.0;
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) { A[i+(i+j)*ld]=1.0; A[(i+j)+i*ld]=1.0; }
  }
  for (j=1;j<3;j++) { A[0+j*ld]=-1.0*(j+2); A[j+0*ld]=-1.0*(j+2); }
  /* Diagonal matrix */
  for (i=0;i<n;i++) B[i+i*ld]=0.1*(i+1);
  CHKERRQ(DSRestoreArray(ds,DS_MAT_A,&A));
  CHKERRQ(DSRestoreArray(ds,DS_MAT_B,&B));
  CHKERRQ(DSSetState(ds,DS_STATE_RAW));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }

  /* Solve */
  CHKERRQ(PetscMalloc1(n,&eig));
  CHKERRQ(PetscNew(&sc));
  sc->comparison    = SlepcCompareLargestMagnitude;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  CHKERRQ(DSSetSlepcSC(ds,sc));
  CHKERRQ(DSSolve(ds,eig,NULL));
  CHKERRQ(DSSort(ds,eig,NULL,NULL,NULL,NULL));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }

  /* Print eigenvalues */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Computed eigenvalues =\n"));
  for (i=0;i<n;i++) {
    re = PetscRealPart(eig[i]);
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %.5f\n",(double)re));
  }

  /* Eigenvectors */
  j = 0;
  CHKERRQ(DSVectors(ds,DS_MAT_X,&j,NULL));  /* all eigenvectors */
  CHKERRQ(ComputeNorm(ds,DS_MAT_X,0,&nrm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of 1st vector = %.3f\n",(double)nrm));
  CHKERRQ(DSVectors(ds,DS_MAT_X,NULL,NULL));  /* all eigenvectors */
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After vectors - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }

  CHKERRQ(PetscFree(eig));
  CHKERRQ(PetscFree(sc));
  CHKERRQ(DSDestroy(&ds));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      requires: !single

TEST*/
