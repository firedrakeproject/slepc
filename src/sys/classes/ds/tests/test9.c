/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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
  PetscCall(DSGetLeadingDimension(ds,&ld));
  PetscCall(DSGetDimensions(ds,&n,NULL,NULL,NULL));
  PetscCall(DSGetArray(ds,mat,&X));
  for (i=0;i<n;i++) {
    aux = PetscAbsScalar(X[i+j*ld]);
    nrm += aux*aux;
  }
  PetscCall(DSRestoreArray(ds,mat,&X));
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

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Solve a System of type GHEP - dimension %" PetscInt_FMT ".\n",n));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));

  /* Create DS object */
  PetscCall(DSCreate(PETSC_COMM_WORLD,&ds));
  PetscCall(DSSetType(ds,DSGHEP));
  PetscCall(DSSetFromOptions(ds));
  ld = n+2;  /* test leading dimension larger than n */
  PetscCall(DSAllocate(ds,ld));
  PetscCall(DSSetDimensions(ds,n,0,0));

  /* Set up viewer */
  PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
  PetscCall(DSView(ds,viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  if (verbose) PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));

  /* Fill with a symmetric Toeplitz matrix */
  PetscCall(DSGetArray(ds,DS_MAT_A,&A));
  PetscCall(DSGetArray(ds,DS_MAT_B,&B));
  for (i=0;i<n;i++) A[i+i*ld]=2.0;
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) { A[i+(i+j)*ld]=1.0; A[(i+j)+i*ld]=1.0; }
  }
  for (j=1;j<3;j++) { A[0+j*ld]=-1.0*(j+2); A[j+0*ld]=-1.0*(j+2); }
  /* Diagonal matrix */
  for (i=0;i<n;i++) B[i+i*ld]=0.1*(i+1);
  PetscCall(DSRestoreArray(ds,DS_MAT_A,&A));
  PetscCall(DSRestoreArray(ds,DS_MAT_B,&B));
  PetscCall(DSSetState(ds,DS_STATE_RAW));
  if (verbose) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n"));
    PetscCall(DSView(ds,viewer));
  }

  /* Solve */
  PetscCall(PetscMalloc1(n,&eig));
  PetscCall(PetscNew(&sc));
  sc->comparison    = SlepcCompareLargestMagnitude;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  PetscCall(DSSetSlepcSC(ds,sc));
  PetscCall(DSSolve(ds,eig,NULL));
  PetscCall(DSSort(ds,eig,NULL,NULL,NULL,NULL));
  if (verbose) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n"));
    PetscCall(DSView(ds,viewer));
  }

  /* Print eigenvalues */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Computed eigenvalues =\n"));
  for (i=0;i<n;i++) {
    re = PetscRealPart(eig[i]);
    PetscCall(PetscViewerASCIIPrintf(viewer,"  %.5f\n",(double)re));
  }

  /* Eigenvectors */
  j = 0;
  PetscCall(DSVectors(ds,DS_MAT_X,&j,NULL));  /* all eigenvectors */
  PetscCall(ComputeNorm(ds,DS_MAT_X,0,&nrm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm of 1st vector = %.3f\n",(double)nrm));
  PetscCall(DSVectors(ds,DS_MAT_X,NULL,NULL));  /* all eigenvectors */
  if (verbose) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"After vectors - - - - - - - - -\n"));
    PetscCall(DSView(ds,viewer));
  }

  PetscCall(PetscFree(eig));
  PetscCall(PetscFree(sc));
  PetscCall(DSDestroy(&ds));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      requires: !single

TEST*/
