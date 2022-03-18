/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test DSPEP.\n\n";

#include <slepcds.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DS             ds;
  SlepcSC        sc;
  Mat            X;
  Vec            x0;
  PetscScalar    *K,*C,*M,*wr,*wi,z=1.0;
  PetscReal      re,im,nrm,*pbc;
  PetscInt       i,j,n=10,d=2,ld;
  PetscViewer    viewer;
  PetscBool      verbose;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Solve a Dense System of type PEP - n=%" PetscInt_FMT ".\n",n));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));

  /* Create DS object */
  CHKERRQ(DSCreate(PETSC_COMM_WORLD,&ds));
  CHKERRQ(DSSetType(ds,DSPEP));
  CHKERRQ(DSSetFromOptions(ds));
  CHKERRQ(DSPEPSetDegree(ds,d));

  /* Set dimensions */
  ld = n+2;  /* test leading dimension larger than n */
  CHKERRQ(DSAllocate(ds,ld));
  CHKERRQ(DSSetDimensions(ds,n,0,0));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
  CHKERRQ(DSView(ds,viewer));
  CHKERRQ(PetscViewerPopFormat(viewer));
  if (verbose) {
    CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));
  }

  /* Fill matrices */
  CHKERRQ(DSGetArray(ds,DS_MAT_E0,&K));
  for (i=0;i<n-1;i++) K[i+i*ld] = 2.0*n;
  K[n-1+(n-1)*ld] = 1.0*n;
  for (i=1;i<n;i++) {
    K[i+(i-1)*ld] = -1.0*n;
    K[(i-1)+i*ld] = -1.0*n;
  }
  CHKERRQ(DSRestoreArray(ds,DS_MAT_E0,&K));
  CHKERRQ(DSGetArray(ds,DS_MAT_E1,&C));
  C[n-1+(n-1)*ld] = 2.0*PETSC_PI/z;
  CHKERRQ(DSRestoreArray(ds,DS_MAT_E1,&C));
  CHKERRQ(DSGetArray(ds,DS_MAT_E2,&M));
  for (i=0;i<n-1;i++) M[i+i*ld] = -4.0*PETSC_PI*PETSC_PI/n;
  M[i+i*ld] = -2.0*PETSC_PI*PETSC_PI/n;
  CHKERRQ(DSRestoreArray(ds,DS_MAT_E2,&M));

  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }

  /* Solve */
  CHKERRQ(PetscMalloc2(d*n,&wr,d*n,&wi));
  CHKERRQ(DSGetSlepcSC(ds,&sc));
  sc->comparison    = SlepcCompareLargestReal;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  CHKERRQ(DSSolve(ds,wr,wi));
  CHKERRQ(DSSort(ds,wr,wi,NULL,NULL,NULL));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }

  /* Print polynomial coefficients */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Polynomial coefficients (alpha,beta,gamma) =\n"));
  CHKERRQ(DSPEPGetCoefficients(ds,&pbc));
  for (j=0;j<3;j++) {
    for (i=0;i<d+1;i++) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %.5f",(double)pbc[j+3*i]));
    }
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
  }
  CHKERRQ(PetscFree(pbc));

  /* Print eigenvalues */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Computed eigenvalues =\n"));
  for (i=0;i<d*n;i++) {
#if defined(PETSC_USE_COMPLEX)
    re = PetscRealPart(wr[i]);
    im = PetscImaginaryPart(wr[i]);
#else
    re = wr[i];
    im = wi[i];
#endif
    if (PetscAbs(im)<1e-10) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %.5f\n",(double)re));
    } else {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %.5f%+.5fi\n",(double)re,(double)im));
    }
  }

  /* Eigenvectors */
  CHKERRQ(DSVectors(ds,DS_MAT_X,NULL,NULL));  /* all eigenvectors */
  CHKERRQ(DSGetMat(ds,DS_MAT_X,&X));
  CHKERRQ(MatCreateVecs(X,NULL,&x0));
  CHKERRQ(MatGetColumnVector(X,x0,0));
  CHKERRQ(VecNorm(x0,NORM_2,&nrm));
  CHKERRQ(MatDestroy(&X));
  CHKERRQ(VecDestroy(&x0));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of 1st vector = %.3f\n",(double)nrm));
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
      requires: !single

TEST*/
