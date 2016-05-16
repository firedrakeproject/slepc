/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.

   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test DSGNHEP.\n\n";

#include <slepcds.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DS             ds;
  SlepcSC        sc;
  PetscScalar    *A,*B,*X,*wr,*wi;
  PetscReal      re,im,rnorm,aux;
  PetscInt       i,j,n=10,ld;
  PetscViewer    viewer;
  PetscBool      verbose;

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Solve a Dense System of type GNHEP - dimension %D.\n",n);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-verbose",&verbose);CHKERRQ(ierr);

  /* Create DS object */
  ierr = DSCreate(PETSC_COMM_WORLD,&ds);CHKERRQ(ierr);
  ierr = DSSetType(ds,DSGNHEP);CHKERRQ(ierr);
  ierr = DSSetFromOptions(ds);CHKERRQ(ierr);
  ld = n+2;  /* test leading dimension larger than n */
  ierr = DSAllocate(ds,ld);CHKERRQ(ierr);
  ierr = DSSetDimensions(ds,n,0,0,0);CHKERRQ(ierr);

  /* Set up viewer */
  ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
  ierr = DSView(ds,viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  }

  /* Fill A with Grcar matrix */
  ierr = DSGetArray(ds,DS_MAT_A,&A);CHKERRQ(ierr);
  ierr = PetscMemzero(A,sizeof(PetscScalar)*ld*n);CHKERRQ(ierr);
  for (i=1;i<n;i++) A[i+(i-1)*ld]=-1.0;
  for (j=0;j<4;j++) {
    for (i=0;i<n-j;i++) A[i+(i+j)*ld]=1.0;
  }
  ierr = DSRestoreArray(ds,DS_MAT_A,&A);CHKERRQ(ierr);
  /* Fill B with an identity matrix */
  ierr = DSGetArray(ds,DS_MAT_B,&B);CHKERRQ(ierr);
  ierr = PetscMemzero(B,sizeof(PetscScalar)*ld*n);CHKERRQ(ierr);
  for (i=0;i<n;i++) B[i+i*ld]=1.0;
  ierr = DSRestoreArray(ds,DS_MAT_B,&B);CHKERRQ(ierr);

  if (verbose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n");CHKERRQ(ierr);
    ierr = DSView(ds,viewer);CHKERRQ(ierr);
  }

  /* Solve */
  ierr = PetscMalloc2(n,&wr,n,&wi);CHKERRQ(ierr);
  ierr = DSGetSlepcSC(ds,&sc);CHKERRQ(ierr);
  sc->comparison    = SlepcCompareLargestMagnitude;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  ierr = DSSolve(ds,wr,wi);CHKERRQ(ierr);
  ierr = DSSort(ds,wr,wi,NULL,NULL,NULL);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n");CHKERRQ(ierr);
    ierr = DSView(ds,viewer);CHKERRQ(ierr);
  }

  /* Print eigenvalues */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Computed eigenvalues =\n");CHKERRQ(ierr);
  for (i=0;i<n;i++) {
#if defined(PETSC_USE_COMPLEX)
    re = PetscRealPart(wr[i]);
    im = PetscImaginaryPart(wr[i]);
#else
    re = wr[i];
    im = wi[i];
#endif
    if (PetscAbs(im)<1e-10) {
      ierr = PetscViewerASCIIPrintf(viewer,"  %.5f\n",(double)re);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  %.5f%+.5fi\n",(double)re,(double)im);CHKERRQ(ierr);
    }
  }

  /* Eigenvectors */
  j = 2;
  ierr = DSVectors(ds,DS_MAT_X,&j,&rnorm);CHKERRQ(ierr);  /* third eigenvector */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Value of rnorm for 3rd vector = %.3f\n",(double)rnorm);CHKERRQ(ierr);
  ierr = DSVectors(ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);  /* all eigenvectors */
  j = 0;
  rnorm = 0.0;
  ierr = DSGetArray(ds,DS_MAT_X,&X);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
#if defined(PETSC_USE_COMPLEX)
    aux = PetscAbsScalar(X[i+j*ld]);
#else
    if (PetscAbs(wi[j])==0.0) aux = PetscAbsScalar(X[i+j*ld]);
    else aux = SlepcAbsEigenvalue(X[i+j*ld],X[i+(j+1)*ld]);
#endif
    rnorm += aux*aux;
  }
  ierr = DSRestoreArray(ds,DS_MAT_X,&X);CHKERRQ(ierr);
  rnorm = PetscSqrtReal(rnorm);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of 1st vector = %.3f\n",(double)rnorm);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"After vectors - - - - - - - - -\n");CHKERRQ(ierr);
    ierr = DSView(ds,viewer);CHKERRQ(ierr);
  }

  ierr = PetscFree2(wr,wi);CHKERRQ(ierr);
  ierr = DSDestroy(&ds);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}
