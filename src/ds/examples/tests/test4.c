/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

#include "slepcds.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv )
{
  PetscErrorCode ierr;
  DS             ds;
  PetscScalar    *A,*B,*wr,*wi;
  PetscReal      re,im;
  PetscInt       i,j,n=10,ld;
  PetscViewer    viewer;
  PetscBool      verbose;

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Solve a Dense System of type GNHEP - dimension %D.\n",n);CHKERRQ(ierr); 
  ierr = PetscOptionsHasName(PETSC_NULL,"-verbose",&verbose);CHKERRQ(ierr);

  /* Create DS object */
  ierr = DSCreate(PETSC_COMM_WORLD,&ds);CHKERRQ(ierr);
  ierr = DSSetType(ds,DSGNHEP);CHKERRQ(ierr);
  ierr = DSSetFromOptions(ds);CHKERRQ(ierr);
  ld = n+2;  /* test leading dimension larger than n */
  ierr = DSAllocate(ds,ld);CHKERRQ(ierr);
  ierr = DSSetDimensions(ds,n,PETSC_IGNORE,0,0);CHKERRQ(ierr);

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
  ierr = PetscMalloc(n*sizeof(PetscScalar),&wr);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscScalar),&wi);CHKERRQ(ierr);
  ierr = DSSetEigenvalueComparison(ds,SlepcCompareLargestMagnitude,PETSC_NULL);CHKERRQ(ierr);
  ierr = DSSolve(ds,wr,wi);CHKERRQ(ierr);
  ierr = DSSort(ds,wr,wi,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  if (verbose) { 
    ierr = PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n");CHKERRQ(ierr);
    ierr = DSView(ds,viewer);CHKERRQ(ierr);
  }

  /* Print eigenvalues */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Computed eigenvalues =\n",n);CHKERRQ(ierr); 
  for (i=0;i<n;i++) {
#if defined(PETSC_USE_COMPLEX)
    re = PetscRealPart(wr[i]);
    im = PetscImaginaryPart(wr[i]);
#else
    re = wr[i];
    im = wi[i];
#endif 
    if (PetscAbs(im)<1e-10) { ierr = PetscViewerASCIIPrintf(viewer,"  %.5F\n",re);CHKERRQ(ierr); }
    else { ierr = PetscViewerASCIIPrintf(viewer,"  %.5F%+.5Fi\n",re,im);CHKERRQ(ierr); }
  }

  ierr = PetscFree(wr);CHKERRQ(ierr);
  ierr = PetscFree(wi);CHKERRQ(ierr);
  ierr = DSDestroy(&ds);CHKERRQ(ierr);
  ierr = SlepcFinalize();CHKERRQ(ierr);
  return 0;
}
