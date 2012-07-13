/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain

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

static char help[] = "Test PSSVD.\n\n";

#include "slepcps.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv )
{
  PetscErrorCode ierr;
  PS             ps;
  PetscReal      sigma;
  PetscScalar    *A,*w;
  PetscInt       i,j,k,n=15,m=10,ld;
  PetscViewer    viewer;
  PetscBool      verbose;

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  k = PetscMin(n,m);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Solve a Projected System of type PSSVD - dimension %Dx%D.\n",n,m);CHKERRQ(ierr); 
  ierr = PetscOptionsHasName(PETSC_NULL,"-verbose",&verbose);CHKERRQ(ierr);

  /* Create PS object */
  ierr = PSCreate(PETSC_COMM_WORLD,&ps);CHKERRQ(ierr);
  ierr = PSSetType(ps,PSSVD);CHKERRQ(ierr);
  ierr = PSSetFromOptions(ps);CHKERRQ(ierr);
  ld = n+2;  /* test leading dimension larger than n */
  ierr = PSAllocate(ps,ld);CHKERRQ(ierr);
  ierr = PSSetDimensions(ps,n,m,0,0);CHKERRQ(ierr);

  /* Set up viewer */
  ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
  ierr = PSView(ps,viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  if (verbose) { 
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  }

  /* Fill with a rectangular Toeplitz matrix */
  ierr = PSGetArray(ps,PS_MAT_A,&A);CHKERRQ(ierr);
  for (i=0;i<k;i++) A[i+i*ld]=1.0;
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) { if ((i+j)<m) A[i+(i+j)*ld]=(PetscScalar)(j+1); }
  }
  for (j=1;j<n/2;j++) {
    for (i=0;i<n-j;i++) { if ((i+j)<n && i<m) A[(i+j)+i*ld]=-1.0; }
  }
  ierr = PSRestoreArray(ps,PS_MAT_A,&A);CHKERRQ(ierr);
  ierr = PSSetState(ps,PS_STATE_RAW);CHKERRQ(ierr);
  if (verbose) { 
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n");CHKERRQ(ierr);
    ierr = PSView(ps,viewer);CHKERRQ(ierr);
  }

  /* Solve */
  ierr = PetscMalloc(k*sizeof(PetscScalar),&w);CHKERRQ(ierr);
  ierr = PSSetEigenvalueComparison(ps,SlepcCompareLargestReal,PETSC_NULL);CHKERRQ(ierr);
  ierr = PSSolve(ps,w,PETSC_NULL);CHKERRQ(ierr);
  if (verbose) { 
    ierr = PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n");CHKERRQ(ierr);
    ierr = PSView(ps,viewer);CHKERRQ(ierr);
  }
  
  /* Print singular values */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Computed singular values =\n",n);CHKERRQ(ierr); 
  for (i=0;i<k;i++) {
    sigma = PetscRealPart(w[i]);
    ierr = PetscViewerASCIIPrintf(viewer,"  %.5F\n",sigma);CHKERRQ(ierr);
  }
  ierr = PetscFree(w);CHKERRQ(ierr);
  ierr = PSDestroy(&ps);CHKERRQ(ierr);
  ierr = SlepcFinalize();CHKERRQ(ierr);
  return 0;
}
