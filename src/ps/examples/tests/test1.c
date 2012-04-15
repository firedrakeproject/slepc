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

static char help[] = "Test PSNHEP.\n\n";

#include "slepcps.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv )
{
  PetscErrorCode ierr;
  PS             ps;
  PetscScalar    *A,*wr,*wi;
  PetscInt       i,j,n=7,ld;
  PetscViewer    viewer;

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Solve a Projected System of type NHEP - dimension %D.\n",n);CHKERRQ(ierr); 

  /* Create PS object */
  ierr = PSCreate(PETSC_COMM_WORLD,&ps);CHKERRQ(ierr);
  ierr = PSSetType(ps,PSNHEP);CHKERRQ(ierr);
  ierr = PSSetFromOptions(ps);CHKERRQ(ierr);
  ld = n+2;  /* test leading dimension larger than n */
  ierr = PSAllocate(ps,ld);CHKERRQ(ierr);
  ierr = PSSetDimensions(ps,n,0,0);CHKERRQ(ierr);

  /* Set up viewer */
  ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
  ierr = PSView(ps,viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);

  /* Fill with Grcar matrix */
  ierr = PSGetArray(ps,PS_MAT_A,&A);CHKERRQ(ierr);
  for (i=1;i<n;i++) A[i+(i-1)*ld]=-1.0;
  for (j=0;j<4;j++) {
    for (i=0;i<n-j;i++) A[i+(i+j)*ld]=1.0;
  }
  ierr = PSRestoreArray(ps,PS_MAT_A,&A);CHKERRQ(ierr);
  ierr = PSSetState(ps,PS_STATE_INTERMEDIATE);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n");CHKERRQ(ierr);
  ierr = PSView(ps,viewer);CHKERRQ(ierr);

  /* Solve */
  ierr = PetscMalloc(n*sizeof(PetscScalar),&wr);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscScalar),&wi);CHKERRQ(ierr);
  ierr = PSSolve(ps,wr,wi);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n");CHKERRQ(ierr);
  ierr = PSView(ps,viewer);CHKERRQ(ierr);

  /* Sort */
  ierr = PSSort(ps,wr,wi,SlepcCompareLargestMagnitude,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"After sort - - - - - - - - -\n");CHKERRQ(ierr);
  ierr = PSView(ps,viewer);CHKERRQ(ierr);

  ierr = PetscFree(wr);CHKERRQ(ierr);
  ierr = PetscFree(wi);CHKERRQ(ierr);
  ierr = PSDestroy(&ps);CHKERRQ(ierr);
  ierr = SlepcFinalize();CHKERRQ(ierr);
  return 0;
}
