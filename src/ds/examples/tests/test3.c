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

static char help[] = "Test DSHEP with compact storage.\n\n";

#include "slepcds.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DS             ds;
  PetscReal      *T;
  PetscScalar    *eig;
  PetscInt       i,n=10,l=2,k=5,ld;
  PetscViewer    viewer;
  PetscBool      verbose,extrarow;

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Solve a Dense System of type HEP with compact storage - dimension %D.\n",n);CHKERRQ(ierr); 
  ierr = PetscOptionsGetInt(NULL,"-l",&l,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-k",&k,NULL);CHKERRQ(ierr);
  if (l>n || k>n || l>k) SETERRQ(PETSC_COMM_WORLD,1,"Wrong value of dimensions");
  ierr = PetscOptionsHasName(NULL,"-verbose",&verbose);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,"-extrarow",&extrarow);CHKERRQ(ierr);

  /* Create DS object */
  ierr = DSCreate(PETSC_COMM_WORLD,&ds);CHKERRQ(ierr);
  ierr = DSSetType(ds,DSHEP);CHKERRQ(ierr);
  ierr = DSSetFromOptions(ds);CHKERRQ(ierr);
  ld = n+2;  /* test leading dimension larger than n */
  ierr = DSAllocate(ds,ld);CHKERRQ(ierr);
  ierr = DSSetDimensions(ds,n,0,l,k);CHKERRQ(ierr);
  ierr = DSSetCompact(ds,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DSSetExtraRow(ds,extrarow);CHKERRQ(ierr);

  /* Set up viewer */
  ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
  ierr = DSView(ds,viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  if (verbose) { 
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  }

  /* Fill arrow-tridiagonal matrix */
  ierr = DSGetArrayReal(ds,DS_MAT_T,&T);CHKERRQ(ierr);
  for (i=0;i<n;i++) T[i] = (PetscReal)(i+1);
  for (i=l;i<n-1;i++) T[i+ld] = 1.0;
  if (extrarow) T[n-1+ld] = 1.0;
  ierr = DSRestoreArrayReal(ds,DS_MAT_T,&T);CHKERRQ(ierr);
  if (l==0 && k==0) {
    ierr = DSSetState(ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
  } else {
    ierr = DSSetState(ds,DS_STATE_RAW);CHKERRQ(ierr);
  }
  if (verbose) { 
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n");CHKERRQ(ierr);
    ierr = DSView(ds,viewer);CHKERRQ(ierr);
  }

  /* Solve */
  ierr = PetscMalloc(n*sizeof(PetscScalar),&eig);CHKERRQ(ierr);
  ierr = DSSetEigenvalueComparison(ds,SlepcCompareLargestMagnitude,NULL);CHKERRQ(ierr);
  ierr = DSSolve(ds,eig,NULL);CHKERRQ(ierr);
  ierr = DSSort(ds,eig,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  if (extrarow) { ierr = DSUpdateExtraRow(ds);CHKERRQ(ierr); }
  if (verbose) { 
    ierr = PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n");CHKERRQ(ierr);
    ierr = DSView(ds,viewer);CHKERRQ(ierr);
  }

  /* Print eigenvalues */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Computed eigenvalues =\n",n);CHKERRQ(ierr); 
  for (i=0;i<n;i++) {
    ierr = PetscViewerASCIIPrintf(viewer,"  %.5F\n",PetscRealPart(eig[i]));CHKERRQ(ierr);
  }

  ierr = PetscFree(eig);CHKERRQ(ierr);
  ierr = DSDestroy(&ds);CHKERRQ(ierr);
  ierr = SlepcFinalize();CHKERRQ(ierr);
  return 0;
}
