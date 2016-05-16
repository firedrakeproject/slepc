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

static char help[] = "Test DSNEP.\n\n";

#include <slepcds.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DS             ds;
  FN             f1,f2,f3,funs[3];
  PetscScalar    *Id,*A,*B,*wr,*wi,coeffs[2];
  PetscReal      tau=0.001,h,a=20,xi,re,im;
  PetscInt       i,n=10,ld,nev;
  PetscViewer    viewer;
  PetscBool      verbose;

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-tau",&tau,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Solve a Dense System of type NEP - dimension %D, tau=%g.\n",n,(double)tau);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-verbose",&verbose);CHKERRQ(ierr);

  /* Create DS object */
  ierr = DSCreate(PETSC_COMM_WORLD,&ds);CHKERRQ(ierr);
  ierr = DSSetType(ds,DSNEP);CHKERRQ(ierr);
  ierr = DSSetFromOptions(ds);CHKERRQ(ierr);

  /* Set functions (prior to DSAllocate) */
  ierr = FNCreate(PETSC_COMM_WORLD,&f1);CHKERRQ(ierr);
  ierr = FNSetType(f1,FNRATIONAL);CHKERRQ(ierr);
  coeffs[0] = -1.0; coeffs[1] = 0.0;
  ierr = FNRationalSetNumerator(f1,2,coeffs);CHKERRQ(ierr);

  ierr = FNCreate(PETSC_COMM_WORLD,&f2);CHKERRQ(ierr);
  ierr = FNSetType(f2,FNRATIONAL);CHKERRQ(ierr);
  coeffs[0] = 1.0;
  ierr = FNRationalSetNumerator(f2,1,coeffs);CHKERRQ(ierr);

  ierr = FNCreate(PETSC_COMM_WORLD,&f3);CHKERRQ(ierr);
  ierr = FNSetType(f3,FNEXP);CHKERRQ(ierr);
  ierr = FNSetScale(f3,-tau,1.0);CHKERRQ(ierr);

  funs[0] = f1;
  funs[1] = f2;
  funs[2] = f3;
  ierr = DSNEPSetFN(ds,3,funs);CHKERRQ(ierr);

  /* Set dimensions */
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

  /* Fill matrices */
  ierr = DSGetArray(ds,DS_MAT_E0,&Id);CHKERRQ(ierr);
  for (i=0;i<n;i++) Id[i+i*ld]=1.0;
  ierr = DSRestoreArray(ds,DS_MAT_E0,&Id);CHKERRQ(ierr);
  h = PETSC_PI/(PetscReal)(n+1);
  ierr = DSGetArray(ds,DS_MAT_E1,&A);CHKERRQ(ierr);
  for (i=0;i<n;i++) A[i+i*ld]=-2.0/(h*h)+a;
  for (i=1;i<n;i++) {
    A[i+(i-1)*ld]=1.0/(h*h);
    A[(i-1)+i*ld]=1.0/(h*h);
  }
  ierr = DSRestoreArray(ds,DS_MAT_E1,&A);CHKERRQ(ierr);
  ierr = DSGetArray(ds,DS_MAT_E2,&B);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    xi = (i+1)*h;
    B[i+i*ld] = -4.1+xi*(1.0-PetscExpReal(xi-PETSC_PI));
  }
  ierr = DSRestoreArray(ds,DS_MAT_E2,&B);CHKERRQ(ierr);

  if (verbose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n");CHKERRQ(ierr);
    ierr = DSView(ds,viewer);CHKERRQ(ierr);
  }

  /* Solve */
  ierr = PetscMalloc2(n,&wr,n,&wi);CHKERRQ(ierr);
  ierr = DSSolve(ds,wr,wi);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n");CHKERRQ(ierr);
    ierr = DSView(ds,viewer);CHKERRQ(ierr);
  }

  /* Print first eigenvalue */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Computed eigenvalue =\n");CHKERRQ(ierr);
  nev = 1;
  for (i=0;i<nev;i++) {
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

  ierr = PetscFree2(wr,wi);CHKERRQ(ierr);
  ierr = FNDestroy(&f1);CHKERRQ(ierr);
  ierr = FNDestroy(&f2);CHKERRQ(ierr);
  ierr = FNDestroy(&f3);CHKERRQ(ierr);
  ierr = DSDestroy(&ds);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}
