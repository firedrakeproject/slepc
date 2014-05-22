/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

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

static char help[] = "Test BV operations, changing the number of active columns.\n\n";

#include <slepcbv.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  Vec            t,v;
  Mat            Q,M;
  BV             X,Y;
  PetscInt       i,j,n=10,kx=6,lx=3,ky=5,ly=2;
  PetscScalar    *q,*z;
  PetscReal      nrm;
  PetscViewer    view;
  PetscBool      verbose,trans;

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-kx",&kx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-lx",&lx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-ky",&ky,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-ly",&ly,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,"-verbose",&verbose);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"First BV with %D active columns (%D leading columns) of dimension %D.\n",kx,lx,n);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Second BV with %D active columns (%D leading columns) of dimension %D.\n",ky,ly,n);CHKERRQ(ierr);

  /* Create template vector */
  ierr = VecCreate(PETSC_COMM_WORLD,&t);CHKERRQ(ierr);
  ierr = VecSetSizes(t,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(t);CHKERRQ(ierr);

  /* Create BV object X */
  ierr = BVCreate(PETSC_COMM_WORLD,&X);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)X,"X");CHKERRQ(ierr);
  ierr = BVSetSizesFromVec(X,t,kx+2);CHKERRQ(ierr);  /* two extra columns to test active columns */
  ierr = BVSetFromOptions(X);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(X,lx,kx);CHKERRQ(ierr);

  /* Set up viewer */
  ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&view);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscViewerPushFormat(view,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  }

  /* Fill X entries */
  for (j=0;j<kx+2;j++) {
    ierr = BVGetColumn(X,j,&v);CHKERRQ(ierr);
    ierr = VecZeroEntries(v);CHKERRQ(ierr);
    for (i=0;i<4;i++) {
      if (i+j<n) {
        ierr = VecSetValue(v,i+j,(PetscScalar)(3*i+j-2),INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = VecAssemblyBegin(v);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(v);CHKERRQ(ierr);
    ierr = BVRestoreColumn(X,j,&v);CHKERRQ(ierr);
  }
  if (verbose) {
    ierr = BVView(X,view);CHKERRQ(ierr);
  }

  /* Create BV object Y */
  ierr = BVCreate(PETSC_COMM_WORLD,&Y);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Y,"Y");CHKERRQ(ierr);
  ierr = BVSetSizesFromVec(Y,t,ky+1);CHKERRQ(ierr);
  ierr = BVSetFromOptions(Y);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(Y,ly,ky);CHKERRQ(ierr);

  /* Fill Y entries */
  for (j=0;j<ky+1;j++) {
    ierr = BVGetColumn(Y,j,&v);CHKERRQ(ierr);
    ierr = VecSet(v,(PetscScalar)(j+1)/4.0);CHKERRQ(ierr);
    ierr = BVRestoreColumn(Y,j,&v);CHKERRQ(ierr);
  }
  if (verbose) {
    ierr = BVView(Y,view);CHKERRQ(ierr);
  }

  /* Create Mat */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,kx,ky,NULL,&Q);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Q,"Q");CHKERRQ(ierr);
  ierr = MatDenseGetArray(Q,&q);CHKERRQ(ierr);
  for (i=0;i<kx;i++)
    for (j=0;j<ky;j++)
      q[i+j*kx] = (i<j)? 2.0: -0.5;
  ierr = MatDenseRestoreArray(Q,&q);CHKERRQ(ierr);
  if (verbose) {
    ierr = MatView(Q,NULL);CHKERRQ(ierr);
  }

  /* Test BVMult */
  ierr = BVMult(Y,2.0,1.0,X,Q);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"After BVMult - - - - - - - - -\n");CHKERRQ(ierr);
    ierr = BVView(Y,view);CHKERRQ(ierr);
  }

  /* Test BVMultVec */
  ierr = BVGetColumn(Y,0,&v);CHKERRQ(ierr);
  ierr = PetscMalloc1(kx-lx,&z);CHKERRQ(ierr);
  z[0] = 2.0;
  for (i=1;i<kx-lx;i++) z[i] = -0.5*z[i-1];
  ierr = BVMultVec(X,-1.0,1.0,v,z);CHKERRQ(ierr);
  ierr = PetscFree(z);CHKERRQ(ierr);
  ierr = BVRestoreColumn(Y,0,&v);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"After BVMultVec - - - - - - -\n");CHKERRQ(ierr);
    ierr = BVView(Y,view);CHKERRQ(ierr);
  }

  /* Test BVDot */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,ky,kx,NULL,&M);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)M,"M");CHKERRQ(ierr);
  ierr = BVDot(X,Y,M);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"After BVDot - - - - - - - - -\n");CHKERRQ(ierr);
    ierr = MatView(M,NULL);CHKERRQ(ierr);
  }

  /* Test BVDotVec */
  ierr = BVGetColumn(Y,0,&v);CHKERRQ(ierr);
  ierr = PetscMalloc1(kx-lx,&z);CHKERRQ(ierr);
  ierr = BVDotVec(X,v,z);CHKERRQ(ierr);
  ierr = BVRestoreColumn(Y,0,&v);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"After BVDotVec - - - - - - -\n");CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,kx-lx,z,&v);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)v,"z");CHKERRQ(ierr);
    ierr = VecView(v,view);CHKERRQ(ierr);
    ierr = VecDestroy(&v);CHKERRQ(ierr);
  }
  ierr = PetscFree(z);CHKERRQ(ierr);

  /* Test BVMultInPlace and BVScale */
  ierr = PetscOptionsHasName(NULL,"-trans",&trans);CHKERRQ(ierr);
  if (trans) {
    Mat Qt;
    ierr = MatTranspose(Q,MAT_INITIAL_MATRIX,&Qt);CHKERRQ(ierr);
    ierr = BVMultInPlaceTranspose(X,Qt,lx+1,ky);CHKERRQ(ierr);
    ierr = MatDestroy(&Qt);CHKERRQ(ierr);
  } else {
    ierr = BVMultInPlace(X,Q,lx+1,ky);CHKERRQ(ierr);
  }
  ierr = BVScale(X,2.0);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"After BVMultInPlace - - - - -\n");CHKERRQ(ierr);
    ierr = BVView(X,view);CHKERRQ(ierr);
  }

  /* Test BVNorm */
  ierr = BVNorm(X,0,NORM_2,&nrm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"2-Norm or X[0] = %g\n",(double)nrm);CHKERRQ(ierr);
  ierr = BVNorm(X,-1,NORM_FROBENIUS,&nrm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Frobenius Norm or X = %g\n",(double)nrm);CHKERRQ(ierr);

  ierr = BVDestroy(&X);CHKERRQ(ierr);
  ierr = BVDestroy(&Y);CHKERRQ(ierr);
  ierr = MatDestroy(&Q);CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  ierr = VecDestroy(&t);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return 0;
}
