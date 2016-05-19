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

static char help[] = "Test multiplication of a Mat times a BV.\n\n";

#include <slepcbv.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  Vec            t,v;
  Mat            B;
  BV             X,Y,Z,Zcopy;
  PetscInt       i,j,n=10,k=5,Istart,Iend;
  PetscScalar    *pZ;
  PetscReal      norm;
  PetscViewer    view;
  PetscBool      verbose;

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-verbose",&verbose);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test BVMatMult (n=%D, k=%D).\n",n,k);CHKERRQ(ierr);

  /* Create Laplacian matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)B,"B");CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(B,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    if (i>0) { ierr = MatSetValue(B,i,i-1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (i<n-1) { ierr = MatSetValue(B,i,i+1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    ierr = MatSetValue(B,i,i,2.0,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateVecs(B,&t,NULL);CHKERRQ(ierr);

  /* Create BV object X */
  ierr = BVCreate(PETSC_COMM_WORLD,&X);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)X,"X");CHKERRQ(ierr);
  ierr = BVSetSizesFromVec(X,t,k);CHKERRQ(ierr);
  ierr = BVSetFromOptions(X);CHKERRQ(ierr);

  /* Set up viewer */
  ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&view);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscViewerPushFormat(view,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  }

  /* Fill X entries */
  for (j=0;j<k;j++) {
    ierr = BVGetColumn(X,j,&v);CHKERRQ(ierr);
    ierr = VecSet(v,0.0);CHKERRQ(ierr);
    for (i=Istart;i<PetscMin(j+1,Iend);i++) {
      ierr = VecSetValue(v,i,1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(v);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(v);CHKERRQ(ierr);
    ierr = BVRestoreColumn(X,j,&v);CHKERRQ(ierr);
  }
  if (verbose) {
    ierr = MatView(B,view);CHKERRQ(ierr);
    ierr = BVView(X,view);CHKERRQ(ierr);
  }

  /* Create BV object Y */
  ierr = BVDuplicateResize(X,k+4,&Y);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Y,"Y");CHKERRQ(ierr);
  ierr = BVSetActiveColumns(Y,2,k+2);CHKERRQ(ierr);

  /* Test BVMatMult */
  ierr = BVMatMult(X,B,Y);CHKERRQ(ierr);
  if (verbose) {
    ierr = BVView(Y,view);CHKERRQ(ierr);
  }

  /* Create BV object Z */
  ierr = BVDuplicate(X,&Z);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Z,"Z");CHKERRQ(ierr);

  /* Fill Z entries */
  for (j=0;j<k;j++) {
    ierr = BVGetColumn(Z,j,&v);CHKERRQ(ierr);
    ierr = VecSet(v,0.0);CHKERRQ(ierr);
    if (!Istart) { ierr = VecSetValue(v,0,1.0,ADD_VALUES);CHKERRQ(ierr); }
    if (j<n && j>=Istart && j<Iend) { ierr = VecSetValue(v,j,1.0,ADD_VALUES);CHKERRQ(ierr); }
    if (j+1<n && j>=Istart && j<Iend) { ierr = VecSetValue(v,j+1,-1.0,ADD_VALUES);CHKERRQ(ierr); }
    ierr = VecAssemblyBegin(v);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(v);CHKERRQ(ierr);
    ierr = BVRestoreColumn(Z,j,&v);CHKERRQ(ierr);
  }
  if (verbose) {
    ierr = BVView(Z,view);CHKERRQ(ierr);
  }

  /* Save a copy of Z */
  ierr = BVDuplicate(Z,&Zcopy);CHKERRQ(ierr);
  ierr = BVCopy(Z,Zcopy);CHKERRQ(ierr);

  /* Test BVMult, check result of previous operations */
  ierr = BVMult(Z,-1.0,1.0,Y,NULL);CHKERRQ(ierr);
  ierr = BVNorm(Z,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error: %g\n",(double)norm);CHKERRQ(ierr);

  /* Test BVMatMultColumn, multiply Y(:,2), result in Y(:,3) */
  ierr = BVMatMultColumn(Y,B,2);CHKERRQ(ierr);
  if (verbose) {
    ierr = BVView(Y,view);CHKERRQ(ierr);
  }

  /* Test BVGetArray, modify Z to match Y */
  ierr = BVCopy(Zcopy,Z);CHKERRQ(ierr);
  ierr = BVGetArray(Z,&pZ);CHKERRQ(ierr);
  if (Istart==0) {
    if (Iend<3) SETERRQ(PETSC_COMM_SELF,1,"First process must have at least 3 rows");
    pZ[Iend]   = 5.0;   /* modify 3 first entries of second column */
    pZ[Iend+1] = -4.0;
    pZ[Iend+2] = 1.0;
  }
  ierr = BVRestoreArray(Z,&pZ);CHKERRQ(ierr);
  if (verbose) {
    ierr = BVView(Z,view);CHKERRQ(ierr);
  }

  /* Check result again with BVMult */
  ierr = BVMult(Z,-1.0,1.0,Y,NULL);CHKERRQ(ierr);
  ierr = BVNorm(Z,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error: %g\n",(double)norm);CHKERRQ(ierr);

  ierr = BVDestroy(&X);CHKERRQ(ierr);
  ierr = BVDestroy(&Y);CHKERRQ(ierr);
  ierr = BVDestroy(&Z);CHKERRQ(ierr);
  ierr = BVDestroy(&Zcopy);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = VecDestroy(&t);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}
