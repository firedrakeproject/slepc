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

static char help[] = "Test BV orthogonalization functions.\n\n";

#include <slepcbv.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  BV             X,Y,Z;
  Mat            M,R;
  Vec            v,t,e;
  PetscInt       i,j,n=20,k=8;
  PetscViewer    view;
  PetscBool      verbose;
  PetscReal      norm;
  PetscScalar    alpha;

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-verbose",&verbose);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test BV orthogonalization with %D columns of length %D.\n",k,n);CHKERRQ(ierr);

  /* Create template vector */
  ierr = VecCreate(PETSC_COMM_WORLD,&t);CHKERRQ(ierr);
  ierr = VecSetSizes(t,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(t);CHKERRQ(ierr);

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
    for (i=0;i<=n/2;i++) {
      if (i+j<n) {
        alpha = (3.0*i+j-2)/(2*(i+j+1));
        ierr = VecSetValue(v,i+j,alpha,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = VecAssemblyBegin(v);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(v);CHKERRQ(ierr);
    ierr = BVRestoreColumn(X,j,&v);CHKERRQ(ierr);
  }
  if (verbose) {
    ierr = BVView(X,view);CHKERRQ(ierr);
  }

  /* Create copies on Y and Z */
  ierr = BVDuplicate(X,&Y);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Y,"Y");CHKERRQ(ierr);
  ierr = BVCopy(X,Y);CHKERRQ(ierr);
  ierr = BVDuplicate(X,&Z);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Z,"Z");CHKERRQ(ierr);
  ierr = BVCopy(X,Z);CHKERRQ(ierr);

  /* Test BVOrthogonalizeColumn */
  for (j=0;j<k;j++) {
    ierr = BVOrthogonalizeColumn(X,j,NULL,&norm,NULL);CHKERRQ(ierr);
    alpha = 1.0/norm;
    ierr = BVScaleColumn(X,j,alpha);CHKERRQ(ierr);
  }
  if (verbose) {
    ierr = BVView(X,view);CHKERRQ(ierr);
  }

  /* Check orthogonality */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&M);CHKERRQ(ierr);
  ierr = BVDot(X,X,M);CHKERRQ(ierr);
  ierr = MatShift(M,-1.0);CHKERRQ(ierr);
  ierr = MatNorm(M,NORM_1,&norm);CHKERRQ(ierr);
  if (norm<100*PETSC_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality < 100*eps\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %g\n",(double)norm);CHKERRQ(ierr);
  }

  /* Test BVOrthogonalize */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&R);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)R,"R");CHKERRQ(ierr);
  ierr = BVOrthogonalize(Y,R);CHKERRQ(ierr);
  if (verbose) {
    ierr = BVView(Y,view);CHKERRQ(ierr);
    ierr = MatView(R,view);CHKERRQ(ierr);
  }

  /* Check orthogonality */
  ierr = BVDot(Y,Y,M);CHKERRQ(ierr);
  ierr = MatShift(M,-1.0);CHKERRQ(ierr);
  ierr = MatNorm(M,NORM_1,&norm);CHKERRQ(ierr);
  if (norm<100*PETSC_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality < 100*eps\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %g\n",(double)norm);CHKERRQ(ierr);
  }

  /* Check residual */
  ierr = BVMult(Z,-1.0,1.0,Y,R);CHKERRQ(ierr);
  ierr = BVNorm(Z,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
  if (norm<100*PETSC_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual ||X-QR|| < 100*eps\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual ||X-QR||: %g\n",(double)norm);CHKERRQ(ierr);
  }

  /* Test BVOrthogonalizeVec */
  ierr = VecDuplicate(t,&e);CHKERRQ(ierr);
  ierr = VecSet(e,1.0);CHKERRQ(ierr);
  ierr = BVOrthogonalizeVec(X,e,NULL,&norm,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of ones(n,1) after orthogonalizing against X: %g\n",(double)norm);CHKERRQ(ierr);

  ierr = MatDestroy(&M);CHKERRQ(ierr);
  ierr = MatDestroy(&R);CHKERRQ(ierr);
  ierr = BVDestroy(&X);CHKERRQ(ierr);
  ierr = BVDestroy(&Y);CHKERRQ(ierr);
  ierr = BVDestroy(&Z);CHKERRQ(ierr);
  ierr = VecDestroy(&e);CHKERRQ(ierr);
  ierr = VecDestroy(&t);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}
