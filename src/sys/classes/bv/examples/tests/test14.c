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

static char help[] = "Test BV created from a dense Mat.\n\n";

#include <slepcbv.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  BV             X;
  Mat            A,B,M;
  PetscInt       i,j,n=20,k=8,Istart,Iend;
  PetscViewer    view;
  PetscBool      verbose;
  PetscReal      norm;
  PetscScalar    alpha;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-verbose",&verbose);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test BV created from a dense Mat (length %D, k=%D).\n",n,k);CHKERRQ(ierr);

  /* Create dense matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,k);CHKERRQ(ierr);
  ierr = MatSetType(A,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (j=0;j<k;j++) {
    for (i=0;i<=n/2;i++) {
      if (i+j<n && i>=Istart && i<Iend) {
        alpha = (3.0*i+j-2)/(2*(i+j+1));
        ierr = MatSetValue(A,i+j,j,alpha,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Create BV object X */
  ierr = BVCreateFromMat(A,&X);CHKERRQ(ierr);
  ierr = BVSetFromOptions(X);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)X,"X");CHKERRQ(ierr);

  /* Set up viewer */
  ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&view);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscViewerPushFormat(view,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = BVView(X,view);CHKERRQ(ierr);
  }

  /* Test BVCreateMat */
  ierr = BVCreateMat(X,&B);CHKERRQ(ierr);
  ierr = MatAXPY(B,-1.0,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(B,NORM_1,&norm);CHKERRQ(ierr);
  if (norm<100*PETSC_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of difference < 100*eps\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of difference: %g\n",(double)norm);CHKERRQ(ierr);
  }

  /* Test BVOrthogonalize */
  ierr = BVOrthogonalize(X,NULL);CHKERRQ(ierr);
  if (verbose) {
    ierr = BVView(X,view);CHKERRQ(ierr);
  }

  /* Check orthogonality */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&M);CHKERRQ(ierr);
  ierr = MatShift(M,1.0);CHKERRQ(ierr);   /* set leading part to identity */
  ierr = BVDot(X,X,M);CHKERRQ(ierr);
  ierr = MatShift(M,-1.0);CHKERRQ(ierr);
  ierr = MatNorm(M,NORM_1,&norm);CHKERRQ(ierr);
  if (norm<100*PETSC_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality < 100*eps\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %g\n",(double)norm);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&M);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = BVDestroy(&X);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}
