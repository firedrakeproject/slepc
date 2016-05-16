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

static char help[] = "Test BV block orthogonalization.\n\n";

#include <slepcbv.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode    ierr;
  BV                X,Y,Z,cached;
  Mat               B,M;
  Vec               v,t;
  PetscInt          i,j,n=20,l=2,k=8,Istart,Iend;
  PetscViewer       view;
  PetscBool         verbose;
  PetscReal         norm;
  PetscScalar       alpha;
  BVOrthogBlockType btype;

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-l",&l,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-verbose",&verbose);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test BV block orthogonalization (length %D, l=%D, k=%D).\n",n,l,k);CHKERRQ(ierr);

  /* Create template vector */
  ierr = VecCreate(PETSC_COMM_WORLD,&t);CHKERRQ(ierr);
  ierr = VecSetSizes(t,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(t);CHKERRQ(ierr);

  /* Create BV object X */
  ierr = BVCreate(PETSC_COMM_WORLD,&X);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)X,"X");CHKERRQ(ierr);
  ierr = BVSetSizesFromVec(X,t,k);CHKERRQ(ierr);
  ierr = BVSetFromOptions(X);CHKERRQ(ierr);
  ierr = BVGetOrthogonalization(X,NULL,NULL,NULL,&btype);CHKERRQ(ierr);

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
  if (btype==BV_ORTHOG_BLOCK_GS) {  /* GS requires the leading columns to be orthogonal */
    for (j=0;j<l;j++) {
      ierr = BVOrthogonalizeColumn(X,j,NULL,&norm,NULL);CHKERRQ(ierr);
      alpha = 1.0/norm;
      ierr = BVScaleColumn(X,j,alpha);CHKERRQ(ierr);
    }
  }
  if (verbose) {
    ierr = BVView(X,view);CHKERRQ(ierr);
  }

  /* Create copy on Y */
  ierr = BVDuplicate(X,&Y);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Y,"Y");CHKERRQ(ierr);
  ierr = BVCopy(X,Y);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(Y,l,k);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(X,l,k);CHKERRQ(ierr);

  /* Test BVOrthogonalize */
  ierr = BVOrthogonalize(Y,NULL);CHKERRQ(ierr);
  if (verbose) {
    ierr = BVView(Y,view);CHKERRQ(ierr);
  }

  /* Check orthogonality */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&M);CHKERRQ(ierr);
  ierr = MatShift(M,1.0);CHKERRQ(ierr);   /* set leading part to identity */
  ierr = BVDot(Y,Y,M);CHKERRQ(ierr);
  ierr = MatShift(M,-1.0);CHKERRQ(ierr);
  ierr = MatNorm(M,NORM_1,&norm);CHKERRQ(ierr);
  if (norm<100*PETSC_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality < 100*eps\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %g\n",(double)norm);CHKERRQ(ierr);
  }

  /* Create inner product matrix */
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

  /* Prepare to repeat test, now with a non-standard inner product */
  ierr = BVCopy(X,Y);CHKERRQ(ierr);
  ierr = BVDuplicate(X,&Z);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Z,"Z");CHKERRQ(ierr);
  ierr = BVSetActiveColumns(Z,l,k);CHKERRQ(ierr);
  ierr = BVSetMatrix(X,B,PETSC_FALSE);CHKERRQ(ierr);
  ierr = BVSetMatrix(Y,B,PETSC_FALSE);CHKERRQ(ierr);
  if (btype==BV_ORTHOG_BLOCK_GS) {  /* GS requires the leading columns to be orthogonal */
    for (j=0;j<l;j++) {
      ierr = BVOrthogonalizeColumn(Y,j,NULL,&norm,NULL);CHKERRQ(ierr);
      alpha = 1.0/norm;
      ierr = BVScaleColumn(Y,j,alpha);CHKERRQ(ierr);
    }
  }
  if (verbose) {
    ierr = BVView(X,view);CHKERRQ(ierr);
  }

  /* Test BVOrthogonalize */
  ierr = BVOrthogonalize(Y,NULL);CHKERRQ(ierr);
  if (verbose) {
    ierr = BVView(Y,view);CHKERRQ(ierr);
  }

  /* Extract cached BV and check it is equal to B*X */
  ierr = BVGetCachedBV(Y,&cached);CHKERRQ(ierr);
  ierr = BVMatMult(X,B,Z);CHKERRQ(ierr);
  ierr = BVMult(Z,-1.0,1.0,cached,NULL);CHKERRQ(ierr);
  ierr = BVNorm(Z,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
  if (norm<100*PETSC_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual ||cached-BX|| < 100*eps\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual ||cached-BX||: %g\n",(double)norm);CHKERRQ(ierr);
  }

  /* Check orthogonality */
  ierr = MatZeroEntries(M);CHKERRQ(ierr);
  ierr = MatShift(M,1.0);CHKERRQ(ierr);   /* set leading part to identity */
  ierr = BVDot(Y,Y,M);CHKERRQ(ierr);
  ierr = MatShift(M,-1.0);CHKERRQ(ierr);
  ierr = MatNorm(M,NORM_1,&norm);CHKERRQ(ierr);
  if (norm<100*PETSC_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality < 100*eps\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %g\n",(double)norm);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&M);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = BVDestroy(&X);CHKERRQ(ierr);
  ierr = BVDestroy(&Y);CHKERRQ(ierr);
  ierr = BVDestroy(&Z);CHKERRQ(ierr);
  ierr = VecDestroy(&t);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}
