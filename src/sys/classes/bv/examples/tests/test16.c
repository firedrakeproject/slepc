/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2017, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test tensor BV.\n\n";

#include <slepcbv.h>

int main(int argc,char **argv)
{
  PetscErrorCode    ierr;
  Vec               t,v;
  Mat               S;
  BV                U,V,UU;
  PetscInt          i,j,n=10,k=5,l=3,d=3,deg;
  //PetscScalar       *q,*z;
  PetscViewer       view;
  PetscBool         verbose;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-l",&l,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-d",&d,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-verbose",&verbose);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test tensor BV of degree %D with %D columns of dimension %D*d.\n",d,k,n);CHKERRQ(ierr);

  /* Create template vector */
  ierr = VecCreate(PETSC_COMM_WORLD,&t);CHKERRQ(ierr);
  ierr = VecSetSizes(t,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(t);CHKERRQ(ierr);

  /* Create BV object U */
  ierr = BVCreate(PETSC_COMM_WORLD,&U);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)U,"U");CHKERRQ(ierr);
  ierr = BVSetSizesFromVec(U,t,k+d);CHKERRQ(ierr);
  ierr = BVSetFromOptions(U);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)U,"U");CHKERRQ(ierr);

  /* Fill first d columns of U */
  for (j=0;j<d;j++) {
    ierr = BVGetColumn(U,j,&v);CHKERRQ(ierr);
    ierr = VecSet(v,0.0);CHKERRQ(ierr);
    for (i=0;i<4;i++) {
      if (i+j<n) {
        ierr = VecSetValue(v,i+j,(PetscScalar)(3*i+j-2),INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = VecAssemblyBegin(v);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(v);CHKERRQ(ierr);
    ierr = BVRestoreColumn(U,j,&v);CHKERRQ(ierr);
  }

  /* Create tensor BV */
  ierr = BVCreateTensor(U,d,&V);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)V,"V");CHKERRQ(ierr);
  ierr = BVTensorGetDegree(V,&deg);CHKERRQ(ierr);
  if (deg!=d) SETERRQ(PETSC_COMM_WORLD,1,"Wrong degree");

  /* Set up viewer */
  ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&view);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(view,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
  ierr = BVView(V,view);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(view);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscViewerPushFormat(view,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = BVView(V,view);CHKERRQ(ierr);
  }

  /* Build first column from previously introduced coefficients */
  ierr = BVTensorBuildFirstColumn(V,d);CHKERRQ(ierr);
  if (verbose) {
    ierr = BVView(V,view);CHKERRQ(ierr);
  }

  ierr = BVTensorGetFactors(V,&UU,&S);CHKERRQ(ierr);
  if (verbose) {
    ierr = BVView(UU,view);CHKERRQ(ierr);
  }
  ierr = BVTensorRestoreFactors(V,&UU,&S);CHKERRQ(ierr);

  ierr = BVDestroy(&U);CHKERRQ(ierr);
  ierr = BVDestroy(&V);CHKERRQ(ierr);
  ierr = VecDestroy(&t);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}
