/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test BV operations using internal buffer instead of array arguments.\n\n";

#include <slepcbv.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  Vec            t,v,z;
  BV             X;
  PetscInt       i,j,n=10,k=5,l=3;
  PetscReal      nrm;
  PetscViewer    view;
  PetscBool      verbose;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-l",&l,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test BV with %" PetscInt_FMT " columns of dimension %" PetscInt_FMT ".\n",k,n));

  /* Create template vector */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&t));
  CHKERRQ(VecSetSizes(t,PETSC_DECIDE,n));
  CHKERRQ(VecSetFromOptions(t));

  /* Create BV object X */
  CHKERRQ(BVCreate(PETSC_COMM_WORLD,&X));
  CHKERRQ(PetscObjectSetName((PetscObject)X,"X"));
  CHKERRQ(BVSetSizesFromVec(X,t,k));
  CHKERRQ(BVSetFromOptions(X));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&view));
  if (verbose) CHKERRQ(PetscViewerPushFormat(view,PETSC_VIEWER_ASCII_MATLAB));

  /* Fill X entries */
  for (j=0;j<k;j++) {
    CHKERRQ(BVGetColumn(X,j,&v));
    CHKERRQ(VecSet(v,0.0));
    for (i=0;i<4;i++) {
      if (i+j<n) CHKERRQ(VecSetValue(v,i+j,(PetscScalar)(3*i+j-2),INSERT_VALUES));
    }
    CHKERRQ(VecAssemblyBegin(v));
    CHKERRQ(VecAssemblyEnd(v));
    CHKERRQ(BVRestoreColumn(X,j,&v));
  }
  if (verbose) CHKERRQ(BVView(X,view));

  /* Test BVDotColumn */
  CHKERRQ(BVDotColumn(X,2,NULL));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After BVDotColumn - - - - - - -\n"));
    CHKERRQ(BVGetBufferVec(X,&z));
    CHKERRQ(VecView(z,view));
  }
  /* Test BVMultColumn */
  CHKERRQ(BVMultColumn(X,-1.0,1.0,2,NULL));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After BVMultColumn - - - - - - - - -\n"));
    CHKERRQ(BVView(X,view));
  }

  CHKERRQ(BVNorm(X,NORM_FROBENIUS,&nrm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Frobenius Norm or X = %g\n",(double)nrm));

  CHKERRQ(BVDestroy(&X));
  CHKERRQ(VecDestroy(&t));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      output_file: output/test13_1.out
      test:
         suffix: 1
         args: -bv_type {{vecs contiguous svec mat}shared output}
      test:
         suffix: 1_cuda
         args: -bv_type svec -vec_type cuda
         requires: cuda

TEST*/
