/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test BV orthogonalization with selected columns.\n\n";

#include <slepcbv.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  BV             X;
  Vec            v,t,z;
  PetscInt       i,j,n=20,k=8;
  PetscViewer    view;
  PetscBool      verbose,*which;
  PetscReal      norm;
  PetscScalar    alpha,*pz;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test BV orthogonalization with selected columns of length %" PetscInt_FMT ".\n",n));

  /* Create template vector */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&t));
  CHKERRQ(VecSetSizes(t,PETSC_DECIDE,n));
  CHKERRQ(VecSetFromOptions(t));

  /* Create BV object X */
  CHKERRQ(BVCreate(PETSC_COMM_WORLD,&X));
  CHKERRQ(PetscObjectSetName((PetscObject)X,"X"));
  CHKERRQ(BVSetSizesFromVec(X,t,k));
  CHKERRQ(BVSetOrthogonalization(X,BV_ORTHOG_MGS,BV_ORTHOG_REFINE_IFNEEDED,PETSC_DEFAULT,BV_ORTHOG_BLOCK_GS));
  CHKERRQ(BVSetFromOptions(X));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&view));
  if (verbose) {
    CHKERRQ(PetscViewerPushFormat(view,PETSC_VIEWER_ASCII_MATLAB));
  }

  /* Fill X entries */
  for (j=0;j<k;j++) {
    CHKERRQ(BVGetColumn(X,j,&v));
    CHKERRQ(VecSet(v,0.0));
    for (i=0;i<=n/2;i++) {
      if (i+j<n) {
        alpha = (3.0*i+j-2)/(2*(i+j+1));
        CHKERRQ(VecSetValue(v,i+j,alpha,INSERT_VALUES));
      }
    }
    CHKERRQ(VecAssemblyBegin(v));
    CHKERRQ(VecAssemblyEnd(v));
    CHKERRQ(BVRestoreColumn(X,j,&v));
  }
  if (verbose) {
    CHKERRQ(BVView(X,view));
  }

  /* Orthonormalize first k-1 columns */
  for (j=0;j<k-1;j++) {
    CHKERRQ(BVOrthogonalizeColumn(X,j,NULL,&norm,NULL));
    alpha = 1.0/norm;
    CHKERRQ(BVScaleColumn(X,j,alpha));
  }
  if (verbose) {
    CHKERRQ(BVView(X,view));
  }

  /* Select odd columns and orthogonalize last column against those only */
  CHKERRQ(PetscMalloc1(k,&which));
  for (i=0;i<k;i++) which[i] = (i%2)? PETSC_TRUE: PETSC_FALSE;
  CHKERRQ(BVOrthogonalizeSomeColumn(X,k-1,which,NULL,NULL,NULL));
  CHKERRQ(PetscFree(which));
  if (verbose) {
    CHKERRQ(BVView(X,view));
  }

  /* Check orthogonality */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Orthogonalization coefficients:\n"));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,k-1,&z));
  CHKERRQ(PetscObjectSetName((PetscObject)z,"z"));
  CHKERRQ(VecGetArray(z,&pz));
  CHKERRQ(BVDotColumn(X,k-1,pz));
  for (i=0;i<k-1;i++) {
    if (PetscAbsScalar(pz[i])<5.0*PETSC_MACHINE_EPSILON) pz[i]=0.0;
  }
  CHKERRQ(VecRestoreArray(z,&pz));
  CHKERRQ(VecView(z,view));
  CHKERRQ(VecDestroy(&z));

  CHKERRQ(BVDestroy(&X));
  CHKERRQ(VecDestroy(&t));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      output_file: output/test8_1.out
      test:
         suffix: 1
         args: -bv_type {{vecs contiguous svec mat}shared output}
      test:
         suffix: 1_cuda
         args: -bv_type svec -vec_type cuda
         requires: cuda
      test:
         suffix: 2
         args: -bv_type {{vecs contiguous svec mat}shared output} -bv_orthog_refine never
         requires: !single
      test:
         suffix: 3
         args: -bv_type {{vecs contiguous svec mat}shared output} -bv_orthog_refine always
      test:
         suffix: 4
         args: -bv_type {{vecs contiguous svec mat}shared output} -bv_orthog_type mgs
      test:
         suffix: 4_cuda
         args: -bv_type svec -vec_type cuda -bv_orthog_type mgs
         requires: cuda

TEST*/
