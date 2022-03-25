/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test BV operations with non-standard inner product.\n\n";

#include <slepcbv.h>

int main(int argc,char **argv)
{
  Vec            t,v;
  Mat            B,M;
  BV             X;
  PetscInt       i,j,n=10,k=5,Istart,Iend;
  PetscScalar    alpha;
  PetscReal      nrm;
  PetscViewer    view;
  PetscBool      verbose;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test BV with non-standard inner product (n=%" PetscInt_FMT ", k=%" PetscInt_FMT ").\n",n,k));

  /* Create inner product matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(B));
  CHKERRQ(PetscObjectSetName((PetscObject)B,"B"));

  CHKERRQ(MatGetOwnershipRange(B,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) CHKERRQ(MatSetValue(B,i,i-1,-1.0,INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(B,i,i+1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(B,i,i,2.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCreateVecs(B,&t,NULL));

  /* Create BV object X */
  CHKERRQ(BVCreate(PETSC_COMM_WORLD,&X));
  CHKERRQ(PetscObjectSetName((PetscObject)X,"X"));
  CHKERRQ(BVSetSizesFromVec(X,t,k));
  CHKERRQ(BVSetFromOptions(X));
  CHKERRQ(BVSetMatrix(X,B,PETSC_FALSE));

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
  if (verbose) {
    CHKERRQ(MatView(B,view));
    CHKERRQ(BVView(X,view));
  }

  /* Test BVNormColumn */
  CHKERRQ(BVNormColumn(X,0,NORM_2,&nrm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"B-Norm of X[0] = %g\n",(double)nrm));

  /* Test BVOrthogonalizeColumn */
  for (j=0;j<k;j++) {
    CHKERRQ(BVOrthogonalizeColumn(X,j,NULL,&nrm,NULL));
    alpha = 1.0/nrm;
    CHKERRQ(BVScaleColumn(X,j,alpha));
  }
  if (verbose) CHKERRQ(BVView(X,view));

  /* Check orthogonality */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&M));
  CHKERRQ(BVDot(X,X,M));
  CHKERRQ(MatShift(M,-1.0));
  CHKERRQ(MatNorm(M,NORM_1,&nrm));
  if (nrm<100*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality < 100*eps\n"));
  else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %g\n",(double)nrm));

  /* Test BVNormVecBegin/End */
  CHKERRQ(BVGetColumn(X,0,&v));
  CHKERRQ(BVNormVecBegin(X,v,NORM_1,&nrm));
  CHKERRQ(BVNormVecEnd(X,v,NORM_1,&nrm));
  CHKERRQ(BVRestoreColumn(X,0,&v));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"B-Norm of X[0] = %g\n",(double)nrm));

  CHKERRQ(BVDestroy(&X));
  CHKERRQ(MatDestroy(&M));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(VecDestroy(&t));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   testset:
      output_file: output/test3_1.out
      test:
         suffix: 1
         args: -bv_type {{vecs contiguous svec mat}shared output}
      test:
         suffix: 1_svec_vecs
         args: -bv_type svec -bv_matmult vecs
      test:
         suffix: 1_cuda
         args: -bv_type svec -mat_type aijcusparse
         requires: cuda
      test:
         suffix: 2
         nsize: 2
         args: -bv_type {{vecs contiguous svec mat}shared output}
      test:
         suffix: 3
         nsize: 2
         args: -bv_type {{vecs contiguous svec mat}shared output} -bv_orthog_type mgs

TEST*/
