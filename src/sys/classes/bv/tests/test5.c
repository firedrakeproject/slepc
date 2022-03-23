/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test BV operations with indefinite inner product.\n\n";

#include <slepcbv.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  Vec            t,v,w,omega;
  Mat            B,M;
  BV             X,Y;
  PetscInt       i,j,n=10,k=5,l,Istart,Iend;
  PetscScalar    alpha;
  PetscReal      nrm;
  PetscViewer    view;
  PetscBool      verbose;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test BV with indefinite inner product (n=%" PetscInt_FMT ", k=%" PetscInt_FMT ").\n",n,k));

  /* Create inner product matrix (standard involutionary permutation) */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(B));
  CHKERRQ(PetscObjectSetName((PetscObject)B,"B"));

  CHKERRQ(MatGetOwnershipRange(B,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) CHKERRQ(MatSetValue(B,i,n-i-1,1.0,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCreateVecs(B,&t,NULL));

  /* Create BV object X */
  CHKERRQ(BVCreate(PETSC_COMM_WORLD,&X));
  CHKERRQ(PetscObjectSetName((PetscObject)X,"X"));
  CHKERRQ(BVSetSizesFromVec(X,t,k));
  CHKERRQ(BVSetFromOptions(X));
  CHKERRQ(BVSetMatrix(X,B,PETSC_TRUE));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&view));
  if (verbose) CHKERRQ(PetscViewerPushFormat(view,PETSC_VIEWER_ASCII_MATLAB));

  /* Fill X entries */
  l = -3;
  for (j=0;j<k;j++) {
    CHKERRQ(BVGetColumn(X,j,&v));
    CHKERRQ(VecSet(v,-1.0));
    for (i=0;i<n/2;i++) {
      if (i+j<n) {
        l = (l + 3*i+j-2) % n;
        CHKERRQ(VecSetValue(v,i+j,(PetscScalar)l,INSERT_VALUES));
      }
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

  /* Create a copy on Y */
  CHKERRQ(BVDuplicate(X,&Y));
  CHKERRQ(PetscObjectSetName((PetscObject)Y,"Y"));
  CHKERRQ(BVCopy(X,Y));

  /* Check orthogonality */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&M));
  CHKERRQ(BVDot(Y,Y,M));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,k,&omega));
  CHKERRQ(BVGetSignature(Y,omega));
  CHKERRQ(VecScale(omega,-1.0));
  CHKERRQ(MatDiagonalSet(M,omega,ADD_VALUES));
  CHKERRQ(MatNorm(M,NORM_1,&nrm));
  if (nrm<100*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality < 100*eps\n"));
  else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %g\n",(double)nrm));

  /* Test BVSetSignature */
  CHKERRQ(VecScale(omega,-1.0));
  CHKERRQ(BVSetSignature(Y,omega));
  CHKERRQ(VecDestroy(&omega));

  /* Test BVApplyMatrix */
  CHKERRQ(VecDuplicate(t,&w));
  CHKERRQ(BVGetColumn(X,0,&v));
  CHKERRQ(BVApplyMatrix(X,v,w));
  CHKERRQ(BVApplyMatrix(X,w,t));
  CHKERRQ(VecAXPY(t,-1.0,v));
  CHKERRQ(BVRestoreColumn(X,0,&v));
  CHKERRQ(VecNorm(t,NORM_2,&nrm));
  PetscCheck(PetscAbsReal(nrm)<10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_FP,"Wrong value, nrm = %g",(double)nrm);

  CHKERRQ(BVApplyMatrixBV(X,Y));
  CHKERRQ(BVGetColumn(Y,0,&v));
  CHKERRQ(VecAXPY(w,-1.0,v));
  CHKERRQ(BVRestoreColumn(Y,0,&v));
  CHKERRQ(VecNorm(w,NORM_2,&nrm));
  PetscCheck(PetscAbsReal(nrm)<10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_FP,"Wrong value, nrm = %g",(double)nrm);

  CHKERRQ(BVDestroy(&X));
  CHKERRQ(BVDestroy(&Y));
  CHKERRQ(MatDestroy(&M));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(VecDestroy(&w));
  CHKERRQ(VecDestroy(&t));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      output_file: output/test5_1.out
      args: -bv_orthog_refine always
      test:
         suffix: 1
         args: -bv_type {{vecs contiguous svec mat}shared output}
      test:
         suffix: 1_cuda
         args: -bv_type svec -mat_type aijcusparse
         requires: cuda
      test:
         suffix: 2
         args: -bv_type {{vecs contiguous svec mat}shared output} -bv_orthog_type mgs
      test:
         suffix: 2_cuda
         args: -bv_type svec -mat_type aijcusparse -bv_orthog_type mgs
         requires: cuda

TEST*/
