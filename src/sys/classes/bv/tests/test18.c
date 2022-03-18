/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test BVNormalize().\n\n";

#include <slepcbv.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  BV             X,Y,Z;
  Mat            B;
  Vec            v,t;
  PetscInt       i,j,n=20,k=8,l=3,Istart,Iend;
  PetscViewer    view;
  PetscBool      verbose;
  PetscReal      norm,error;
  PetscScalar    alpha;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar    *eigi;
  PetscRandom    rand;
  PetscReal      normr,normi;
  Vec            vi;
#endif

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-l",&l,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test BV normalization with %" PetscInt_FMT " columns of length %" PetscInt_FMT ".\n",k,n));

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

  /* Create copies on Y and Z */
  CHKERRQ(BVDuplicate(X,&Y));
  CHKERRQ(PetscObjectSetName((PetscObject)Y,"Y"));
  CHKERRQ(BVCopy(X,Y));
  CHKERRQ(BVDuplicate(X,&Z));
  CHKERRQ(PetscObjectSetName((PetscObject)Z,"Z"));
  CHKERRQ(BVCopy(X,Z));
  CHKERRQ(BVSetActiveColumns(X,l,k));
  CHKERRQ(BVSetActiveColumns(Y,l,k));
  CHKERRQ(BVSetActiveColumns(Z,l,k));

  /* Test BVNormalize */
  CHKERRQ(BVNormalize(X,NULL));
  if (verbose) {
    CHKERRQ(BVView(X,view));
  }

  /* Check unit norm of columns */
  error = 0.0;
  for (j=l;j<k;j++) {
    CHKERRQ(BVNormColumn(X,j,NORM_2,&norm));
    error = PetscMax(error,PetscAbsReal(norm-PetscRealConstant(1.0)));
  }
  if (error<100*PETSC_MACHINE_EPSILON) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Deviation from normalized vectors < 100*eps\n"));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Deviation from normalized vectors: %g\n",(double)norm));
  }

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
  if (verbose) {
    CHKERRQ(MatView(B,view));
  }

  /* Test BVNormalize with B-norm */
  CHKERRQ(BVSetMatrix(Y,B,PETSC_FALSE));
  CHKERRQ(BVNormalize(Y,NULL));
  if (verbose) {
    CHKERRQ(BVView(Y,view));
  }

  /* Check unit B-norm of columns */
  error = 0.0;
  for (j=l;j<k;j++) {
    CHKERRQ(BVNormColumn(Y,j,NORM_2,&norm));
    error = PetscMax(error,PetscAbsReal(norm-PetscRealConstant(1.0)));
  }
  if (error<100*PETSC_MACHINE_EPSILON) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Deviation from B-normalized vectors < 100*eps\n"));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Deviation from B-normalized vectors: %g\n",(double)norm));
  }

#if !defined(PETSC_USE_COMPLEX)
  /* fill imaginary parts */
  CHKERRQ(PetscCalloc1(k,&eigi));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rand));
  CHKERRQ(PetscRandomSetFromOptions(rand));
  for (j=l+1;j<k-1;j+=5) {
    CHKERRQ(PetscRandomGetValue(rand,&alpha));
    eigi[j]   =  alpha;
    eigi[j+1] = -alpha;
  }
  CHKERRQ(PetscRandomDestroy(&rand));
  if (verbose) {
    CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,k,eigi,&v));
    CHKERRQ(VecView(v,view));
    CHKERRQ(VecDestroy(&v));
  }

  /* Test BVNormalize with complex conjugate columns */
  CHKERRQ(BVNormalize(Z,eigi));
  if (verbose) {
    CHKERRQ(BVView(Z,view));
  }

  /* Check unit norm of (complex conjugate) columns */
  error = 0.0;
  for (j=l;j<k;j++) {
    if (eigi[j]) {
      CHKERRQ(BVGetColumn(Z,j,&v));
      CHKERRQ(BVGetColumn(Z,j+1,&vi));
      CHKERRQ(VecNormBegin(v,NORM_2,&normr));
      CHKERRQ(VecNormBegin(vi,NORM_2,&normi));
      CHKERRQ(VecNormEnd(v,NORM_2,&normr));
      CHKERRQ(VecNormEnd(vi,NORM_2,&normi));
      CHKERRQ(BVRestoreColumn(Z,j+1,&vi));
      CHKERRQ(BVRestoreColumn(Z,j,&v));
      norm = SlepcAbsEigenvalue(normr,normi);
      j++;
    } else {
      CHKERRQ(BVGetColumn(Z,j,&v));
      CHKERRQ(VecNorm(v,NORM_2,&norm));
      CHKERRQ(BVRestoreColumn(Z,j,&v));
    }
    error = PetscMax(error,PetscAbsReal(norm-PetscRealConstant(1.0)));
  }
  if (error<100*PETSC_MACHINE_EPSILON) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Deviation from normalized conjugate vectors < 100*eps\n"));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Deviation from normalized conjugate vectors: %g\n",(double)norm));
  }
  CHKERRQ(PetscFree(eigi));
#endif

  CHKERRQ(BVDestroy(&X));
  CHKERRQ(BVDestroy(&Y));
  CHKERRQ(BVDestroy(&Z));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(VecDestroy(&t));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      args: -n 250 -l 6 -k 15
      nsize: {{1 2}}
      requires: !complex
      output_file: output/test18_1.out
      test:
         suffix: 1
         args: -bv_type {{vecs contiguous svec mat}}
      test:
         suffix: 1_cuda
         args: -bv_type svec -vec_type cuda
         requires: cuda

   testset:
      args: -n 250 -l 6 -k 15
      nsize: {{1 2}}
      requires: complex
      output_file: output/test18_1_complex.out
      test:
         suffix: 1_complex
         args: -bv_type {{vecs contiguous svec mat}}
      test:
         suffix: 1_cuda_complex
         args: -bv_type svec -vec_type cuda
         requires: cuda

TEST*/
