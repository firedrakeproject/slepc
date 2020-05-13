/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2020, Universitat Politecnica de Valencia, Spain

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
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-l",&l,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-verbose",&verbose);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test BV normalization with %D columns of length %D.\n",k,n);CHKERRQ(ierr);

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
  ierr = BVSetActiveColumns(X,l,k);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(Y,l,k);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(Z,l,k);CHKERRQ(ierr);

  /* Test BVNormalize */
  ierr = BVNormalize(X,NULL);CHKERRQ(ierr);
  if (verbose) {
    ierr = BVView(X,view);CHKERRQ(ierr);
  }

  /* Check unit norm of columns */
  error = 0.0;
  for (j=l;j<k;j++) {
    ierr = BVNormColumn(X,j,NORM_2,&norm);CHKERRQ(ierr);
    error = PetscMax(error,PetscAbsReal(norm-PetscRealConstant(1.0)));
  }
  if (error<100*PETSC_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Deviation from normalized vectors < 100*eps\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Deviation from normalized vectors: %g\n",(double)norm);CHKERRQ(ierr);
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
  if (verbose) {
    ierr = MatView(B,view);CHKERRQ(ierr);
  }

  /* Test BVNormalize with B-norm */
  ierr = BVSetMatrix(Y,B,PETSC_FALSE);CHKERRQ(ierr);
  ierr = BVNormalize(Y,NULL);CHKERRQ(ierr);
  if (verbose) {
    ierr = BVView(Y,view);CHKERRQ(ierr);
  }

  /* Check unit B-norm of columns */
  error = 0.0;
  for (j=l;j<k;j++) {
    ierr = BVNormColumn(Y,j,NORM_2,&norm);CHKERRQ(ierr);
    error = PetscMax(error,PetscAbsReal(norm-PetscRealConstant(1.0)));
  }
  if (error<100*PETSC_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Deviation from B-normalized vectors < 100*eps\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Deviation from B-normalized vectors: %g\n",(double)norm);CHKERRQ(ierr);
  }

#if !defined(PETSC_USE_COMPLEX)
  /* fill imaginary parts */
  ierr = PetscCalloc1(k,&eigi);CHKERRQ(ierr);
  ierr = BVGetRandomContext(Z,&rand);CHKERRQ(ierr);
  for (j=l+1;j<k-1;j+=5) {
    ierr = PetscRandomGetValue(rand,&alpha);CHKERRQ(ierr);
    eigi[j]   =  alpha;
    eigi[j+1] = -alpha;
  }
  if (verbose) {
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,k,eigi,&v);CHKERRQ(ierr);
    ierr = VecView(v,view);CHKERRQ(ierr);
    ierr = VecDestroy(&v);CHKERRQ(ierr);
  }

  /* Test BVNormalize with complex conjugate columns */
  ierr = BVNormalize(Z,eigi);CHKERRQ(ierr);
  if (verbose) {
    ierr = BVView(Z,view);CHKERRQ(ierr);
  }

  /* Check unit norm of (complex conjugate) columns */
  error = 0.0;
  for (j=l;j<k;j++) {
    if (eigi[j]) {
      ierr = BVGetColumn(Z,j,&v);CHKERRQ(ierr);
      ierr = BVGetColumn(Z,j+1,&vi);CHKERRQ(ierr);
      ierr = VecNormBegin(v,NORM_2,&normr);CHKERRQ(ierr);
      ierr = VecNormBegin(vi,NORM_2,&normi);CHKERRQ(ierr);
      ierr = VecNormEnd(v,NORM_2,&normr);CHKERRQ(ierr);
      ierr = VecNormEnd(vi,NORM_2,&normi);CHKERRQ(ierr);
      ierr = BVRestoreColumn(Z,j+1,&vi);CHKERRQ(ierr);
      ierr = BVRestoreColumn(Z,j,&v);CHKERRQ(ierr);
      norm = SlepcAbsEigenvalue(normr,normi);
      j++;
    } else {
      ierr = BVGetColumn(Z,j,&v);CHKERRQ(ierr);
      ierr = VecNorm(v,NORM_2,&norm);CHKERRQ(ierr);
      ierr = BVRestoreColumn(Z,j,&v);CHKERRQ(ierr);
    }
    error = PetscMax(error,PetscAbsReal(norm-PetscRealConstant(1.0)));
  }
  if (error<100*PETSC_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Deviation from normalized conjugate vectors < 100*eps\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Deviation from normalized conjugate vectors: %g\n",(double)norm);CHKERRQ(ierr);
  }
  ierr = PetscFree(eigi);CHKERRQ(ierr);
#endif

  ierr = BVDestroy(&X);CHKERRQ(ierr);
  ierr = BVDestroy(&Y);CHKERRQ(ierr);
  ierr = BVDestroy(&Z);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = VecDestroy(&t);CHKERRQ(ierr);
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
