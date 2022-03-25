/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test BV orthogonalization functions with constraints.\n\n";

#include <slepcbv.h>

int main(int argc,char **argv)
{
  BV             X;
  Mat            M;
  Vec            v,t,*C;
  PetscInt       i,j,n=20,k=8,nc=2,nloc;
  PetscViewer    view;
  PetscBool      verbose;
  PetscReal      norm;
  PetscScalar    alpha;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nc",&nc,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test BV orthogonalization with %" PetscInt_FMT " columns + %" PetscInt_FMT " constraints, of length %" PetscInt_FMT ".\n",k,nc,n));

  /* Create template vector */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&t));
  CHKERRQ(VecSetSizes(t,PETSC_DECIDE,n));
  CHKERRQ(VecSetFromOptions(t));
  CHKERRQ(VecGetLocalSize(t,&nloc));

  /* Create BV object X */
  CHKERRQ(BVCreate(PETSC_COMM_WORLD,&X));
  CHKERRQ(PetscObjectSetName((PetscObject)X,"X"));
  CHKERRQ(BVSetSizes(X,nloc,n,k));
  CHKERRQ(BVSetFromOptions(X));

  /* Generate constraints and attach them to X */
  if (nc>0) {
    CHKERRQ(VecDuplicateVecs(t,nc,&C));
    for (j=0;j<nc;j++) {
      for (i=0;i<=j;i++) CHKERRQ(VecSetValue(C[j],i,1.0,INSERT_VALUES));
      CHKERRQ(VecAssemblyBegin(C[j]));
      CHKERRQ(VecAssemblyEnd(C[j]));
    }
    CHKERRQ(BVInsertConstraints(X,&nc,C));
    CHKERRQ(VecDestroyVecs(nc,&C));
  }

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&view));
  if (verbose) CHKERRQ(PetscViewerPushFormat(view,PETSC_VIEWER_ASCII_MATLAB));

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
  if (verbose) CHKERRQ(BVView(X,view));

  /* Test BVOrthogonalizeColumn */
  for (j=0;j<k;j++) {
    CHKERRQ(BVOrthogonalizeColumn(X,j,NULL,&norm,NULL));
    alpha = 1.0/norm;
    CHKERRQ(BVScaleColumn(X,j,alpha));
  }
  if (verbose) CHKERRQ(BVView(X,view));

  /* Check orthogonality */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&M));
  CHKERRQ(BVDot(X,X,M));
  CHKERRQ(MatShift(M,-1.0));
  CHKERRQ(MatNorm(M,NORM_1,&norm));
  if (norm<100*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality < 100*eps\n"));
  else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %g\n",(double)norm));

  CHKERRQ(MatDestroy(&M));
  CHKERRQ(BVDestroy(&X));
  CHKERRQ(VecDestroy(&t));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   testset:
      output_file: output/test6_1.out
      test:
         suffix: 1
         args: -bv_type {{vecs contiguous svec mat}shared output}
      test:
         suffix: 1_cuda
         args: -bv_type svec -vec_type cuda
         requires: cuda
      test:
         suffix: 2
         nsize: 2
         args: -bv_type {{vecs contiguous svec mat}shared output}
      test:
         suffix: 3
         args: -bv_type {{vecs contiguous svec mat}shared output} -bv_orthog_type mgs
      test:
         suffix: 3_cuda
         args: -bv_type svec -vec_type cuda -bv_orthog_type mgs
         requires: cuda

TEST*/
