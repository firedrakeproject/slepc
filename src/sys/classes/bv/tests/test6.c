/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nc",&nc,NULL));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test BV orthogonalization with %" PetscInt_FMT " columns + %" PetscInt_FMT " constraints, of length %" PetscInt_FMT ".\n",k,nc,n));

  /* Create template vector */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&t));
  PetscCall(VecSetSizes(t,PETSC_DECIDE,n));
  PetscCall(VecSetFromOptions(t));
  PetscCall(VecGetLocalSize(t,&nloc));

  /* Create BV object X */
  PetscCall(BVCreate(PETSC_COMM_WORLD,&X));
  PetscCall(PetscObjectSetName((PetscObject)X,"X"));
  PetscCall(BVSetSizes(X,nloc,n,k));
  PetscCall(BVSetFromOptions(X));

  /* Generate constraints and attach them to X */
  if (nc>0) {
    PetscCall(VecDuplicateVecs(t,nc,&C));
    for (j=0;j<nc;j++) {
      for (i=0;i<=j;i++) PetscCall(VecSetValue(C[j],i,1.0,INSERT_VALUES));
      PetscCall(VecAssemblyBegin(C[j]));
      PetscCall(VecAssemblyEnd(C[j]));
    }
    PetscCall(BVInsertConstraints(X,&nc,C));
    PetscCall(VecDestroyVecs(nc,&C));
  }

  /* Set up viewer */
  PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&view));
  if (verbose) PetscCall(PetscViewerPushFormat(view,PETSC_VIEWER_ASCII_MATLAB));

  /* Fill X entries */
  for (j=0;j<k;j++) {
    PetscCall(BVGetColumn(X,j,&v));
    PetscCall(VecSet(v,0.0));
    for (i=0;i<=n/2;i++) {
      if (i+j<n) {
        alpha = (3.0*i+j-2)/(2*(i+j+1));
        PetscCall(VecSetValue(v,i+j,alpha,INSERT_VALUES));
      }
    }
    PetscCall(VecAssemblyBegin(v));
    PetscCall(VecAssemblyEnd(v));
    PetscCall(BVRestoreColumn(X,j,&v));
  }
  if (verbose) PetscCall(BVView(X,view));

  /* Test BVOrthogonalizeColumn */
  for (j=0;j<k;j++) {
    PetscCall(BVOrthogonalizeColumn(X,j,NULL,&norm,NULL));
    alpha = 1.0/norm;
    PetscCall(BVScaleColumn(X,j,alpha));
  }
  if (verbose) PetscCall(BVView(X,view));

  /* Check orthogonality */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&M));
  PetscCall(BVDot(X,X,M));
  PetscCall(MatShift(M,-1.0));
  PetscCall(MatNorm(M,NORM_1,&norm));
  if (norm<100*PETSC_MACHINE_EPSILON) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality < 100*eps\n"));
  else PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %g\n",(double)norm));

  PetscCall(MatDestroy(&M));
  PetscCall(BVDestroy(&X));
  PetscCall(VecDestroy(&t));
  PetscCall(SlepcFinalize());
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
