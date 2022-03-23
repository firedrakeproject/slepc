/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test BV bi-orthogonalization functions.\n\n";

#include <slepcbv.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  BV             X,Y;
  Mat            M;
  Vec            v,t;
  PetscInt       i,j,n=20,k=7;
  PetscViewer    view;
  PetscBool      verbose;
  PetscReal      norm,condn=1.0;
  PetscScalar    alpha;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-condn",&condn,NULL));
  PetscCheck(condn>=1.0,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"The condition number must be > 1");
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test BV bi-orthogonalization with %" PetscInt_FMT " columns of length %" PetscInt_FMT ".\n",k,n));
  if (condn>1.0) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," - Using random BVs with condition number = %g\n",(double)condn));

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
  if (condn==1.0) {
    for (j=0;j<k;j++) {
      CHKERRQ(BVGetColumn(X,j,&v));
      CHKERRQ(VecSet(v,0.0));
      for (i=0;i<=n/2;i++) {
        if (i+j<n) {
          alpha = (3.0*i+j-2)/(2*(i+j+1));
#if defined(PETSC_USE_COMPLEX)
          alpha += (1.2*i+j-2)/(0.1*(i+j+1))*PETSC_i;
#endif
          CHKERRQ(VecSetValue(v,i+j,alpha,INSERT_VALUES));
        }
      }
      CHKERRQ(VecAssemblyBegin(v));
      CHKERRQ(VecAssemblyEnd(v));
      CHKERRQ(BVRestoreColumn(X,j,&v));
    }
  } else CHKERRQ(BVSetRandomCond(X,condn));
  if (verbose) CHKERRQ(BVView(X,view));

  /* Create Y and fill its entries */
  CHKERRQ(BVDuplicate(X,&Y));
  CHKERRQ(PetscObjectSetName((PetscObject)Y,"Y"));
  if (condn==1.0) {
    for (j=0;j<k;j++) {
      CHKERRQ(BVGetColumn(Y,j,&v));
      CHKERRQ(VecSet(v,0.0));
      for (i=PetscMin(n,2+(2*j)%6);i<PetscMin(n,6+(3*j)%9);i++) {
        if (i%5 && i!=j) {
          alpha = (1.5*i+j)/(2.2*(i-j));
          CHKERRQ(VecSetValue(v,i+j,alpha,INSERT_VALUES));
        }
      }
      CHKERRQ(VecAssemblyBegin(v));
      CHKERRQ(VecAssemblyEnd(v));
      CHKERRQ(BVRestoreColumn(Y,j,&v));
    }
  } else CHKERRQ(BVSetRandomCond(Y,condn));
  if (verbose) CHKERRQ(BVView(Y,view));

  /* Test BVBiorthonormalizeColumn */
  for (j=0;j<k;j++) CHKERRQ(BVBiorthonormalizeColumn(X,Y,j,NULL));
  if (verbose) {
    CHKERRQ(BVView(X,view));
    CHKERRQ(BVView(Y,view));
  }

  /* Check orthogonality */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&M));
  CHKERRQ(PetscObjectSetName((PetscObject)M,"M"));
  CHKERRQ(BVDot(X,Y,M));
  if (verbose) CHKERRQ(MatView(M,view));
  CHKERRQ(MatShift(M,-1.0));
  CHKERRQ(MatNorm(M,NORM_1,&norm));
  if (norm<200*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Level of bi-orthogonality < 200*eps\n"));
  else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Level of bi-orthogonality: %g\n",(double)norm));

  CHKERRQ(MatDestroy(&M));
  CHKERRQ(BVDestroy(&X));
  CHKERRQ(BVDestroy(&Y));
  CHKERRQ(VecDestroy(&t));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      output_file: output/test17_1.out
      test:
         suffix: 1
         args: -bv_type {{vecs contiguous svec mat}} -bv_orthog_type cgs
         requires: double
      test:
         suffix: 1_cuda
         args: -bv_type svec -vec_type cuda -bv_orthog_type cgs
         requires: cuda
      test:
         suffix: 2
         args: -bv_type {{vecs contiguous svec mat}} -bv_orthog_type mgs
      test:
         suffix: 2_cuda
         args: -bv_type svec -vec_type cuda -bv_orthog_type mgs
         requires: cuda

TEST*/
