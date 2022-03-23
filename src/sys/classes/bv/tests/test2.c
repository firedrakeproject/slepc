/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test BV orthogonalization functions.\n\n";

#include <slepcbv.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  BV             X,Y,Z;
  Mat            M,R;
  Vec            v,t,e;
  PetscInt       i,j,n=20,k=8;
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
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test BV orthogonalization with %" PetscInt_FMT " columns of length %" PetscInt_FMT ".\n",k,n));
  if (condn>1.0) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," - Using a random BV with condition number = %g\n",(double)condn));

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
          CHKERRQ(VecSetValue(v,i+j,alpha,INSERT_VALUES));
        }
      }
      CHKERRQ(VecAssemblyBegin(v));
      CHKERRQ(VecAssemblyEnd(v));
      CHKERRQ(BVRestoreColumn(X,j,&v));
    }
  } else CHKERRQ(BVSetRandomCond(X,condn));
  if (verbose) CHKERRQ(BVView(X,view));

  /* Create copies on Y and Z */
  CHKERRQ(BVDuplicate(X,&Y));
  CHKERRQ(PetscObjectSetName((PetscObject)Y,"Y"));
  CHKERRQ(BVCopy(X,Y));
  CHKERRQ(BVDuplicate(X,&Z));
  CHKERRQ(PetscObjectSetName((PetscObject)Z,"Z"));
  CHKERRQ(BVCopy(X,Z));

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

  /* Test BVOrthogonalize */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&R));
  CHKERRQ(PetscObjectSetName((PetscObject)R,"R"));
  CHKERRQ(BVOrthogonalize(Y,R));
  if (verbose) {
    CHKERRQ(BVView(Y,view));
    CHKERRQ(MatView(R,view));
  }

  /* Check orthogonality */
  CHKERRQ(BVDot(Y,Y,M));
  CHKERRQ(MatShift(M,-1.0));
  CHKERRQ(MatNorm(M,NORM_1,&norm));
  if (norm<100*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality < 100*eps\n"));
  else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %g\n",(double)norm));

  /* Check residual */
  CHKERRQ(BVMult(Z,-1.0,1.0,Y,R));
  CHKERRQ(BVNorm(Z,NORM_FROBENIUS,&norm));
  if (norm<100*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Residual ||X-QR|| < 100*eps\n"));
  else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Residual ||X-QR||: %g\n",(double)norm));

  /* Test BVOrthogonalizeVec */
  CHKERRQ(VecDuplicate(t,&e));
  CHKERRQ(VecSet(e,1.0));
  CHKERRQ(BVOrthogonalizeVec(X,e,NULL,&norm,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of ones(n,1) after orthogonalizing against X: %g\n",(double)norm));

  CHKERRQ(MatDestroy(&M));
  CHKERRQ(MatDestroy(&R));
  CHKERRQ(BVDestroy(&X));
  CHKERRQ(BVDestroy(&Y));
  CHKERRQ(BVDestroy(&Z));
  CHKERRQ(VecDestroy(&e));
  CHKERRQ(VecDestroy(&t));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      output_file: output/test2_1.out
      test:
         suffix: 1
         args: -bv_type {{vecs contiguous svec mat}shared output} -bv_orthog_type cgs
      test:
         suffix: 1_cuda
         args: -bv_type svec -vec_type cuda -bv_orthog_type cgs
         requires: cuda
      test:
         suffix: 2
         args: -bv_type {{vecs contiguous svec mat}shared output} -bv_orthog_type mgs
      test:
         suffix: 2_cuda
         args: -bv_type svec -vec_type cuda -bv_orthog_type mgs
         requires: cuda

   test:
      suffix: 3
      nsize: 1
      args: -bv_type {{vecs contiguous svec mat}shared output} -condn 1e8
      requires: !single
      filter: grep -v "against"
      output_file: output/test2_3.out

TEST*/
