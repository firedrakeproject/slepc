/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test BV orthogonalization functions.\n\n";

#include <slepcbv.h>

int main(int argc,char **argv)
{
  BV             X,Y,Z;
  Mat            M,R;
  Vec            v,t,e;
  PetscInt       i,j,n=20,k=8;
  PetscViewer    view;
  PetscBool      verbose;
  PetscReal      norm,condn=1.0;
  PetscScalar    alpha;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-condn",&condn,NULL));
  PetscCheck(condn>=1.0,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"The condition number must be > 1");
  PetscCall(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test BV orthogonalization with %" PetscInt_FMT " columns of length %" PetscInt_FMT ".\n",k,n));
  if (condn>1.0) PetscCall(PetscPrintf(PETSC_COMM_WORLD," - Using a random BV with condition number = %g\n",(double)condn));

  /* Create template vector */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&t));
  PetscCall(VecSetSizes(t,PETSC_DECIDE,n));
  PetscCall(VecSetFromOptions(t));

  /* Create BV object X */
  PetscCall(BVCreate(PETSC_COMM_WORLD,&X));
  PetscCall(PetscObjectSetName((PetscObject)X,"X"));
  PetscCall(BVSetSizesFromVec(X,t,k));
  PetscCall(BVSetFromOptions(X));

  /* Set up viewer */
  PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&view));
  if (verbose) PetscCall(PetscViewerPushFormat(view,PETSC_VIEWER_ASCII_MATLAB));

  /* Fill X entries */
  if (condn==1.0) {
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
  } else PetscCall(BVSetRandomCond(X,condn));
  if (verbose) PetscCall(BVView(X,view));

  /* Create copies on Y and Z */
  PetscCall(BVDuplicate(X,&Y));
  PetscCall(PetscObjectSetName((PetscObject)Y,"Y"));
  PetscCall(BVCopy(X,Y));
  PetscCall(BVDuplicate(X,&Z));
  PetscCall(PetscObjectSetName((PetscObject)Z,"Z"));
  PetscCall(BVCopy(X,Z));

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

  /* Test BVOrthogonalize */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&R));
  PetscCall(PetscObjectSetName((PetscObject)R,"R"));
  PetscCall(BVOrthogonalize(Y,R));
  if (verbose) {
    PetscCall(BVView(Y,view));
    PetscCall(MatView(R,view));
  }

  /* Check orthogonality */
  PetscCall(BVDot(Y,Y,M));
  PetscCall(MatShift(M,-1.0));
  PetscCall(MatNorm(M,NORM_1,&norm));
  if (norm<100*PETSC_MACHINE_EPSILON) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality < 100*eps\n"));
  else PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %g\n",(double)norm));

  /* Check residual */
  PetscCall(BVMult(Z,-1.0,1.0,Y,R));
  PetscCall(BVNorm(Z,NORM_FROBENIUS,&norm));
  if (norm<100*PETSC_MACHINE_EPSILON) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Residual ||X-QR|| < 100*eps\n"));
  else PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Residual ||X-QR||: %g\n",(double)norm));

  /* Test BVOrthogonalizeVec */
  PetscCall(VecDuplicate(t,&e));
  PetscCall(VecSet(e,1.0));
  PetscCall(BVOrthogonalizeVec(X,e,NULL,&norm,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm of ones(n,1) after orthogonalizing against X: %g\n",(double)norm));

  PetscCall(MatDestroy(&M));
  PetscCall(MatDestroy(&R));
  PetscCall(BVDestroy(&X));
  PetscCall(BVDestroy(&Y));
  PetscCall(BVDestroy(&Z));
  PetscCall(VecDestroy(&e));
  PetscCall(VecDestroy(&t));
  PetscCall(SlepcFinalize());
  return 0;
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
