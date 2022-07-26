/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test BV created from a dense Mat.\n\n";

#include <slepcbv.h>

int main(int argc,char **argv)
{
  BV             X;
  Mat            A,B,M;
  PetscInt       i,j,n=20,k=8,Istart,Iend;
  PetscViewer    view;
  PetscBool      verbose;
  PetscReal      norm;
  PetscScalar    alpha;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test BV created from a dense Mat (length %" PetscInt_FMT ", k=%" PetscInt_FMT ").\n",n,k));

  /* Create dense matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,k));
  PetscCall(MatSetType(A,MATDENSE));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (j=0;j<k;j++) {
    for (i=0;i<=n/2;i++) {
      if (i+j<n && i>=Istart && i<Iend) {
        alpha = (3.0*i+j-2)/(2*(i+j+1));
        PetscCall(MatSetValue(A,i+j,j,alpha,INSERT_VALUES));
      }
    }
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Create BV object X */
  PetscCall(BVCreateFromMat(A,&X));
  PetscCall(BVSetFromOptions(X));
  PetscCall(PetscObjectSetName((PetscObject)X,"X"));

  /* Set up viewer */
  PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&view));
  if (verbose) {
    PetscCall(PetscViewerPushFormat(view,PETSC_VIEWER_ASCII_MATLAB));
    PetscCall(BVView(X,view));
  }

  /* Test BVCreateMat */
  PetscCall(BVCreateMat(X,&B));
  PetscCall(MatAXPY(B,-1.0,A,SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(B,NORM_1,&norm));
  if (norm<100*PETSC_MACHINE_EPSILON) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm of difference < 100*eps\n"));
  else PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm of difference: %g\n",(double)norm));

  /* Test BVOrthogonalize */
  PetscCall(BVOrthogonalize(X,NULL));
  if (verbose) PetscCall(BVView(X,view));

  /* Check orthogonality */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&M));
  PetscCall(MatShift(M,1.0));   /* set leading part to identity */
  PetscCall(BVDot(X,X,M));
  PetscCall(MatShift(M,-1.0));
  PetscCall(MatNorm(M,NORM_1,&norm));
  if (norm<100*PETSC_MACHINE_EPSILON) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality < 100*eps\n"));
  else PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %g\n",(double)norm));

  PetscCall(MatDestroy(&M));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(BVDestroy(&X));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 2
      args: -bv_type {{vecs contiguous svec mat}shared output}
      output_file: output/test14_1.out

TEST*/
