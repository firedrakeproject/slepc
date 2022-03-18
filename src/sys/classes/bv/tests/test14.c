/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test BV created from a dense Mat.\n\n";

#include <slepcbv.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  BV             X;
  Mat            A,B,M;
  PetscInt       i,j,n=20,k=8,Istart,Iend;
  PetscViewer    view;
  PetscBool      verbose;
  PetscReal      norm;
  PetscScalar    alpha;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test BV created from a dense Mat (length %" PetscInt_FMT ", k=%" PetscInt_FMT ").\n",n,k));

  /* Create dense matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,k));
  CHKERRQ(MatSetType(A,MATDENSE));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (j=0;j<k;j++) {
    for (i=0;i<=n/2;i++) {
      if (i+j<n && i>=Istart && i<Iend) {
        alpha = (3.0*i+j-2)/(2*(i+j+1));
        CHKERRQ(MatSetValue(A,i+j,j,alpha,INSERT_VALUES));
      }
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Create BV object X */
  CHKERRQ(BVCreateFromMat(A,&X));
  CHKERRQ(BVSetFromOptions(X));
  CHKERRQ(PetscObjectSetName((PetscObject)X,"X"));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&view));
  if (verbose) {
    CHKERRQ(PetscViewerPushFormat(view,PETSC_VIEWER_ASCII_MATLAB));
    CHKERRQ(BVView(X,view));
  }

  /* Test BVCreateMat */
  CHKERRQ(BVCreateMat(X,&B));
  CHKERRQ(MatAXPY(B,-1.0,A,SAME_NONZERO_PATTERN));
  CHKERRQ(MatNorm(B,NORM_1,&norm));
  if (norm<100*PETSC_MACHINE_EPSILON) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of difference < 100*eps\n"));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of difference: %g\n",(double)norm));
  }

  /* Test BVOrthogonalize */
  CHKERRQ(BVOrthogonalize(X,NULL));
  if (verbose) {
    CHKERRQ(BVView(X,view));
  }

  /* Check orthogonality */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&M));
  CHKERRQ(MatShift(M,1.0));   /* set leading part to identity */
  CHKERRQ(BVDot(X,X,M));
  CHKERRQ(MatShift(M,-1.0));
  CHKERRQ(MatNorm(M,NORM_1,&norm));
  if (norm<100*PETSC_MACHINE_EPSILON) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality < 100*eps\n"));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %g\n",(double)norm));
  }

  CHKERRQ(MatDestroy(&M));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(BVDestroy(&X));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      nsize: 2
      args: -bv_type {{vecs contiguous svec mat}shared output}
      output_file: output/test14_1.out

TEST*/
