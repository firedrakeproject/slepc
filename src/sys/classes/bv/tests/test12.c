/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test block orthogonalization on a rank-deficient BV.\n\n";

#include <slepcbv.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  BV             X,Z;
  Mat            M,R;
  Vec            v,w,t;
  PetscInt       i,j,n=20,k=8;
  PetscViewer    view;
  PetscBool      verbose;
  PetscReal      norm;
  PetscScalar    alpha;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test BV block orthogonalization (length %" PetscInt_FMT ", k=%" PetscInt_FMT ").\n",n,k));
  PetscCheck(k>5,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"k must be at least 6");

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

  /* Fill X entries (first half) */
  for (j=0;j<k/2;j++) {
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

  /* make middle column linearly dependent wrt columns 0 and 1 */
  CHKERRQ(BVCopyColumn(X,0,j));
  CHKERRQ(BVGetColumn(X,j,&v));
  CHKERRQ(BVGetColumn(X,1,&w));
  CHKERRQ(VecAXPY(v,0.5,w));
  CHKERRQ(BVRestoreColumn(X,1,&w));
  CHKERRQ(BVRestoreColumn(X,j,&v));
  j++;

  /* Fill X entries (second half) */
  for (;j<k-1;j++) {
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

  /* make middle column linearly dependent wrt columns 1 and k/2+1 */
  CHKERRQ(BVCopyColumn(X,1,j));
  CHKERRQ(BVGetColumn(X,j,&v));
  CHKERRQ(BVGetColumn(X,k/2+1,&w));
  CHKERRQ(VecAXPY(v,-1.2,w));
  CHKERRQ(BVRestoreColumn(X,k/2+1,&w));
  CHKERRQ(BVRestoreColumn(X,j,&v));

  if (verbose) CHKERRQ(BVView(X,view));

  /* Create a copy on Z */
  CHKERRQ(BVDuplicate(X,&Z));
  CHKERRQ(PetscObjectSetName((PetscObject)Z,"Z"));
  CHKERRQ(BVCopy(X,Z));

  /* Test BVOrthogonalize */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&R));
  CHKERRQ(PetscObjectSetName((PetscObject)R,"R"));
  CHKERRQ(BVOrthogonalize(X,R));
  if (verbose) {
    CHKERRQ(BVView(X,view));
    CHKERRQ(MatView(R,view));
  }

  /* Check orthogonality */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&M));
  CHKERRQ(MatShift(M,1.0));   /* set leading part to identity */
  CHKERRQ(BVDot(X,X,M));
  CHKERRQ(MatShift(M,-1.0));
  CHKERRQ(MatNorm(M,NORM_1,&norm));
  if (norm<100*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality < 100*eps\n"));
  else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %g\n",(double)norm));

  /* Check residual */
  CHKERRQ(BVMult(Z,-1.0,1.0,X,R));
  CHKERRQ(BVNorm(Z,NORM_FROBENIUS,&norm));
  if (norm<100*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Residual ||X-QR|| < 100*eps\n"));
  else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Residual ||X-QR||: %g\n",(double)norm));

  CHKERRQ(MatDestroy(&R));
  CHKERRQ(MatDestroy(&M));
  CHKERRQ(BVDestroy(&X));
  CHKERRQ(BVDestroy(&Z));
  CHKERRQ(VecDestroy(&t));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      nsize: 1
      args: -bv_orthog_block gs -bv_type {{vecs contiguous svec mat}shared output}
      output_file: output/test12_1.out

TEST*/
