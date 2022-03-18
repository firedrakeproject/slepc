/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test BVGetSplit().\n\n";

#include <slepcbv.h>

/*
   Print the first row of a BV
 */
PetscErrorCode PrintFirstRow(BV X)
{
  PetscMPIInt       rank;
  PetscInt          i,nloc,k,nc;
  const PetscScalar *pX;
  const char        *name;

  PetscFunctionBeginUser;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)X),&rank));
  if (!rank) {
    CHKERRQ(BVGetActiveColumns(X,NULL,&k));
    CHKERRQ(BVGetSizes(X,&nloc,NULL,NULL));
    CHKERRQ(BVGetNumConstraints(X,&nc));
    CHKERRQ(PetscObjectGetName((PetscObject)X,&name));
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)X),"First row of %s =\n",name));
    CHKERRQ(BVGetArrayRead(X,&pX));
    for (i=0;i<nc+k;i++) {
      CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)X),"%g ",(double)PetscRealPart(pX[i*nloc])));
    }
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)X),"\n"));
    CHKERRQ(BVRestoreArrayRead(X,&pX));
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  Vec            t,v,*C;
  BV             X,L,R;
  PetscInt       i,j,n=10,k=5,l=3,nc=0,nloc;
  PetscReal      norm;
  PetscScalar    alpha;
  PetscViewer    view;
  PetscBool      verbose;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-l",&l,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nc",&nc,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test BVGetSplit (length %" PetscInt_FMT ", l=%" PetscInt_FMT ", k=%" PetscInt_FMT ", nc=%" PetscInt_FMT ").\n",n,l,k,nc));

  /* Create template vector */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&t));
  CHKERRQ(VecSetSizes(t,PETSC_DECIDE,n));
  CHKERRQ(VecSetFromOptions(t));
  CHKERRQ(VecGetLocalSize(t,&nloc));

  /* Create BV object X */
  CHKERRQ(BVCreate(PETSC_COMM_WORLD,&X));
  CHKERRQ(PetscObjectSetName((PetscObject)X,"X"));
  CHKERRQ(BVSetSizesFromVec(X,t,k));
  CHKERRQ(BVSetFromOptions(X));

  /* Generate constraints and attach them to X */
  if (nc>0) {
    CHKERRQ(VecDuplicateVecs(t,nc,&C));
    for (j=0;j<nc;j++) {
      for (i=0;i<=j;i++) {
        CHKERRQ(VecSetValue(C[j],nc-i+1,1.0,INSERT_VALUES));
      }
      CHKERRQ(VecAssemblyBegin(C[j]));
      CHKERRQ(VecAssemblyEnd(C[j]));
    }
    CHKERRQ(BVInsertConstraints(X,&nc,C));
    CHKERRQ(VecDestroyVecs(nc,&C));
  }

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&view));
  if (verbose) {
    CHKERRQ(PetscViewerPushFormat(view,PETSC_VIEWER_ASCII_MATLAB));
  }

  /* Fill X entries */
  for (j=0;j<k;j++) {
    CHKERRQ(BVGetColumn(X,j,&v));
    CHKERRQ(VecSet(v,0.0));
    for (i=0;i<4;i++) {
      if (i+j<n) {
        CHKERRQ(VecSetValue(v,i+j,(PetscScalar)(3*i+j-2),INSERT_VALUES));
      }
    }
    CHKERRQ(VecAssemblyBegin(v));
    CHKERRQ(VecAssemblyEnd(v));
    CHKERRQ(BVRestoreColumn(X,j,&v));
  }
  if (verbose) {
    CHKERRQ(BVView(X,view));
  }

  /* Get split BVs */
  CHKERRQ(BVSetActiveColumns(X,l,k));
  CHKERRQ(BVGetSplit(X,&L,&R));
  CHKERRQ(PetscObjectSetName((PetscObject)L,"L"));
  CHKERRQ(PetscObjectSetName((PetscObject)R,"R"));

  if (verbose) {
    CHKERRQ(BVView(L,view));
    CHKERRQ(BVView(R,view));
  }

  /* Modify first column of R */
  CHKERRQ(BVGetColumn(R,0,&v));
  CHKERRQ(VecSet(v,-1.0));
  CHKERRQ(BVRestoreColumn(R,0,&v));

  /* Finished using the split BVs */
  CHKERRQ(BVRestoreSplit(X,&L,&R));
  CHKERRQ(PrintFirstRow(X));
  if (verbose) {
    CHKERRQ(BVView(X,view));
  }

  /* Get the left split BV only */
  CHKERRQ(BVGetSplit(X,&L,NULL));
  for (j=0;j<l;j++) {
    CHKERRQ(BVOrthogonalizeColumn(L,j,NULL,&norm,NULL));
    alpha = 1.0/norm;
    CHKERRQ(BVScaleColumn(L,j,alpha));
  }
  CHKERRQ(BVRestoreSplit(X,&L,NULL));
  CHKERRQ(PrintFirstRow(X));
  if (verbose) {
    CHKERRQ(BVView(X,view));
  }

  /* Now get the right split BV after changing the number of leading columns */
  CHKERRQ(BVSetActiveColumns(X,l-1,k));
  CHKERRQ(BVGetSplit(X,NULL,&R));
  CHKERRQ(BVGetColumn(R,0,&v));
  CHKERRQ(BVInsertVec(X,0,v));
  CHKERRQ(BVRestoreColumn(R,0,&v));
  CHKERRQ(BVRestoreSplit(X,NULL,&R));
  CHKERRQ(PrintFirstRow(X));
  if (verbose) {
    CHKERRQ(BVView(X,view));
  }

  CHKERRQ(BVDestroy(&X));
  CHKERRQ(VecDestroy(&t));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      nsize: 2
      output_file: output/test15_1.out
      test:
         suffix: 1
         args: -bv_type {{vecs contiguous svec mat}shared output}

      test:
         suffix: 1_cuda
         args: -bv_type svec -vec_type cuda
         requires: cuda

   testset:
      nsize: 2
      output_file: output/test15_2.out
      test:
         suffix: 2
         args: -nc 2 -bv_type {{vecs contiguous svec mat}shared output}
      test:
         suffix: 2_cuda
         args: -nc 2 -bv_type svec -vec_type cuda
         requires: cuda

TEST*/
