/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test split reductions in BV.\n\n";

#include <slepcbv.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  Vec            t,v,w,y,z,zsplit;
  BV             X;
  PetscInt       i,j,n=10,k=5;
  PetscScalar    *zarray;
  PetscReal      nrm;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  PetscCheck(k>2,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Should specify at least k=3 columns");
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"BV split ops (%" PetscInt_FMT " columns of dimension %" PetscInt_FMT ").\n",k,n));

  /* Create template vector */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&t));
  CHKERRQ(VecSetSizes(t,PETSC_DECIDE,n));
  CHKERRQ(VecSetFromOptions(t));
  CHKERRQ(VecDuplicate(t,&v));
  CHKERRQ(VecSet(v,1.0));

  /* Create BV object X */
  CHKERRQ(BVCreate(PETSC_COMM_WORLD,&X));
  CHKERRQ(PetscObjectSetName((PetscObject)X,"X"));
  CHKERRQ(BVSetSizesFromVec(X,t,k));
  CHKERRQ(BVSetFromOptions(X));

  /* Fill X entries */
  for (j=0;j<k;j++) {
    CHKERRQ(BVGetColumn(X,j,&w));
    CHKERRQ(VecSet(w,0.0));
    for (i=0;i<4;i++) {
      if (i+j<n) {
        CHKERRQ(VecSetValue(w,i+j,(PetscScalar)(3*i+j-2),INSERT_VALUES));
      }
    }
    CHKERRQ(VecAssemblyBegin(w));
    CHKERRQ(VecAssemblyEnd(w));
    CHKERRQ(BVRestoreColumn(X,j,&w));
  }

  /* Use regular operations */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,k+6,&z));
  CHKERRQ(VecGetArray(z,&zarray));
  CHKERRQ(BVGetColumn(X,0,&w));
  CHKERRQ(VecDot(w,v,zarray));
  CHKERRQ(BVRestoreColumn(X,0,&w));
  CHKERRQ(BVDotVec(X,v,zarray+1));
  CHKERRQ(BVDotColumn(X,2,zarray+1+k));

  CHKERRQ(BVGetColumn(X,1,&y));
  CHKERRQ(VecNorm(y,NORM_2,&nrm));
  CHKERRQ(BVRestoreColumn(X,1,&y));
  zarray[k+3] = nrm;
  CHKERRQ(BVNormVec(X,v,NORM_2,&nrm));
  zarray[k+4] = nrm;
  CHKERRQ(BVNormColumn(X,0,NORM_2,&nrm));
  zarray[k+5] = nrm;
  CHKERRQ(VecRestoreArray(z,&zarray));

  /* Use split operations */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,k+6,&zsplit));
  CHKERRQ(VecGetArray(zsplit,&zarray));
  CHKERRQ(BVGetColumn(X,0,&w));
  CHKERRQ(VecDotBegin(w,v,zarray));
  CHKERRQ(BVDotVecBegin(X,v,zarray+1));
  CHKERRQ(BVDotColumnBegin(X,2,zarray+1+k));
  CHKERRQ(VecDotEnd(w,v,zarray));
  CHKERRQ(BVRestoreColumn(X,0,&w));
  CHKERRQ(BVDotVecEnd(X,v,zarray+1));
  CHKERRQ(BVDotColumnEnd(X,2,zarray+1+k));

  CHKERRQ(BVGetColumn(X,1,&y));
  CHKERRQ(VecNormBegin(y,NORM_2,&nrm));
  CHKERRQ(BVNormVecBegin(X,v,NORM_2,&nrm));
  CHKERRQ(BVNormColumnBegin(X,0,NORM_2,&nrm));
  CHKERRQ(VecNormEnd(y,NORM_2,&nrm));
  CHKERRQ(BVRestoreColumn(X,1,&y));
  zarray[k+3] = nrm;
  CHKERRQ(BVNormVecEnd(X,v,NORM_2,&nrm));
  zarray[k+4] = nrm;
  CHKERRQ(BVNormColumnEnd(X,0,NORM_2,&nrm));
  zarray[k+5] = nrm;
  CHKERRQ(VecRestoreArray(zsplit,&zarray));

  /* Show difference */
  CHKERRQ(VecAXPY(z,-1.0,zsplit));
  CHKERRQ(VecNorm(z,NORM_1,&nrm));
  CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%g\n",(double)nrm));
  CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,NULL));

  CHKERRQ(BVDestroy(&X));
  CHKERRQ(VecDestroy(&t));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&z));
  CHKERRQ(VecDestroy(&zsplit));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      nsize: 2
      output_file: output/test10_1.out
      test:
         suffix: 1
         args: -bv_type {{vecs contiguous svec mat}shared output}
      test:
         suffix: 1_cuda
         args: -bv_type svec -vec_type cuda
         requires: cuda

TEST*/
