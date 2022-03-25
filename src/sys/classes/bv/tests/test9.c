/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test BV matrix projection.\n\n";

#include <slepcbv.h>

int main(int argc,char **argv)
{
  Vec            t,v;
  Mat            B,G,H0,H1;
  BV             X,Y,Z;
  PetscInt       i,j,n=20,kx=6,lx=3,ky=5,ly=2,Istart,Iend,col[5];
  PetscScalar    alpha,value[] = { -1, 1, 1, 1, 1 };
  PetscViewer    view;
  PetscReal      norm;
  PetscBool      verbose;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-kx",&kx,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-lx",&lx,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ky",&ky,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ly",&ly,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test BV projection (n=%" PetscInt_FMT ").\n",n));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"X has %" PetscInt_FMT " active columns (%" PetscInt_FMT " leading columns).\n",kx,lx));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Y has %" PetscInt_FMT " active columns (%" PetscInt_FMT " leading columns).\n",ky,ly));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&view));
  if (verbose) CHKERRQ(PetscViewerPushFormat(view,PETSC_VIEWER_ASCII_MATLAB));

  /* Create non-symmetric matrix G (Toeplitz) */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&G));
  CHKERRQ(MatSetSizes(G,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(G));
  CHKERRQ(MatSetUp(G));
  CHKERRQ(PetscObjectSetName((PetscObject)G,"G"));

  CHKERRQ(MatGetOwnershipRange(G,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    col[0]=i-1; col[1]=i; col[2]=i+1; col[3]=i+2; col[4]=i+3;
    if (i==0) CHKERRQ(MatSetValues(G,1,&i,PetscMin(4,n-i),col+1,value+1,INSERT_VALUES));
    else CHKERRQ(MatSetValues(G,1,&i,PetscMin(5,n-i+1),col,value,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(G,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(G,MAT_FINAL_ASSEMBLY));
  if (verbose) CHKERRQ(MatView(G,view));

  /* Create symmetric matrix B (1-D Laplacian) */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(B));
  CHKERRQ(PetscObjectSetName((PetscObject)B,"B"));

  CHKERRQ(MatGetOwnershipRange(B,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) CHKERRQ(MatSetValue(B,i,i-1,-1.0,INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(B,i,i+1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(B,i,i,2.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCreateVecs(B,&t,NULL));
  if (verbose) CHKERRQ(MatView(B,view));

  /* Create BV object X */
  CHKERRQ(BVCreate(PETSC_COMM_WORLD,&X));
  CHKERRQ(PetscObjectSetName((PetscObject)X,"X"));
  CHKERRQ(BVSetSizesFromVec(X,t,kx+2));  /* two extra columns to test active columns */
  CHKERRQ(BVSetFromOptions(X));

  /* Fill X entries */
  for (j=0;j<kx+2;j++) {
    CHKERRQ(BVGetColumn(X,j,&v));
    CHKERRQ(VecSet(v,0.0));
    for (i=0;i<4;i++) {
      if (i+j<n) {
#if defined(PETSC_USE_COMPLEX)
        alpha = PetscCMPLX((PetscReal)(3*i+j-2),(PetscReal)(2*i));
#else
        alpha = (PetscReal)(3*i+j-2);
#endif
        CHKERRQ(VecSetValue(v,i+j,alpha,INSERT_VALUES));
      }
    }
    CHKERRQ(VecAssemblyBegin(v));
    CHKERRQ(VecAssemblyEnd(v));
    CHKERRQ(BVRestoreColumn(X,j,&v));
  }
  if (verbose) CHKERRQ(BVView(X,view));

  /* Duplicate BV object and store Z=G*X */
  CHKERRQ(BVDuplicate(X,&Z));
  CHKERRQ(PetscObjectSetName((PetscObject)Z,"Z"));
  CHKERRQ(BVSetActiveColumns(X,0,kx));
  CHKERRQ(BVSetActiveColumns(Z,0,kx));
  CHKERRQ(BVMatMult(X,G,Z));
  CHKERRQ(BVSetActiveColumns(X,lx,kx));
  CHKERRQ(BVSetActiveColumns(Z,lx,kx));

  /* Create BV object Y */
  CHKERRQ(BVCreate(PETSC_COMM_WORLD,&Y));
  CHKERRQ(PetscObjectSetName((PetscObject)Y,"Y"));
  CHKERRQ(BVSetSizesFromVec(Y,t,ky+1));
  CHKERRQ(BVSetFromOptions(Y));
  CHKERRQ(BVSetActiveColumns(Y,ly,ky));

  /* Fill Y entries */
  for (j=0;j<ky+1;j++) {
    CHKERRQ(BVGetColumn(Y,j,&v));
#if defined(PETSC_USE_COMPLEX)
    alpha = PetscCMPLX((PetscReal)(j+1)/4.0,-(PetscReal)j);
#else
    alpha = (PetscReal)(j+1)/4.0;
#endif
    CHKERRQ(VecSet(v,(PetscScalar)(j+1)/4.0));
    CHKERRQ(BVRestoreColumn(Y,j,&v));
  }
  if (verbose) CHKERRQ(BVView(Y,view));

  /* Test BVMatProject for non-symmetric matrix G */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,ky,kx,NULL,&H0));
  CHKERRQ(PetscObjectSetName((PetscObject)H0,"H0"));
  CHKERRQ(BVMatProject(X,G,Y,H0));
  if (verbose) CHKERRQ(MatView(H0,view));

  /* Test BVMatProject with previously stored G*X */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,ky,kx,NULL,&H1));
  CHKERRQ(PetscObjectSetName((PetscObject)H1,"H1"));
  CHKERRQ(BVMatProject(Z,NULL,Y,H1));
  if (verbose) CHKERRQ(MatView(H1,view));

  /* Check that H0 and H1 are equal */
  CHKERRQ(MatAXPY(H0,-1.0,H1,SAME_NONZERO_PATTERN));
  CHKERRQ(MatNorm(H0,NORM_1,&norm));
  if (norm<10*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"||H0-H1|| < 10*eps\n"));
  else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"||H0-H1||=%g\n",(double)norm));
  CHKERRQ(MatDestroy(&H0));
  CHKERRQ(MatDestroy(&H1));

  /* Test BVMatProject for symmetric matrix B with orthogonal projection */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,kx,kx,NULL,&H0));
  CHKERRQ(PetscObjectSetName((PetscObject)H0,"H0"));
  CHKERRQ(BVMatProject(X,B,X,H0));
  if (verbose) CHKERRQ(MatView(H0,view));

  /* Repeat previous test with symmetry flag set */
  CHKERRQ(MatSetOption(B,MAT_HERMITIAN,PETSC_TRUE));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,kx,kx,NULL,&H1));
  CHKERRQ(PetscObjectSetName((PetscObject)H1,"H1"));
  CHKERRQ(BVMatProject(X,B,X,H1));
  if (verbose) CHKERRQ(MatView(H1,view));

  /* Check that H0 and H1 are equal */
  CHKERRQ(MatAXPY(H0,-1.0,H1,SAME_NONZERO_PATTERN));
  CHKERRQ(MatNorm(H0,NORM_1,&norm));
  if (norm<10*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"||H0-H1|| < 10*eps\n"));
  else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"||H0-H1||=%g\n",(double)norm));
  CHKERRQ(MatDestroy(&H0));
  CHKERRQ(MatDestroy(&H1));

  CHKERRQ(BVDestroy(&X));
  CHKERRQ(BVDestroy(&Y));
  CHKERRQ(BVDestroy(&Z));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&G));
  CHKERRQ(VecDestroy(&t));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   testset:
      output_file: output/test9_1.out
      test:
         suffix: 1
         args: -bv_type {{vecs contiguous svec mat}shared output}
      test:
         suffix: 1_svec_vecs
         args: -bv_type svec -bv_matmult vecs
      test:
         suffix: 1_cuda
         args: -bv_type svec -mat_type aijcusparse
         requires: cuda
      test:
         suffix: 2
         nsize: 2
         args: -bv_type {{vecs contiguous svec mat}shared output}
      test:
         suffix: 2_svec_vecs
         nsize: 2
         args: -bv_type svec -bv_matmult vecs

TEST*/
