/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test multiplication of a Mat times a BV.\n\n";

#include <slepcbv.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  Vec            t,r,v;
  Mat            B,Ymat;
  BV             X,Y,Z=NULL,Zcopy=NULL;
  PetscInt       i,j,m=10,n,k=5,rep=1,Istart,Iend;
  PetscScalar    *pZ;
  PetscReal      norm;
  PetscViewer    view;
  PetscBool      flg,verbose,fromfile;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  BVMatMultType  vmm;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-rep",&rep,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-file",filename,sizeof(filename),&fromfile));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(PetscObjectSetName((PetscObject)B,"B"));
  if (fromfile) {
#if defined(PETSC_USE_COMPLEX)
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Reading COMPLEX matrix from a binary file...\n"));
#else
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Reading REAL matrix from a binary file...\n"));
#endif
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
    CHKERRQ(MatSetFromOptions(B));
    CHKERRQ(MatLoad(B,viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
    CHKERRQ(MatGetSize(B,&m,&n));
    CHKERRQ(MatGetOwnershipRange(B,&Istart,&Iend));
  } else {
    /* Create 1-D Laplacian matrix */
    CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
    CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,&flg));
    if (!flg) n = m;
    CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,m,n));
    CHKERRQ(MatSetFromOptions(B));
    CHKERRQ(MatSetUp(B));
    CHKERRQ(MatGetOwnershipRange(B,&Istart,&Iend));
    for (i=Istart;i<Iend;i++) {
      if (i>0 && i-1<n) CHKERRQ(MatSetValue(B,i,i-1,-1.0,INSERT_VALUES));
      if (i+1<n) CHKERRQ(MatSetValue(B,i,i+1,-1.0,INSERT_VALUES));
      if (i<n) CHKERRQ(MatSetValue(B,i,i,2.0,INSERT_VALUES));
    }
    CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  }

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test BVMatMult (m=%" PetscInt_FMT ", n=%" PetscInt_FMT ", k=%" PetscInt_FMT ").\n",m,n,k));
  CHKERRQ(MatCreateVecs(B,&t,&r));

  /* Create BV object X */
  CHKERRQ(BVCreate(PETSC_COMM_WORLD,&X));
  CHKERRQ(PetscObjectSetName((PetscObject)X,"X"));
  CHKERRQ(BVSetSizesFromVec(X,t,k));
  CHKERRQ(BVSetMatMultMethod(X,BV_MATMULT_VECS));
  CHKERRQ(BVSetFromOptions(X));
  CHKERRQ(BVGetMatMultMethod(X,&vmm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Using method: %s\n",BVMatMultTypes[vmm]));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&view));
  if (verbose) {
    CHKERRQ(PetscViewerPushFormat(view,PETSC_VIEWER_ASCII_MATLAB));
  }

  /* Fill X entries */
  for (j=0;j<k;j++) {
    CHKERRQ(BVGetColumn(X,j,&v));
    CHKERRQ(VecSet(v,0.0));
    for (i=Istart;i<PetscMin(j+1,Iend);i++) {
      CHKERRQ(VecSetValue(v,i,1.0,INSERT_VALUES));
    }
    CHKERRQ(VecAssemblyBegin(v));
    CHKERRQ(VecAssemblyEnd(v));
    CHKERRQ(BVRestoreColumn(X,j,&v));
  }
  if (verbose) {
    CHKERRQ(MatView(B,view));
    CHKERRQ(BVView(X,view));
  }

  /* Create BV object Y */
  CHKERRQ(BVCreate(PETSC_COMM_WORLD,&Y));
  CHKERRQ(PetscObjectSetName((PetscObject)Y,"Y"));
  CHKERRQ(BVSetSizesFromVec(Y,r,k+4));
  CHKERRQ(BVSetMatMultMethod(Y,BV_MATMULT_VECS));
  CHKERRQ(BVSetFromOptions(Y));
  CHKERRQ(BVSetActiveColumns(Y,2,k+2));

  /* Test BVMatMult */
  for (i=0;i<rep;i++) {
    CHKERRQ(BVMatMult(X,B,Y));
  }
  if (verbose) {
    CHKERRQ(BVView(Y,view));
  }

  if (fromfile) {
    /* Test BVMatMultTranspose */
    CHKERRQ(BVDuplicate(X,&Z));
    CHKERRQ(BVSetRandom(Z));
    for (i=0;i<rep;i++) {
      CHKERRQ(BVMatMultTranspose(Z,B,Y));
    }
    if (verbose) {
      CHKERRQ(BVView(Z,view));
      CHKERRQ(BVView(Y,view));
    }
    CHKERRQ(BVDestroy(&Z));
    CHKERRQ(BVMatMultTransposeColumn(Y,B,2));
    if (verbose) {
      CHKERRQ(BVView(Y,view));
    }
  }

  /* Test BVGetMat/RestoreMat */
  CHKERRQ(BVGetMat(Y,&Ymat));
  CHKERRQ(PetscObjectSetName((PetscObject)Ymat,"Ymat"));
  if (verbose) {
    CHKERRQ(MatView(Ymat,view));
  }
  CHKERRQ(BVRestoreMat(Y,&Ymat));

  if (!fromfile) {
    /* Create BV object Z */
    CHKERRQ(BVDuplicateResize(Y,k,&Z));
    CHKERRQ(PetscObjectSetName((PetscObject)Z,"Z"));

    /* Fill Z entries */
    for (j=0;j<k;j++) {
      CHKERRQ(BVGetColumn(Z,j,&v));
      CHKERRQ(VecSet(v,0.0));
      if (!Istart) CHKERRQ(VecSetValue(v,0,1.0,ADD_VALUES));
      if (j<n && j>=Istart && j<Iend) CHKERRQ(VecSetValue(v,j,1.0,ADD_VALUES));
      if (j+1<n && j>=Istart && j<Iend) CHKERRQ(VecSetValue(v,j+1,-1.0,ADD_VALUES));
      CHKERRQ(VecAssemblyBegin(v));
      CHKERRQ(VecAssemblyEnd(v));
      CHKERRQ(BVRestoreColumn(Z,j,&v));
    }
    if (verbose) {
      CHKERRQ(BVView(Z,view));
    }

    /* Save a copy of Z */
    CHKERRQ(BVDuplicate(Z,&Zcopy));
    CHKERRQ(BVCopy(Z,Zcopy));

    /* Test BVMult, check result of previous operations */
    CHKERRQ(BVMult(Z,-1.0,1.0,Y,NULL));
    CHKERRQ(BVNorm(Z,NORM_FROBENIUS,&norm));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of error: %g\n",(double)norm));
  }

  /* Test BVMatMultColumn, multiply Y(:,2), result in Y(:,3) */
  if (m==n) {
    CHKERRQ(BVMatMultColumn(Y,B,2));
    if (verbose) {
      CHKERRQ(BVView(Y,view));
    }

    if (!fromfile) {
      /* Test BVGetArray, modify Z to match Y */
      CHKERRQ(BVCopy(Zcopy,Z));
      CHKERRQ(BVGetArray(Z,&pZ));
      if (Istart==0) {
        PetscCheck(Iend>2,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"First process must have at least 3 rows");
        pZ[Iend]   = 5.0;   /* modify 3 first entries of second column */
        pZ[Iend+1] = -4.0;
        pZ[Iend+2] = 1.0;
      }
      CHKERRQ(BVRestoreArray(Z,&pZ));
      if (verbose) {
        CHKERRQ(BVView(Z,view));
      }

      /* Check result again with BVMult */
      CHKERRQ(BVMult(Z,-1.0,1.0,Y,NULL));
      CHKERRQ(BVNorm(Z,NORM_FROBENIUS,&norm));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of error: %g\n",(double)norm));
    }
  }

  CHKERRQ(BVDestroy(&Z));
  CHKERRQ(BVDestroy(&Zcopy));
  CHKERRQ(BVDestroy(&X));
  CHKERRQ(BVDestroy(&Y));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(VecDestroy(&t));
  CHKERRQ(VecDestroy(&r));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      output_file: output/test7_1.out
      filter: grep -v "Using method"
      test:
         suffix: 1
         args: -bv_type {{vecs contiguous svec mat}shared output} -bv_matmult vecs
      test:
         suffix: 1_cuda
         args: -bv_type svec -mat_type aijcusparse -bv_matmult vecs
         requires: cuda
      test:
         suffix: 1_mat
         args: -bv_type {{vecs contiguous svec mat}shared output} -bv_matmult mat

   testset:
      output_file: output/test7_2.out
      filter: grep -v "Using method"
      args: -m 34 -n 38 -k 9
      nsize: 2
      test:
         suffix: 2
         args: -bv_type {{vecs contiguous svec mat}shared output} -bv_matmult vecs
      test:
         suffix: 2_cuda
         args: -bv_type svec -mat_type aijcusparse -bv_matmult vecs
         requires: cuda
      test:
         suffix: 2_mat
         args: -bv_type {{vecs contiguous svec mat}shared output} -bv_matmult mat

   testset:
      output_file: output/test7_3.out
      filter: grep -v "Using method"
      args: -file ${SLEPC_DIR}/share/slepc/datafiles/matrices/bfw62a.petsc -bv_reproducible_random
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES)
      nsize: 2
      test:
         suffix: 3
         args: -bv_type {{vecs contiguous svec mat}shared output} -bv_matmult {{vecs mat}}

TEST*/
