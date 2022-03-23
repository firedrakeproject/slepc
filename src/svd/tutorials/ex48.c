/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Solves a GSVD problem with matrices loaded from a file.\n"
  "The command line options are:\n"
  "  -f1 <filename>, PETSc binary file containing matrix A.\n"
  "  -f2 <filename>, PETSc binary file containing matrix B (optional).\n\n";

#include <slepcsvd.h>

int main(int argc,char **argv)
{
  Mat            A,B;             /* matrices */
  SVD            svd;             /* singular value problem solver context */
  PetscInt       i,m,n,Istart,Iend,col[2];
  PetscScalar    vals[] = { -1, 1 };
  char           filename[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscBool      flg,terse;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Load matrices that define the generalized singular value problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nGeneralized singular value problem stored in file.\n\n"));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f1",filename,sizeof(filename),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate a file name for matrix A with the -f1 option");

#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Reading COMPLEX matrix from a binary file...\n"));
#else
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Reading REAL matrix from a binary file...\n"));
#endif
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatLoad(A,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(MatGetSize(A,&m,&n));

  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f2",filename,sizeof(filename),&flg));
  if (flg) {
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
    CHKERRQ(MatSetFromOptions(B));
    CHKERRQ(MatLoad(B,viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Matrix B was not provided, setting B=bidiag(1,-1)\n\n"));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
    CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n+1,n));
    CHKERRQ(MatSetFromOptions(B));
    CHKERRQ(MatSetUp(B));
    CHKERRQ(MatGetOwnershipRange(B,&Istart,&Iend));
    for (i=Istart;i<Iend;i++) {
      col[0]=i-1; col[1]=i;
      if (i==0) CHKERRQ(MatSetValue(B,i,col[1],vals[1],INSERT_VALUES));
      else if (i<n) CHKERRQ(MatSetValues(B,1,&i,2,col,vals,INSERT_VALUES));
      else if (i==n) CHKERRQ(MatSetValue(B,i,col[0],vals[0],INSERT_VALUES));
    }
    CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Create the singular value solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create singular value solver context
  */
  CHKERRQ(SVDCreate(PETSC_COMM_WORLD,&svd));

  /*
     Set operators of GSVD problem
  */
  CHKERRQ(SVDSetOperators(svd,A,B));
  CHKERRQ(SVDSetProblemType(svd,SVD_GENERALIZED));

  /*
     Set solver parameters at runtime
  */
  CHKERRQ(SVDSetFromOptions(svd));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the problem and print solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(SVDSolve(svd));

  /* show detailed info unless -terse option is given by user */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) CHKERRQ(SVDErrorView(svd,SVD_ERROR_NORM,NULL));
  else {
    CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    CHKERRQ(SVDConvergedReasonView(svd,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(SVDErrorView(svd,SVD_ERROR_NORM,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(SVDDestroy(&svd));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  ierr = SlepcFinalize();
  return ierr;
}
/*TEST

   testset:
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -f1 ${SLEPC_DIR}/share/slepc/datafiles/matrices/bfw62a.petsc -f2 ${SLEPC_DIR}/share/slepc/datafiles/matrices/bfw62b.petsc -svd_nsv 3 -terse
      output_file: output/ex48_1.out
      test:
         suffix: 1
         args: -svd_type trlanczos -svd_trlanczos_explicitmatrix {{0 1}}
         TODO: does not work for largest singular values
      test:
         suffix: 1_spqr
         args: -svd_type trlanczos -svd_trlanczos_explicitmatrix -svd_trlanczos_pc_type qr
         requires: suitesparse
         TODO: does not work for largest singular values
      test:
         suffix: 1_cross
         args: -svd_type cross -svd_cross_explicitmatrix
      test:
         suffix: 1_cyclic
         args: -svd_type cyclic -svd_cyclic_explicitmatrix

   testset:
      requires: double complex datafilespath !defined(PETSC_USE_64BIT_INDICES)
      args: -f1 ${DATAFILESPATH}/matrices/complex/qc324.petsc -svd_nsv 3 -terse
      output_file: output/ex48_2.out
      filter: sed -e "s/30749/30748/"
      test:
         suffix: 2
         args: -svd_type trlanczos -svd_trlanczos_explicitmatrix {{0 1}} -svd_trlanczos_ksp_rtol 1e-10
      test:
         suffix: 2_spqr
         args: -svd_type trlanczos -svd_trlanczos_explicitmatrix -svd_trlanczos_pc_type qr -svd_trlanczos_ksp_rtol 1e-10
         requires: suitesparse
      test:
         suffix: 2_cross
         args: -svd_type cross -svd_cross_explicitmatrix
      test:
         suffix: 2_cyclic
         args: -svd_type cyclic -svd_cyclic_explicitmatrix

TEST*/
