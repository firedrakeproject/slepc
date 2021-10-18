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

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nGeneralized singular value problem stored in file.\n\n");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-f1",filename,sizeof(filename),&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate a file name for matrix A with the -f1 option");

#if defined(PETSC_USE_COMPLEX)
  ierr = PetscPrintf(PETSC_COMM_WORLD," Reading COMPLEX matrix from a binary file...\n");CHKERRQ(ierr);
#else
  ierr = PetscPrintf(PETSC_COMM_WORLD," Reading REAL matrix from a binary file...\n");CHKERRQ(ierr);
#endif
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);

  ierr = PetscOptionsGetString(NULL,NULL,"-f2",filename,sizeof(filename),&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
    ierr = MatSetFromOptions(B);CHKERRQ(ierr);
    ierr = MatLoad(B,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD," Matrix B was not provided, setting B=bidiag(1,-1)\n\n");CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
    ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n+1,n);CHKERRQ(ierr);
    ierr = MatSetFromOptions(B);CHKERRQ(ierr);
    ierr = MatSetUp(B);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(B,&Istart,&Iend);CHKERRQ(ierr);
    for (i=Istart;i<Iend;i++) {
      col[0]=i-1; col[1]=i;
      if (i==0) {
        ierr = MatSetValue(B,i,col[1],vals[1],INSERT_VALUES);CHKERRQ(ierr);
      } else if (i<n) {
        ierr = MatSetValues(B,1,&i,2,col,vals,INSERT_VALUES);CHKERRQ(ierr);
      } else if (i==n) {
        ierr = MatSetValue(B,i,col[0],vals[0],INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Create the singular value solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create singular value solver context
  */
  ierr = SVDCreate(PETSC_COMM_WORLD,&svd);CHKERRQ(ierr);

  /*
     Set operators of GSVD problem
  */
  ierr = SVDSetOperators(svd,A,B);CHKERRQ(ierr);
  ierr = SVDSetProblemType(svd,SVD_GENERALIZED);CHKERRQ(ierr);

  /*
     Set solver parameters at runtime
  */
  ierr = SVDSetFromOptions(svd);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the problem and print solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SVDSolve(svd);CHKERRQ(ierr);

  /* show detailed info unless -terse option is given by user */
  ierr = PetscOptionsHasName(NULL,NULL,"-terse",&terse);CHKERRQ(ierr);
  if (terse) {
    ierr = SVDErrorView(svd,SVD_ERROR_RELATIVE,NULL);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
    ierr = SVDConvergedReasonView(svd,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = SVDErrorView(svd,SVD_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = SVDDestroy(&svd);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
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
      test:
         suffix: 2
         args: -svd_type trlanczos -svd_tol 1e-10 -svd_trlanczos_explicitmatrix {{0 1}} -svd_trlanczos_ksp_rtol 1e-11
      test:
         suffix: 2_spqr
         args: -svd_type trlanczos -svd_tol 1e-10 -svd_trlanczos_explicitmatrix -svd_trlanczos_pc_type qr -svd_trlanczos_ksp_rtol 1e-11
         requires: suitesparse
      test:
         suffix: 2_cross
         args: -svd_type cross -svd_cross_explicitmatrix
      test:
         suffix: 2_cyclic
         args: -svd_type cyclic -svd_cyclic_explicitmatrix

TEST*/
