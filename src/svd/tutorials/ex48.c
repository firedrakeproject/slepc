/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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
  PetscInt       i,m,n,p,Istart,Iend,col[2];
  PetscScalar    vals[] = { -1, 1 };
  char           filename[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscBool      flg,terse;

  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Load matrices that define the generalized singular value problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nGeneralized singular value problem stored in file.\n\n"));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f1",filename,sizeof(filename),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate a file name for matrix A with the -f1 option");

#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Reading COMPLEX matrix from a binary file...\n"));
#else
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Reading REAL matrix from a binary file...\n"));
#endif
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatLoad(A,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(MatGetSize(A,&m,&n));

  PetscCall(PetscOptionsGetString(NULL,NULL,"-f2",filename,sizeof(filename),&flg));
  if (flg) {
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
    PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
    PetscCall(MatSetFromOptions(B));
    PetscCall(MatLoad(B,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  } else {
    p = n+1;
    PetscCall(PetscOptionsGetInt(NULL,NULL,"-p",&p,&flg));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," Matrix B was not provided, setting B=bidiag(1,-1) with p=%" PetscInt_FMT "\n\n",p));
    PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
    PetscCall(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,p,n));
    PetscCall(MatSetFromOptions(B));
    PetscCall(MatSetUp(B));
    PetscCall(MatGetOwnershipRange(B,&Istart,&Iend));
    for (i=Istart;i<Iend;i++) {
      col[0]=i-1; col[1]=i;
      if (i==0) PetscCall(MatSetValue(B,i,col[1],vals[1],INSERT_VALUES));
      else if (i<n) PetscCall(MatSetValues(B,1,&i,2,col,vals,INSERT_VALUES));
      else if (i==n) PetscCall(MatSetValue(B,i,col[0],vals[0],INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Create the singular value solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create singular value solver context
  */
  PetscCall(SVDCreate(PETSC_COMM_WORLD,&svd));

  /*
     Set operators of GSVD problem
  */
  PetscCall(SVDSetOperators(svd,A,B));
  PetscCall(SVDSetProblemType(svd,SVD_GENERALIZED));

  /*
     Set solver parameters at runtime
  */
  PetscCall(SVDSetFromOptions(svd));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the problem and print solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(SVDSolve(svd));

  /* show detailed info unless -terse option is given by user */
  PetscCall(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) PetscCall(SVDErrorView(svd,SVD_ERROR_NORM,NULL));
  else {
    PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    PetscCall(SVDConvergedReasonView(svd,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(SVDErrorView(svd,SVD_ERROR_NORM,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(SVDDestroy(&svd));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(SlepcFinalize());
  return 0;
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
      timeoutfactor: 2
      test:
         suffix: 2
         args: -svd_type trlanczos -svd_trlanczos_explicitmatrix {{0 1}} -svd_trlanczos_ksp_rtol 1e-10
         requires: !defined(PETSCTEST_VALGRIND)
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

   test:
      requires: double complex datafilespath !defined(PETSC_USE_64BIT_INDICES) !defined(PETSCTEST_VALGRIND)
      args: -f1 ${DATAFILESPATH}/matrices/complex/qc324.petsc -p 320 -svd_nsv 3 -svd_type trlanczos -svd_trlanczos_ksp_rtol 1e-14 -terse
      timeoutfactor: 2
      suffix: 3

TEST*/
