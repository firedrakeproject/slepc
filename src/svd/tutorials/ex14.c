/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Solves a singular value problem with the matrix loaded from a file.\n"
  "This example works for both real and complex numbers.\n\n"
  "The command line options are:\n"
  "  -file <filename>, where <filename> = matrix file in PETSc binary form.\n\n";

#include <slepcsvd.h>

int main(int argc,char **argv)
{
  Mat            A;               /* operator matrix */
  SVD            svd;             /* singular value problem solver context */
  SVDType        type;
  PetscReal      tol;
  PetscInt       nsv,maxit,its;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscBool      flg,terse;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Load the operator matrix that defines the singular value problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nSingular value problem stored in file.\n\n"));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-file",filename,sizeof(filename),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate a file name with the -file option");

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

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the singular value solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create singular value solver context
  */
  PetscCall(SVDCreate(PETSC_COMM_WORLD,&svd));

  /*
     Set operator
  */
  PetscCall(SVDSetOperators(svd,A,NULL));

  /*
     Set solver parameters at runtime
  */
  PetscCall(SVDSetFromOptions(svd));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the singular value system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(SVDSolve(svd));
  PetscCall(SVDGetIterationNumber(svd,&its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %" PetscInt_FMT "\n",its));

  /*
     Optional: Get some information from the solver and display it
  */
  PetscCall(SVDGetType(svd,&type));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type));
  PetscCall(SVDGetDimensions(svd,&nsv,NULL,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Number of requested singular values: %" PetscInt_FMT "\n",nsv));
  PetscCall(SVDGetTolerances(svd,&tol,&maxit));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%" PetscInt_FMT "\n",(double)tol,maxit));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  PetscCall(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) PetscCall(SVDErrorView(svd,SVD_ERROR_RELATIVE,NULL));
  else {
    PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    PetscCall(SVDConvergedReasonView(svd,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(SVDErrorView(svd,SVD_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(SVDDestroy(&svd));
  PetscCall(MatDestroy(&A));
  PetscCall(SlepcFinalize());
  return 0;
}
/*TEST

   testset:
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -file ${SLEPC_DIR}/share/slepc/datafiles/matrices/rdb200.petsc -terse
      test:
         suffix: 1
         args: -svd_nsv 4 -svd_standard -svd_ncv 12 -svd_type {{trlanczos lanczos randomized cross}}
         filter: grep -v method
      test:
         suffix: 1_scalapack
         nsize: {{1 2 3}}
         args: -svd_nsv 4 -svd_type scalapack
         requires: scalapack
      test:
         suffix: 1_elemental
         nsize: {{1 2 3}}
         args: -svd_nsv 4 -svd_type elemental
         requires: elemental
      test:
         suffix: 2
         args: -svd_nsv 2 -svd_type cyclic -svd_cyclic_explicitmatrix -svd_cyclic_st_type sinvert -svd_cyclic_eps_target 12.5 -svd_cyclic_st_ksp_type preonly -svd_cyclic_st_pc_type lu -svd_view
         filter: grep -v tolerance
      test:
         suffix: 2_cross
         args: -svd_nsv 2 -svd_type cross -svd_cross_explicitmatrix -svd_cross_st_type sinvert -svd_cross_eps_target 100.0
         filter: grep -v tolerance

   testset:
      requires: double complex datafilespath !defined(PETSC_USE_64BIT_INDICES)
      args: -file ${DATAFILESPATH}/matrices/complex/qc324.petsc -terse
      test:
         suffix: 1_complex
         args: -svd_nsv 4
      test:
         suffix: 1_complex_scalapack
         nsize: {{1 2 3}}
         args: -svd_nsv 4 -svd_type scalapack
         requires: scalapack
      test:
         suffix: 1_complex_elemental
         nsize: {{1 2 3}}
         args: -svd_nsv 4 -svd_type elemental
         requires: elemental
      test:
         suffix: 2_complex
         args: -svd_nsv 2 -svd_type cyclic -svd_cyclic_explicitmatrix -svd_cyclic_st_type sinvert -svd_cyclic_eps_target 12.0 -svd_cyclic_st_ksp_type preonly -svd_cyclic_st_pc_type lu -svd_view
         filter: grep -v tolerance

   testset:
      args: -svd_nsv 5 -svd_type randomized -svd_max_it 1 -svd_conv_maxit
      test:
         suffix: 3
         args: -file ${SLEPC_DIR}/share/slepc/datafiles/matrices/rdb200.petsc
         requires: double !complex !defined(PETSC_USE_64BIT_INDICES)
      test:
         suffix: 3_complex
         args: -file ${DATAFILESPATH}/matrices/complex/qc324.petsc
         requires: double complex datafilespath !defined(PETSC_USE_64BIT_INDICES)
         filter: sed -e 's/[0-9][0-9]$//'

TEST*/
