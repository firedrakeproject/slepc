/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Illustrates the PGNHEP problem type. "
  "Based on ex7.\n"
  "The command line options are:\n"
  "  -f1 <filename> -f2 <filename>, PETSc binary files containing A and B.\n\n";

#include <slepceps.h>

int main(int argc,char **argv)
{
  EPS               eps;
  Mat               A,B;
  PetscBool         flg;
  PetscReal         tol=1000*PETSC_MACHINE_EPSILON;
  char              filename[PETSC_MAX_PATH_LEN];
  PetscViewer       viewer;
  PetscErrorCode    ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nPGNHEP problem loaded from file\n\n"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Load the matrices that define the eigensystem, Ax=kBx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f1",filename,sizeof(filename),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate a file name for matrix A with the -f1 option");

#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Reading COMPLEX matrices from binary files...\n"));
#else
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Reading REAL matrices from binary files...\n"));
#endif
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatLoad(A,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f2",filename,sizeof(filename),&flg));
  if (flg) {
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
    CHKERRQ(MatSetFromOptions(B));
    CHKERRQ(MatLoad(B,viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Matrix B was not provided, setting B=I\n\n"));
    B = NULL;
  }

  /* This example is intended for a matrix pair (A,B) where B is symmetric positive definite;
     If we load matrices bfw62a/bfw62b, scale both of them because bfw62b is negative definite */
  CHKERRQ(PetscStrendswith(filename,"bfw62b.petsc",&flg));
  if (flg) {
    CHKERRQ(MatScale(A,-1.0));
    CHKERRQ(MatScale(B,-1.0));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetOperators(eps,A,B));
  CHKERRQ(EPSSetProblemType(eps,EPS_PGNHEP));
  CHKERRQ(EPSSetTolerances(eps,tol,PETSC_DEFAULT));
  CHKERRQ(EPSSetFromOptions(eps));
  CHKERRQ(EPSSolve(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      args: -f1 ${SLEPC_DIR}/share/slepc/datafiles/matrices/bfw62a.petsc -f2 ${SLEPC_DIR}/share/slepc/datafiles/matrices/bfw62b.petsc -eps_largest_real -eps_nev 4
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES)
      output_file: output/test26_1.out
      test:
         args: -eps_true_residual {{0 1}}
         suffix: 1
      test:
         args: -eps_type arpack
         suffix: 1_arpack
         requires: arpack

   testset:
      args: -f1 ${DATAFILESPATH}/matrices/complex/mhd1280a.petsc -f2 ${DATAFILESPATH}/matrices/complex/mhd1280b.petsc -eps_smallest_real -eps_nev 4
      requires: double complex datafilespath !defined(PETSC_USE_64BIT_INDICES)
      output_file: output/test26_2.out
      test:
         suffix: 2
      test:
         args: -eps_type arpack
         suffix: 2_arpack
         requires: arpack

TEST*/
