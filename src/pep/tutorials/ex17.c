/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Solves a polynomial eigenproblem P(l)x = 0 with matrices loaded from a file.\n\n"
  "The command line options are:\n"
  "-A <filename1,filename2, ...> , where <filename1,.. > = matrices A0 ... files in PETSc binary form.\n\n";

#include <slepcpep.h>

#define MAX_MATRICES 40

int main(int argc,char **argv)
{
  Mat            A[MAX_MATRICES]; /* problem matrices */
  PEP            pep;             /* polynomial eigenproblem solver context */
  PetscReal      tol;
  PetscInt       nev,maxit,its,nmat=MAX_MATRICES,i;
  char*          filenames[MAX_MATRICES];
  PetscViewer    viewer;
  PetscBool      flg,terse;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Load the matrices that define the polynomial eigenproblem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nPolynomial eigenproblem stored in file.\n\n"));
#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Reading COMPLEX matrices from binary files...\n"));
#else
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Reading REAL matrices from binary files...\n"));
#endif
  CHKERRQ(PetscOptionsGetStringArray(NULL,NULL,"-A",filenames,&nmat,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate a comma-separated list of file names with the -A option");
  for (i=0;i<nmat;i++) {
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filenames[i],FILE_MODE_READ,&viewer));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A[i]));
    CHKERRQ(MatSetFromOptions(A[i]));
    CHKERRQ(MatLoad(A[i],viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create eigensolver context
  */
  CHKERRQ(PEPCreate(PETSC_COMM_WORLD,&pep));

  /*
     Set matrices
  */
  CHKERRQ(PEPSetOperators(pep,nmat,A));
  /*
     Set solver parameters at runtime
  */
  CHKERRQ(PEPSetFromOptions(pep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PEPSolve(pep));
  CHKERRQ(PEPGetIterationNumber(pep,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %" PetscInt_FMT "\n",its));

  /*
     Optional: Get some information from the solver and display it
  */
  CHKERRQ(PEPGetDimensions(pep,&nev,NULL,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %" PetscInt_FMT "\n",nev));
  CHKERRQ(PEPGetTolerances(pep,&tol,&maxit));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%" PetscInt_FMT "\n",(double)tol,maxit));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) {
    CHKERRQ(PEPErrorView(pep,PEP_ERROR_BACKWARD,NULL));
  } else {
    CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    CHKERRQ(PEPConvergedReasonView(pep,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PEPErrorView(pep,PEP_ERROR_BACKWARD,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(PEPDestroy(&pep));
  for (i=0;i<nmat;i++) {
    CHKERRQ(MatDestroy(&A[i]));
    CHKERRQ(PetscFree(filenames[i]));
  }
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      args: -A ${SLEPC_DIR}/share/slepc/datafiles/matrices/speaker107k.petsc,${SLEPC_DIR}/share/slepc/datafiles/matrices/speaker107c.petsc,${SLEPC_DIR}/share/slepc/datafiles/matrices/speaker107m.petsc -pep_type {{toar qarnoldi linear}} -pep_nev 4 -pep_ncv 20 -pep_scale scalar -terse
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES)

TEST*/
