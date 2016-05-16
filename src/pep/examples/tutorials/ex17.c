/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.

   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Solves a polynomial eigenproblem P(l)x = 0 with matrices loaded from a file.\n\n"
  "The command line options are:\n"
  "-A <filename1,filename2, ...> , where <filename1,.. > = matrices A0 ... files in PETSc binary form.\n\n";

#include <slepcpep.h>

#define MAX_MATRICES 40

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            A[MAX_MATRICES]; /* problem matrices */
  PEP            pep;             /* polynomial eigenproblem solver context */
  PEPType        type;
  PetscReal      tol;
  PetscInt       nev,maxit,its,nmat=MAX_MATRICES,i;
  char*          filenames[MAX_MATRICES];
  PetscViewer    viewer;
  PetscBool      flg,terse;
  PetscErrorCode ierr;

  SlepcInitialize(&argc,&argv,(char*)0,help);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Load the matrices that define the polynomial eigenproblem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nPolynomial eigenproblem stored in file.\n\n");CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscPrintf(PETSC_COMM_WORLD," Reading COMPLEX matrices from binary files...\n");CHKERRQ(ierr);
#else
  ierr = PetscPrintf(PETSC_COMM_WORLD," Reading REAL matrices from binary files...\n");CHKERRQ(ierr);
#endif
  ierr = PetscOptionsGetStringArray(NULL,NULL,"-A",filenames,&nmat,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate a comma-separated list of file names with the -A option");
  for (i=0;i<nmat;i++) { 
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filenames[i],FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&A[i]);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A[i]);CHKERRQ(ierr);
    ierr = MatLoad(A[i],viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create eigensolver context
  */
  ierr = PEPCreate(PETSC_COMM_WORLD,&pep);CHKERRQ(ierr);

  /*
     Set matrices
  */
  ierr = PEPSetOperators(pep,nmat,A);CHKERRQ(ierr);
  /*
     Set solver parameters at runtime
  */
  ierr = PEPSetFromOptions(pep);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  
  ierr = PEPSolve(pep);CHKERRQ(ierr);
  ierr = PEPGetIterationNumber(pep,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);CHKERRQ(ierr);

  /*
     Optional: Get some information from the solver and display it
  */
  ierr = PEPGetType(pep,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
  ierr = PEPGetDimensions(pep,&nev,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRQ(ierr);
  ierr = PEPGetTolerances(pep,&tol,&maxit);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  ierr = PetscOptionsHasName(NULL,NULL,"-terse",&terse);CHKERRQ(ierr);
  if (terse) {
    ierr = PEPErrorView(pep,PEP_ERROR_BACKWARD,NULL);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
    ierr = PEPReasonView(pep,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PEPErrorView(pep,PEP_ERROR_BACKWARD,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = PEPDestroy(&pep);CHKERRQ(ierr);
  for (i=0;i<nmat;i++) {
    ierr = MatDestroy(&A[i]);CHKERRQ(ierr);
    ierr = PetscFree(filenames[i]);CHKERRQ(ierr);
  }
  ierr = SlepcFinalize();
  return ierr;
}
