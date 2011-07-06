/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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

static char help[] = "Solves a generalized eigensystem Ax=kBx with matrices loaded from a file.\n"
  "This example works for both real and complex numbers.\n\n"
  "The command line options are:\n"
  "  -f1 <filename>, where <filename> = matrix (A) file in PETSc binary form.\n"
  "  -f2 <filename>, where <filename> = matrix (B) file in PETSc binary form.\n"
  "  -ninitial <nini>, number of user-provided initial guesses.\n"
  "  -finitial <filename>, where <filename> contains <nini> vectors (binary).\n"
  "  -nconstr <ncon>, number of user-provided constraints.\n"
  "  -fconstr <filename>, where <filename> contains <ncon> vectors (binary).\n\n";

#include <slepceps.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            A,B;             /* matrices */
  EPS            eps;             /* eigenproblem solver context */
  const EPSType  type;
  PetscReal      error,tol,re,im;
  PetscScalar    kr,ki;
  Vec            xr,xi,*is,*ds;
  PetscInt       nev,maxit,i,its,lits,nconv,nini=0,ncon=0;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscBool      flg,evecs,ishermitian;
  PetscErrorCode ierr;

  SlepcInitialize(&argc,&argv,(char*)0,help);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        Load the matrices that define the eigensystem, Ax=kBx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nGeneralized eigenproblem stored in file.\n\n");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(PETSC_NULL,"-f1",filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) {
    SETERRQ(PETSC_COMM_WORLD,1,"Must indicate a file name for matrix A with the -f1 option.");
  }

#if defined(PETSC_USE_COMPLEX)
  ierr = PetscPrintf(PETSC_COMM_WORLD," Reading COMPLEX matrices from binary files...\n");CHKERRQ(ierr);
#else
  ierr = PetscPrintf(PETSC_COMM_WORLD," Reading REAL matrices from binary files...\n");CHKERRQ(ierr);
#endif
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscOptionsGetString(PETSC_NULL,"-f2",filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) {
    SETERRQ(PETSC_COMM_WORLD,1,"Must indicate a file name for matrix B with the -f2 option.");
  }

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatLoad(B,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = MatGetVecs(A,PETSC_NULL,&xr);CHKERRQ(ierr);
  ierr = MatGetVecs(A,PETSC_NULL,&xi);CHKERRQ(ierr);

  /* 
     Read user constraints if available
  */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-nconstr",&ncon,&flg);CHKERRQ(ierr);
  if (flg) {
    if (ncon<=0) SETERRQ(PETSC_COMM_WORLD,1,"The number of constraints must be >0");
    ierr = PetscOptionsGetString(PETSC_NULL,"-fconstr",filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must specify the name of the file storing the constraints");
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(xr,ncon,&ds);CHKERRQ(ierr);
    for (i=0;i<ncon;i++) {
      ierr = VecLoad(ds[i],viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  /* 
     Read initial guesses if available
  */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-ninitial",&nini,&flg);CHKERRQ(ierr);
  if (flg) {
    if (nini<=0) SETERRQ(PETSC_COMM_WORLD,1,"The number of initial vectors must be >0");
    ierr = PetscOptionsGetString(PETSC_NULL,"-finitial",filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must specify the name of the file containing the initial vectors");
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(xr,nini,&is);CHKERRQ(ierr);
    for (i=0;i<nini;i++) {
      ierr = VecLoad(is[i],viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Create eigensolver context
  */
  ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);

  /* 
     Set operators. In this case, it is a generalized eigenvalue problem
  */
  ierr = EPSSetOperators(eps,A,B);CHKERRQ(ierr);

  /* 
     If the user provided initial guesses or constraints, pass them here
  */
  ierr = EPSSetInitialSpace(eps,nini,is);CHKERRQ(ierr);
  ierr = EPSSetDeflationSpace(eps,ncon,ds);CHKERRQ(ierr);

  /*
     Set solver parameters at runtime
  */
  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSSolve(eps);CHKERRQ(ierr);

  /*
     Optional: Get some information from the solver and display it
  */
  ierr = EPSGetIterationNumber(eps,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %d\n",its);CHKERRQ(ierr);
  ierr = EPSGetOperationCounters(eps,PETSC_NULL,PETSC_NULL,&lits);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of linear iterations of the method: %d\n",lits);CHKERRQ(ierr);
  ierr = EPSGetType(eps,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
  ierr = EPSGetDimensions(eps,&nev,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %d\n",nev);CHKERRQ(ierr);
  ierr = EPSGetTolerances(eps,&tol,&maxit);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%d\n",tol,maxit);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Get number of converged eigenpairs
  */
  ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged approximate eigenpairs: %d\n\n",nconv);CHKERRQ(ierr);

  if (nconv>0) {
    /*
       Open file to save eigenvectors, if requested
    */
    ierr = PetscOptionsGetString(PETSC_NULL,"-evecs",filename,PETSC_MAX_PATH_LEN,&evecs);CHKERRQ(ierr);
    if (evecs) {
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
      ierr = EPSIsHermitian(eps,&ishermitian);CHKERRQ(ierr);
    }

    /*
       Display eigenvalues and relative errors
    */
    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "           k             ||Ax-kBx||/||kx||\n"
         "  --------------------- ------------------\n");CHKERRQ(ierr);
    for (i=0;i<nconv;i++) {
      /* 
         Get converged eigenpairs: i-th eigenvalue is stored in kr (real part) and
         ki (imaginary part)
      */
      ierr = EPSGetEigenpair(eps,i,&kr,&ki,xr,xi);CHKERRQ(ierr);
      if (evecs) {
        ierr = VecView(xr,viewer);CHKERRQ(ierr);
        if (!ishermitian) { ierr = VecView(xi,viewer);CHKERRQ(ierr); }
      }

      /*
         Compute the relative error associated to each eigenpair
      */
      ierr = EPSComputeRelativeError(eps,i,&error);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX)
      re = PetscRealPart(kr);
      im = PetscImaginaryPart(kr);
#else
      re = kr;
      im = ki;
#endif
      if (im != 0.0) {
        ierr = PetscPrintf(PETSC_COMM_WORLD," % 6f %+6f i",re,im);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"       % 6f      ",re);CHKERRQ(ierr);
      }
      ierr = PetscPrintf(PETSC_COMM_WORLD," % 12g\n",error);CHKERRQ(ierr);
    }
    if (evecs) {
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
  }
  
  /* 
     Free work space
  */
  ierr = EPSDestroy(&eps);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = VecDestroy(&xr);CHKERRQ(ierr);
  ierr = VecDestroy(&xi);CHKERRQ(ierr);
  if (nini>0) { ierr = VecDestroyVecs(nini,&is);CHKERRQ(ierr); }
  if (ncon>0) { ierr = VecDestroyVecs(ncon,&ds);CHKERRQ(ierr); }
  ierr = SlepcFinalize();CHKERRQ(ierr);
  return 0;
}

