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

static char help[] = "Test SVD with different builds with a matrix loaded from a file"
  " (matrices available in PETSc's distribution).\n\n";

#include <slepcsvd.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            A;               /* operator matrix */
  SVD            svd;             /* singular value problem solver context */
  char           filename[PETSC_MAX_PATH_LEN];
  const char     *prefix,*scalar,*ints,*floats;
  PetscReal      tol=1000*PETSC_MACHINE_EPSILON;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  SlepcInitialize(&argc,&argv,(char*)0,help);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Load the matrix for which the SVD must be computed
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#if defined(PETSC_USE_COMPLEX)
  prefix = "nh";
  scalar = "complex";
#else
  prefix = "ns";
  scalar = "real";
#endif
#if defined(PETSC_USE_64BIT_INDICES)
  ints   = "int64";
#else
  ints   = "int32";
#endif
#if defined(PETSC_USE_REAL_DOUBLE)
  floats = "float64";
#elif defined(PETSC_USE_REAL_SINGLE)
  floats = "float32";
#endif

  ierr = PetscSNPrintf(filename,PETSC_MAX_PATH_LEN,"%s/share/petsc/datafiles/matrices/%s-%s-%s-%s",PETSC_DIR,prefix,scalar,ints,floats);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nReading matrix from binary file...\n\n");CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                     Create the SVD solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SVDCreate(PETSC_COMM_WORLD,&svd);CHKERRQ(ierr);
  ierr = SVDSetOperator(svd,A);CHKERRQ(ierr);
  ierr = SVDSetTolerances(svd,tol,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = SVDSetFromOptions(svd);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Solve the eigensystem and display solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SVDSolve(svd);CHKERRQ(ierr);
  ierr = SVDErrorView(svd,SVD_ERROR_RELATIVE,NULL);CHKERRQ(ierr);
  ierr = SVDDestroy(&svd);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}

