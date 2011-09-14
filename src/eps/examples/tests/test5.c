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

static char help[] = "Tests different builds with a matrix loaded from a file.\n"
  "This test is based on ex4.c in tutorials.\n"
  "It loads test matrices available in PETSc's distribution.\n\n";

#include <slepceps.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            A;               /* operator matrix */
  EPS            eps;             /* eigenproblem solver context */
  PetscReal      error,re,im;
  PetscScalar    kr,ki;
  PetscInt       i,nconv;
  char           filename[PETSC_MAX_PATH_LEN];
  const char     *prefix,*scalar,*ints,*floats;
  PetscViewer    viewer;
  PetscBool      flg,symm;
  PetscErrorCode ierr;

  SlepcInitialize(&argc,&argv,(char*)0,help);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        Load the operator matrix that defines the eigensystem, Ax=kx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptionsHasName(PETSC_NULL,"-symm",&symm);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-herm",&flg);CHKERRQ(ierr);
  if (flg) symm=PETSC_TRUE;
#if defined(PETSC_USE_COMPLEX)
  prefix = symm? "hpd": "nh";
  scalar = "complex";
#else
  prefix = symm? "spd": "ns";
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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nReading matrix from a binary file...\n   %s\n",filename);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                     Create the eigensolver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);
  ierr = EPSSetOperators(eps,A,PETSC_NULL);CHKERRQ(ierr);
  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Solve the eigensystem and display solution 
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = EPSSolve(eps);CHKERRQ(ierr);
  ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);
  if (nconv>0) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "           k             ||Ax-kx||/||kx||\n"
         "  --------------------- ------------------\n");CHKERRQ(ierr);
    for (i=0;i<nconv;i++) {
      ierr = EPSGetEigenpair(eps,i,&kr,&ki,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
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
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
  }
  
  /* 
     Free work space
  */
  ierr = EPSDestroy(&eps);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = SlepcFinalize();CHKERRQ(ierr);
  return 0;
}

