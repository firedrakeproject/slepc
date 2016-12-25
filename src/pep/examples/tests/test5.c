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

static char help[] = "Test PEP view and monitor functionality.\n\n";

#include <slepcpep.h>

int main(int argc,char **argv)
{
  Mat            A[3];
  PEP            pep;
  PetscInt       n=6,Istart,Iend,i;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nPEP of diagonal problem, n=%D\n\n",n);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A[0]);CHKERRQ(ierr);
  ierr = MatSetSizes(A[0],PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A[0]);CHKERRQ(ierr);
  ierr = MatSetUp(A[0]);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A[0],&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    ierr = MatSetValue(A[0],i,i,i+1,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A[0],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A[0],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A[1]);CHKERRQ(ierr);
  ierr = MatSetSizes(A[1],PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A[1]);CHKERRQ(ierr);
  ierr = MatSetUp(A[1]);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    ierr = MatSetValue(A[1],i,i,-1.5,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A[1],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A[1],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A[2]);CHKERRQ(ierr);
  ierr = MatSetSizes(A[2],PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A[2]);CHKERRQ(ierr);
  ierr = MatSetUp(A[2]);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    ierr = MatSetValue(A[2],i,i,-1.0/(i+1),INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A[2],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A[2],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                     Create the PEP solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PEPCreate(PETSC_COMM_WORLD,&pep);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)pep,"pep");CHKERRQ(ierr);
  ierr = PEPSetOperators(pep,3,A);CHKERRQ(ierr);
  ierr = PEPSetFromOptions(pep);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Solve the eigensystem and display solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PEPSolve(pep);CHKERRQ(ierr);
  ierr = PEPErrorView(pep,PEP_ERROR_RELATIVE,NULL);CHKERRQ(ierr);
  ierr = PEPDestroy(&pep);CHKERRQ(ierr);
  ierr = MatDestroy(&A[0]);CHKERRQ(ierr);
  ierr = MatDestroy(&A[1]);CHKERRQ(ierr);
  ierr = MatDestroy(&A[2]);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}

